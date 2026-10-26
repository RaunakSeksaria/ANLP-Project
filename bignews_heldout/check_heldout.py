"""Which BIGNEWS articles did POLITICS never train on?

The public POLITICS release (Liu et al. 2022) has no validation files. It has three corpora:
    BIGNEWS/train_<outlet>.json            all 3.69M cleaned articles, one file per outlet
    BIGNEWSBLN/BIGNEWSBLN_{left,center,right}.json   the 2.33M downsampled articles MLM trained on
    BIGNEWSALIGN/BIGNEWSALIGN.json         story clusters the triplet loss trained on (built from BLN)
So BIGNEWS minus BIGNEWSBLN (~1.36M articles) was never a training input. This script:
    1. collects the key of every trained article (BLN + ALIGN)
    2. from each BIGNEWS outlet, samples up to --per-outlet articles whose key is not trained
    3. drops any sampled article that is an exact or near copy of trained text
       (wire copy, re-posts and updates recur across the corpus)

Matching, all case- and punctuation-insensitive:
    key     (source, date, title)       how POLITICS itself identifies an article
    exact   hash of the full text
    near    Jaccard of word 5-gram sets >= --jaccard (default 0.8, same as the team's dedup),
            over the first --max-words words (POLITICS truncated inputs at 512 tokens)

Standard library only, so it runs on an Ada compute node without a venv.

    python3 check_heldout.py --bignews DIR --bln DIR --align FILE --out DIR [--per-outlet 5000]
    python3 check_heldout.py --selftest
"""
import argparse, hashlib, json, os, re, sys, time
import multiprocessing as mp
from pathlib import Path

WORD = re.compile(r"[a-z0-9]+")
SIDES = ("left", "center", "right")
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def body(rec):
    t = rec.get("text", "")
    return " ".join(t) if isinstance(t, list) else str(t)


def words(s):
    return WORD.findall(s.lower())


def key_of(rec):
    return (str(rec.get("source", "")).lower(), str(rec.get("date", ""))[:10], " ".join(words(rec.get("title", ""))))


def text_hash(rec):
    return hashlib.blake2b(" ".join(words(body(rec))).encode(), digest_size=12).hexdigest()


def shingles(rec, n, max_words, sample):
    w = words(rec.get("title", "") + " " + body(rec))[:max_words]
    out = set()
    for i in range(len(w) - n + 1):
        h = hash(" ".join(w[i:i + n]))
        if h % sample == 0:                     # keep 1/sample of shingles on both sides (unbiased Jaccard estimate)
            out.add(h)
    return out


def stream_items(path, bufsize=1 << 24):
    """Yield the elements of a top-level JSON array one at a time (the release's files are up to 19 GB,
    far too big to json.load). Falls back to json.load if the top level is not an array."""
    dec, ws = json.JSONDecoder(), " \t\r\n,"
    with open(path, encoding="utf-8") as f:
        buf, pos = f.read(bufsize), 0
        while pos < len(buf) and buf[pos] in " \t\r\n":
            pos += 1
        if not buf[pos:pos + 1] == "[":
            f.seek(0); data = json.load(f)
            yield from (data.values() if isinstance(data, dict) else [data]); return
        pos += 1
        eof = False
        while True:
            while pos < len(buf) and buf[pos] in ws:
                pos += 1
            if pos >= len(buf) - 1 and not eof:          # refill
                more = f.read(bufsize); eof = not more
                buf, pos = buf[pos:] + more, 0
                continue
            if pos < len(buf) and buf[pos] == "]":
                return
            try:
                obj, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                if eof:
                    raise
                more = f.read(bufsize); eof = not more
                buf, pos = buf[pos:] + more, 0
                continue
            yield obj
            pos = end
            if pos > bufsize:                             # keep the buffer bounded
                buf, pos = buf[pos:], 0


def iter_articles(path):
    """Yield article records from a list of articles, a list of clusters, or a dict of either."""
    for item in stream_items(path):
        if isinstance(item, list):
            yield from (x for x in item if isinstance(x, dict))
        elif isinstance(item, dict):
            yield item


# ---------------------------------------------------------------- worker state (set before fork)
G = {}


def scan_chunk(args):
    """Return, for one chunk of training articles, which candidates each one exactly or nearly matches."""
    recs, = args
    keys, hashes, inv, csize = G["keys"], G["hashes"], G["inv"], G["csize"]
    n, mw, sm, thr = G["n"], G["max_words"], G["sample"], G["jaccard"]
    hit_key, hit_exact, hit_near = set(), set(), {}
    for r in recs:
        c = keys.get(key_of(r))
        if c is not None:
            hit_key.update(c)
        c = hashes.get(text_hash(r))
        if c is not None:
            hit_exact.update(c)
        s = shingles(r, n, mw, sm) - G["common"]
        if not s:
            continue
        counts = {}
        for h in s:
            for cid in inv.get(h, ()):
                counts[cid] = counts.get(cid, 0) + 1
        for cid, k in counts.items():
            j = k / max(1, len(s) + csize[cid] - k)
            if j >= thr and j > hit_near.get(cid, 0):
                hit_near[cid] = j
    return hit_key, hit_exact, hit_near


def chunks(it, size):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == size:
            yield (buf,); buf = []
    if buf:
        yield (buf,)


IDEO = {"dailykos": 0, "hpo": 0, "cnn": 0, "wpo": 0, "nyt": 0, "usatoday": 1, "ap": 1, "hill": 1,
        "wat": 2, "fox": 2, "breitbart": 2}


def main(a):
    import random
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    trained_files = sorted(Path(a.bln).glob("*.json")) + ([Path(a.align)] if a.align else [])
    bignews_files = sorted(Path(a.bignews).glob("*.json"))
    if not trained_files or not bignews_files:
        sys.exit(f"no files: trained={trained_files} bignews={bignews_files}")

    # ---- 1. keys of every trained article
    trained = set(); n_trained = {}
    for f in trained_files:
        n = 0
        for r in iter_articles(f):
            trained.add(key_of(r)); n += 1
        n_trained[f.name] = n
        log(f"trained keys from {f.name}: {n} articles ({len(trained)} distinct keys so far)")

    # ---- 2. sample untrained articles per outlet
    rng = random.Random(a.seed); cand = []; per_outlet = {}
    for f in bignews_files:
        tot, elig = 0, []
        for r in iter_articles(f):
            tot += 1
            if key_of(r) not in trained:
                elig.append(r)
        pick = elig if len(elig) <= a.per_outlet else rng.sample(elig, a.per_outlet)
        outlet = str(pick[0].get("source", f.stem.replace("train_", ""))).lower() if pick else f.stem
        for r in pick:
            cand.append({"source": r.get("source"), "date": r.get("date"), "title": r.get("title"),
                         "text": r.get("text"), "label": IDEO.get(str(r.get("source", "")).lower())})
        per_outlet[f.name] = {"articles": tot, "untrained_by_key": len(elig), "sampled": len(pick)}
        log(f"{f.name}: {tot} articles, {len(elig)} untrained by key, sampled {len(pick)}")
        del elig
    del trained

    # ---- 3. exact / near-copy scan of the sample against all trained text
    keys, hashes, inv, csize = {}, {}, {}, []
    for i, r in enumerate(cand):
        keys.setdefault(key_of(r), []).append(i)
        hashes.setdefault(text_hash(r), []).append(i)
        sh = shingles(r, a.n, a.max_words, a.sample)
        csize.append(len(sh))
        for h in sh:
            inv.setdefault(h, []).append(i)
    # Boilerplate shingles shared by many candidates ("according to a statement") carry no copy signal and
    # make buckets huge; drop them from both sides so the Jaccard estimate stays consistent.
    common = {h for h, v in inv.items() if len(v) > a.max_bucket}
    for h in common:
        for i in inv.pop(h):
            csize[i] -= 1
    G.update(keys=keys, hashes=hashes, inv=inv, csize=csize, common=common, n=a.n, max_words=a.max_words,
             sample=a.sample, jaccard=a.jaccard)
    log(f"index built: {len(cand)} candidates, {len(inv)} shingle buckets, {len(common)} boilerplate dropped")

    hit_key, hit_exact, hit_near = set(), set(), {}
    with mp.get_context("fork").Pool(a.workers) as pool:   # fork: workers share the index copy-on-write
        for tf in trained_files:
            for hk, he, hn in pool.imap_unordered(scan_chunk, chunks(iter_articles(tf), a.chunk)):
                hit_key |= hk; hit_exact |= he
                for cid, j in hn.items():
                    hit_near[cid] = max(j, hit_near.get(cid, 0))
            log(f"scanned {tf.name} | flagged so far key={len(hit_key)} exact={len(hit_exact)} near={len(hit_near)}")

    # ---- write results
    flagged = hit_key | hit_exact | set(hit_near)
    by_outlet = {}
    with open(out / "clean_heldout.jsonl", "w") as fc, open(out / "flagged.jsonl", "w") as ff:
        for i, r in enumerate(cand):
            src = str(r.get("source", "")).lower()
            d = by_outlet.setdefault(src, {"sampled": 0, "clean": 0}); d["sampled"] += 1
            if i in flagged:
                why = [w for w, st in (("key", hit_key), ("exact", hit_exact)) if i in st]
                if i in hit_near:
                    why.append(f"near:{hit_near[i]:.3f}")
                ff.write(json.dumps({**r, "why": why}) + "\n")
            else:
                d["clean"] += 1
                fc.write(json.dumps(r) + "\n")
    report = {"trained_articles": n_trained, "bignews": per_outlet, "candidates": len(cand),
              "clean": len(cand) - len(flagged), "flagged": len(flagged),
              "flagged_by": {"key": len(hit_key), "exact": len(hit_exact), "near": len(hit_near)},
              "by_outlet": by_outlet,
              "params": {k: getattr(a, k) for k in ("per_outlet", "seed", "n", "max_words", "sample", "jaccard", "max_bucket")},
              "seconds": round(time.time() - T0)}
    json.dump(report, open(out / "report.json", "w"), indent=2)
    log(json.dumps(report))


# ---------------------------------------------------------------- self-test on synthetic release
def selftest():
    import random, tempfile
    rng = random.Random(0)
    vocab = [f"w{i}" for i in range(5000)]
    art = lambda src, k: {"source": src, "date": f"2019-01-{rng.randint(10, 28)}", "title": f"title {k} " + " ".join(rng.choices(vocab, k=6)),
                          "text": [" ".join(rng.choices(vocab, k=60)) for _ in range(5)]}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for d in ("BIGNEWS", "BIGNEWSBLN", "BIGNEWSALIGN"):
            (td / d).mkdir()
        big = {o: [art(o, f"{o}{i}") for i in range(100)] for o in ("cnn", "hill", "fox")}
        # BLN = 60 of each outlet's 100 articles; the other 40 are untrained
        bln = {"left": big["cnn"][:60], "center": big["hill"][:60], "right": big["fox"][:60]}
        untrained = {o: v[60:] for o, v in big.items()}
        # plant copies of untrained articles inside trained text:
        u = untrained["cnn"]
        bln["left"].append({**u[0], "title": "renamed", "date": "2020-01-01"})          # exact text, new key
        t = [x.split() for x in u[1]["text"]]; t[0][3] = "CHANGED"; t[2][9] = "EDITED"
        bln["right"].append({**u[1], "title": "x", "date": "2021-02-02", "text": [" ".join(x) for x in t]})  # near copy
        align = [[big["cnn"][0], big["fox"][0]], [dict(untrained["fox"][0], title="align-only copy", date="2018-01-01")]]
        for o, v in big.items():
            json.dump(v, open(td / "BIGNEWS" / f"train_{o}.json", "w"))
        for s_, v in bln.items():
            json.dump(v, open(td / "BIGNEWSBLN" / f"BIGNEWSBLN_{s_}.json", "w"))
        json.dump(align, open(td / "BIGNEWSALIGN" / "BIGNEWSALIGN.json", "w"))
        ns = argparse.Namespace(bignews=str(td / "BIGNEWS"), bln=str(td / "BIGNEWSBLN"),
                                align=str(td / "BIGNEWSALIGN" / "BIGNEWSALIGN.json"), out=str(td / "out"),
                                per_outlet=1000, seed=0, workers=4, chunk=50, n=5, max_words=400, sample=1,
                                jaccard=0.8, max_bucket=50)
        main(ns)
        rep = json.load(open(td / "out" / "report.json"))
        flagged = {json.loads(l)["title"] for l in open(td / "out" / "flagged.jsonl")}
        want = {u[0]["title"], u[1]["title"], untrained["fox"][0]["title"]}
        assert rep["candidates"] == 120, rep                     # 3 outlets x 40 untrained
        assert flagged == want, sorted(flagged ^ want)
        assert rep["clean"] == 117, rep
        labels = {json.loads(l)["label"] for l in open(td / "out" / "clean_heldout.jsonl")}
        assert labels == {0, 1, 2}, labels
        print("SELFTEST OK:", rep["flagged_by"], rep["by_outlet"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bignews"); ap.add_argument("--bln"); ap.add_argument("--align"); ap.add_argument("--out")
    ap.add_argument("--per-outlet", type=int, default=5000, help="untrained articles to sample per outlet")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--n", type=int, default=5, help="word n-gram size")
    ap.add_argument("--max-words", type=int, default=400, help="~512 RoBERTa tokens")
    ap.add_argument("--sample", type=int, default=4, help="keep 1/sample shingles (memory/speed)")
    ap.add_argument("--jaccard", type=float, default=0.8)
    ap.add_argument("--max-bucket", type=int, default=50, help="ignore shingles shared by more candidates")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else main(a)
