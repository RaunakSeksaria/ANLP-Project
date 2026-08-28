"""Profile every corpus under data/ and print what each can and cannot supply.

An event-aware triplet objective needs three labels on the same example: an
*event* grouping, a *publisher* identity, and a *neutral summary* target. Only
one corpus here carries all three. This script measures, per corpus, which axis
is missing and how expensive it would be to manufacture.

    uv run python eda/profile_datasets.py                 # everything
    uv run python eda/profile_datasets.py neus compare    # selected sections

CPU only, no network. Sections are independent; each prints a MISSING line
naming the axis it cannot supply.
"""

import json
import re
import statistics as st
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
NEUS = DATA / "framing-bias-metric-main/data"

STOP = set(
    "the a an of in on to for and or is are was were with at by from as that this "
    "his her its it be has have new say says said after over amid".split()
)
ISO = re.compile(r"\d{4}-\d{2}-\d{2}")
# "TITLE=> <headline>. ARTICLE=> <body>" -- greedy so headlines containing a
# period survive; the preprocessing appends its own '.' after the title.
PROBE_ARTICLE = re.compile(r"TITLE=>(.*)\.\s*ARTICLE=>(.*)$", re.S)


def title_toks(t):
    return {w for w in re.findall(r"[a-z']+", t.lower()) if w not in STOP and len(w) > 2}


def norm(s):
    """Aggressive key for joining texts across preprocessing steps."""
    return re.sub(r"\W+", "", unicodedata.normalize("NFKD", s or "").lower())


def pct(xs, p):
    return sorted(xs)[int(p * len(xs))]


def spread(xs, label, unit="words"):
    return f"{label} {unit}: median={st.median(xs):.0f} p90={pct(xs, 0.9)} max={max(xs)}"


# --------------------------------------------------------------------------- #
# AllSides / Article-Bias-Prediction
# --------------------------------------------------------------------------- #

def load_allsides():
    for f in (DATA / "Article-Bias-Prediction/data/jsons").glob("*.json"):
        yield json.loads(f.read_text())


def mine_event_clusters(recs, jaccard=0.5):
    """Group articles sharing a topic, a publication date, and >=50% of their
    title's content words. This is the only route to event groupings in
    AllSides -- it ships 108 broad topic labels, not story ids."""
    buckets = defaultdict(list)
    for r in recs:
        if ISO.fullmatch(r.get("date") or ""):
            buckets[(r["topic"], r["date"])].append(r)

    clusters = []
    for group in buckets.values():
        if len(group) < 2:
            continue
        used = set()
        for i, a in enumerate(group):
            if i in used:
                continue
            ta = title_toks(a["title"])
            cluster, used_now = [a], {i}
            for j, b in enumerate(group):
                if j in used or j == i:
                    continue
                tb = title_toks(b["title"])
                if ta and tb and len(ta & tb) / min(len(ta), len(tb)) >= jaccard:
                    cluster.append(b)
                    used_now.add(j)
            used |= used_now
            if len(cluster) > 1:
                clusters.append(cluster)
    return clusters


def allsides():
    print("\n=== AllSides / Article-Bias-Prediction (Baly et al. 2020) ===")
    recs = list(load_allsides())
    print(f"articles: {len(recs)}   sources: {len({r['source'] for r in recs})}   "
          f"topics: {len({r['topic'] for r in recs})}")
    print("bias:", dict(Counter(r["bias_text"] for r in recs)))

    src_bias = defaultdict(set)
    for r in recs:
        src_bias[r["source"]].add(r["bias_text"])
    mixed = sum(1 for v in src_bias.values() if len(v) > 1)
    print(f"sources carrying >1 bias label: {mixed}/{len(src_bias)}"
          f"  -> bias is a PUBLISHER attribute, not an article annotation")

    dates = [r.get("date") or "" for r in recs]
    iso = sum(1 for d in dates if ISO.fullmatch(d))
    empty = sum(1 for d in dates if not d)
    print(f"dates: iso={iso} missing={empty} other-format={len(dates) - iso - empty}")

    print(spread([len(r["content"].split()) for r in recs], "content"),
          " (~x1.3 for BPE -> most articles exceed BART's 1024)")

    clusters = mine_event_clusters(recs)
    multi = [c for c in clusters if len({a["source"] for a in c}) >= 2]
    tri = [c for c in clusters if len({a["bias_text"] for a in c}) == 3]
    covered = sum(len(c) for c in multi)
    print(f"mined event clusters: {len(clusters)}  multi-source: {len(multi)}  "
          f"all-three-leanings: {len(tri)}")
    print(f"articles inside multi-source clusters: {covered} ({100 * covered / len(recs):.1f}% of corpus)")

    for name in ("random", "media"):
        counts = {
            p.stem: sum(1 for _ in p.open()) - 1
            for p in sorted((DATA / f"Article-Bias-Prediction/data/splits/{name}").glob("*.tsv"))
        }
        print(f"{name} split: {counts}")
    print("MISSING: neutral summary targets. Event groupings must be mined and are thin.")


# --------------------------------------------------------------------------- #
# WCEP-10
# --------------------------------------------------------------------------- #

PUBLISHER_CUES = {
    "reuters": r"\bReuters\b", "ap": r"\bAssociated Press\b|\(AP\)", "afp": r"\bAFP\b",
    "cnn": r"\bCNN\b", "bbc": r"\bBBC\b", "nyt": r"New York Times", "fox": r"Fox News",
    "guardian": r"This article is more than|theguardian\.com", "xinhua": r"Xinhua",
    "aljazeera": r"Al Jazeera", "pti": r"\bPTI\b",
    "boilerplate": r"Image copyright|Media playback|All rights reserved|Sign up to|newsletter",
}


def wcep():
    print("\n=== WCEP-10 (Ghalandari et al. 2020, PRIMERA preprocessing) ===")
    cues = {k: re.compile(v) for k, v in PUBLISHER_CUES.items()}
    per_cue = Counter()

    for split in ("train", "val", "test"):
        n = 0
        ndocs, dlen, slen = [], [], []
        leaky = ndoc = 0
        with (DATA / f"WCEP-10/{split}.txt").open() as f:
            for line in f:
                r = json.loads(line)
                n += 1
                ndocs.append(len(r["document"]))
                slen.append(len(r["summary"].split()))
                for d in r["document"]:
                    ndoc += 1
                    dlen.append(len(d.split()))
                    matched = [k for k, p in cues.items() if p.search(d)]
                    per_cue.update(matched)
                    leaky += bool(matched)
        print(f"{split:5} clusters={n:5} docs/cluster={st.mean(ndocs):.1f} "
              f"{spread(dlen, 'doc')} | summary words median={st.median(slen):.0f}")
        print(f"      docs naming a publisher in-text: {leaky}/{ndoc} ({100 * leaky / ndoc:.1f}%)"
              f"  <- both a silver-label route and a leakage confound")
    print("in-text cues overall:", per_cue.most_common(6))
    print("MISSING: publisher labels -- dropped by PRIMERA's preprocessing; the raw WCEP")
    print("         release keeps source URLs. Targets are 25-word current-events blurbs,")
    print("         not neutral summaries: nothing in them was written to counter framing.")


# --------------------------------------------------------------------------- #
# WNC
# --------------------------------------------------------------------------- #

def wnc():
    print("\n=== WNC / neutralising-bias (Pryzant et al. 2020) ===")
    base = DATA / "neutralising-bias/WNC"
    for name in ("biased.word.train", "biased.word.dev", "biased.word.test", "biased.full", "neutral"):
        n = sum(1 for _ in (base / name).open(encoding="utf-8", errors="replace"))
        print(f"  {name:20} {n:7} rows")
    print("format: TSV  id / src_tok / tgt_tok / src_raw / tgt_raw / src_POS / tgt_parse")
    print("Wikipedia NPOV revisions: lowercased, WordPiece-split (##), sentence-level, single-source.")

    rw = DATA / "neutralising-bias/real_world_samples"
    print("real_world_samples:",
          ", ".join(f"{p.name}({sum(1 for _ in p.open())})" for p in sorted(rw.iterdir())))
    print("MISSING: events, publishers, multi-document structure. Domain is encyclopedic,")
    print("         not news -- shares no structure with the multi-source task.")


# --------------------------------------------------------------------------- #
# NeuS / framing-bias-metric
# --------------------------------------------------------------------------- #

RAW_CRAWLS = (
    "raw_crawled/all_crawled_combined_filtered_RoundupLeftCenterRight.jsonl",
    "raw_crawled/all_crawled_combined.jsonl",
    "raw_crawled/headline_11_2021-10-19_crawled_all.jsonl",
)
PROBE = "naacl2022_lrc_roundup_random_order_probe"


def sentences(seq):
    """newBody and roundup are [[idx, sentence], ...]."""
    return " ".join(s for _, s in seq)


def load_neus_raw(which=0):
    return [json.loads(l) for l in (NEUS / RAW_CRAWLS[which]).open()]


def neus_title_index():
    """newsTitle -> (story_id, stance, publisher, source_file), over every crawl file.

    The filtered file alone covers only ~87% of the released train/val/test
    examples; the headline crawl supplies the rest, and the union reaches 100%.
    """
    idx = {}
    for fn in RAW_CRAWLS:
        for r in (json.loads(l) for l in (NEUS / fn).open()):
            for stance, v in (r.get("news") or {}).items():
                if isinstance(v, dict) and v.get("newsTitle"):
                    idx.setdefault(norm(v["newsTitle"]), []).append(
                        (r["id"], stance, v["newsSource"], Path(fn).name)
                    )
    return idx


def read_probe(split):
    """Yield (articles, target) where articles is a list of (title, body)."""
    src = (NEUS / PROBE / f"{split}.source").read_text().splitlines()
    tgt = (NEUS / PROBE / f"{split}.target").read_text().splitlines()
    for s, t in zip(src, tgt):
        arts = []
        for part in s.split("[SEP]"):
            m = PROBE_ARTICLE.search(part)
            if m:
                arts.append((m.group(1).strip(), m.group(2).strip()))
        yield arts, t


def neus():
    print("\n=== NeuS / framing-bias-metric (Lee et al. 2022, NAACL) ===")
    rows = load_neus_raw(0)
    print(f"stories: {len(rows)}  unique ids: {len({r['id'] for r in rows})}")

    pubs = Counter()
    pub_events = defaultdict(set)
    spec = Counter()
    dates = []
    for r in rows:
        dates.append(r["date"][:10])
        for v in r["news"].values():
            pubs[v["newsSource"]] += 1
            pub_events[v["newsSource"]].add(r["id"])
            spec[v["stance_specific"]] += 1
    print(f"publishers: {len(pubs)}   date range: {min(dates)} -> {max(dates)}")
    print("stance_specific:", dict(spec.most_common()))
    print("top publishers:", pubs.most_common(8))

    reusable = {p: len(e) for p, e in pub_events.items() if len(e) >= 2}
    slots = sum(reusable.values())
    print(f"publishers covering >=2 stories: {len(reusable)}/{len(pub_events)}  -> "
          f"{slots}/{3 * len(rows)} article slots can serve as a same-publisher, "
          f"different-event NEGATIVE")

    body = [len(sentences(v["newBody"]).split()) for r in rows for v in r["news"].values()]
    print(spread(body, "article body"),
          "  <- AllSides publishes an EXCERPT, not the article; hard cap near 95")
    print(spread([len(sentences(r["roundup"]).split()) for r in rows], "roundup"))
    print(spread([sum(len(sentences(v["newBody"]).split()) for v in r["news"].values()) for r in rows],
                 "3-article concat"), " (~x1.3 BPE -> comfortably inside BART's 1024)")

    # ---- released splits, joined back to publisher metadata ----
    print()
    idx = neus_title_index()
    split_pubs = {}
    for split in ("train", "val", "test"):
        n = matched = 0
        stories, seen, origin = set(), Counter(), Counter()
        for arts, _ in read_probe(split):
            n += 1
            for title, _body in arts:
                hit = idx.get(norm(title))
                if hit:
                    matched += 1
                    stories.add(hit[0][0])
                    seen[hit[0][2]] += 1
                    origin[hit[0][3]] += 1
        split_pubs[split] = set(seen)
        print(f"{split:5} examples={n:5} articles joined to metadata={matched}/{3 * n} "
              f"({100 * matched / (3 * n):.1f}%)  stories={len(stories)}  publishers={len(seen)}")
        print(f"      sourced from: { {k.replace('all_crawled_combined_filtered_RoundupLeftCenterRight.jsonl', 'filtered'): v for k, v in origin.items()} }")

    overlap = split_pubs["test"] & split_pubs["train"]
    print(f"\npublishers in test also seen in train: {len(overlap)}/{len(split_pubs['test'])}")
    print("  data_preprocessing.ipynb cell 26 uses train_test_split(random_state=42), i.e. a")
    print("  RANDOM story split. A publisher-leakage probe evaluated on it is scored almost")
    print("  entirely on outlets it trained on -- the easy setting. Baly's media-based split")
    print("  exists precisely to avoid this.")
    print("SUPPLIES ALL THREE: event id, publisher, human neutral summary.")


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #

MATRIX = [
    # corpus, scale, event, publisher, neutral target
    ("NeuS / framing-bias-metric", "3,095 stories x 3 articles",
     "yes, curated story ids", "yes, 250 outlets", "yes, AllSides roundup (~57w)"),
    ("AllSides / Article-Bias-Pred", "37,554 articles",
     "no; ~1.1k mineable clusters", "yes, 491 outlets", "no"),
    ("WCEP-10", "10,200 clusters x ~9 docs",
     "yes, native", "no; ~40% guessable in-text", "25-word blurb, not neutralised"),
    ("WNC", "181k sentence pairs",
     "no", "no", "yes, but sentence-level Wikipedia"),
]


def compare():
    print("\n=== capability matrix ===")
    w = [max(len(str(r[i])) for r in MATRIX + [("corpus", "scale", "event", "publisher", "neutral target")])
         for i in range(5)]
    head = ("corpus", "scale", "event grouping", "publisher label", "neutral target")
    w = [max(w[i], len(head[i])) for i in range(5)]
    line = "  ".join("-" * x for x in w)
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(head)))
    print(line)
    for r in MATRIX:
        print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
    print()
    print("Only NeuS carries all three at once, and it is by far the smallest.")
    print("Everything else trades one axis for scale.")


SECTIONS = {"allsides": allsides, "wcep": wcep, "wnc": wnc, "neus": neus, "compare": compare}

if __name__ == "__main__":
    for name in sys.argv[1:] or SECTIONS:
        SECTIONS[name]()
