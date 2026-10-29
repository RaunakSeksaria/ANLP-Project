"""Hard subset: the input articles the POLITICS v2 classifier finds EASIEST to assign to their own side.

    uv run python scripts/make_hard_subset.py score            # resumable; GPU strongly advised
    uv run python scripts/make_hard_subset.py select --top-frac 0.2
    -> hard_{left,right}_neutral.jsonl, runs/hard/scores.jsonl

    # one seed, long release only (CPU-feasible: ~5 h); use the same flags for select
    uv run python scripts/make_hard_subset.py score --releases full1024 --seeds seed42
    -> runs/hard/scores_full1024_seed42.jsonl, hard_{left,right}_neutral_full1024_seed42.jsonl

"Hard" for the rewriter: these are the most confidently sided originals, so neutralising them
has the furthest to go. Sources are the two single-article releases, both stances:
  - dedup     ({st}_neutral_dedup.jsonl; inputs < 512 LED tokens)
  - full1024  ({st}_neutral_full1024.jsonl; inputs <= 1,024 LED tokens; build_full_release.py)

Scoring follows compare_variants.py: article body only (the classifier was trained on bodies),
outlet names masked, 512-token windows with 64 overlap, every seed*_best.pt of the v2 folder.
Per article we keep each seed's raw mean-window logits and derive, with the ensemble temperature
(that seed's own temperature when only one seed is used):
  - margin: mean over seeds of (own-side logit - largest other logit) / T  -- the ranking key
  - p_own:  mean over seeds of softmax(logits / T)[own side]  (the calibrated ensemble probability)
An article in both releases is scored once (same body) and kept once in the output, from dedup.
Selection ranks within (stance, release, split), so long articles don't crowd out short ones and
every split keeps its share.

Test set (validation + test articles that are in no train split of either release), with the
classifier's view of each generated target alongside (scores from scripts/kaggle_hard_subset):
    uv run python scripts/make_hard_subset.py select --test-only \
        --target-scores runs/hard/scores_targets.jsonl --out "hard-test/hard_test_{st}_neutral.jsonl"
"""
import argparse, hashlib, importlib.util, json, os
from pathlib import Path

import numpy as np

from compare_variants import OUTLET_RE

ROOT = Path(__file__).resolve().parents[1]
CLF = ROOT / "data/classifiers/politics-ideology-classifier/v2"
OWN = {"left": 0, "right": 2}
RELEASES = {"dedup": "{st}_neutral_dedup.jsonl", "full1024": "{st}_neutral_full1024.jsonl"}
SPLIT = {"val": "validation"}          # dedup says "val", full1024 "validation"


def key(body):
    return hashlib.md5(body.encode()).hexdigest()


def load_rows(st, releases=RELEASES):
    rows = []
    for rel in releases:
        for line in (ROOT / RELEASES[rel].format(st=st)).open(encoding="utf-8"):
            r = json.loads(line)
            r["_release"] = rel
            rows.append(r)
    return rows


def score(args):
    spec = importlib.util.spec_from_file_location("politics_classifier", CLF / "politics_classifier.py")
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)
    weights = [CLF / f"{s}_best.pt" for s in args.seeds]
    models, _ = pc.load(weights)
    tok = pc.tokenizer()
    device = next(models[0].parameters()).device

    out = ROOT / args.scores
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["key"] for l in out.open()} if out.exists() else set()
    todo = {}
    for st in OWN:
        for r in load_rows(st, args.releases):
            body = r[f"{st}_original"]["body"]
            if key(body) not in done:
                todo[key(body)] = body
    todo = sorted(todo.items(), key=lambda kv: len(kv[1]))           # length-sorted, less padding
    print(f"{len(done):,} scored, {len(todo):,} to go on {device}, {len(weights)} seeds", flush=True)

    for s in range(0, len(todo), args.chunk):
        part = todo[s:s + args.chunk]
        wins = pc.to_windows(tok, [OUTLET_RE.sub("<mask>", b) for _, b in part])
        L = [pc.article_logits(m, wins, tok.pad_token_id, device) for m in models]
        with out.open("a", encoding="utf-8") as f:
            for i, (k, _) in enumerate(part):
                f.write(json.dumps({"key": k, "windows": len(wins[i]),
                                    "logits": {w.name.split("_")[0]: L[j][i].round(4).tolist()
                                               for j, w in enumerate(weights)}}) + "\n")
            f.flush(); os.fsync(f.fileno())
        print(f"  {s + len(part):,}/{len(todo):,}", flush=True)


def politics(logits, own, T):
    z = np.array(list(logits.values()), dtype=np.float64) / T                      # seeds x 3
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    others = np.delete(z, own, axis=1).max(1)
    return {"margin": round(float((z[:, own] - others).mean()), 4),
            "p_own": round(float(p[:, own].mean()), 4), "p": p.mean(0).round(4).tolist()}


def load_scores(path, seeds):
    scores = {}
    for line in (ROOT / path).open():
        r = json.loads(line)
        if set(r["logits"]) != set(seeds):
            raise SystemExit(f"{path} holds seeds {sorted(r['logits'])}, not {seeds}")
        scores[r["key"]] = r
    return scores


def select(args):
    temps = json.load(open(CLF / "calibration.json"))["temperature"]
    T = temps["ensemble"] if len(args.seeds) > 1 else temps[args.seeds[0]]
    scores = load_scores(args.scores, args.seeds)
    targets = load_scores(args.target_scores, args.seeds) if args.target_scores else {}
    for st, own in OWN.items():
        rows = load_rows(st, args.releases)
        # --test-only: an article in a train split of ANY loaded release is out, even if another
        # release has it in validation/test (174 left / 147 right such clashes between dedup and full1024)
        train = {key(r[f"{st}_original"]["body"]) for r in rows if r["split"] == "train"} if args.test_only else set()
        seen, groups, missing = set(), {}, 0
        for r in rows:                                            # dedup first: it wins on overlap
            k = key(r[f"{st}_original"]["body"])
            if k in seen or k in train:
                continue
            seen.add(k)
            if k not in scores:
                missing += 1
                continue
            r["split"] = SPLIT.get(r["split"], r["split"])
            r["politics"] = politics(scores[k]["logits"], own, T) | {"windows": scores[k]["windows"]}
            if targets:
                t = targets.get(key(r[f"{st}_neutralised"]))
                if t is None:
                    missing += 1
                    continue
                r["politics_target"] = politics(t["logits"], own, T) | {"windows": t["windows"]}
            groups.setdefault((r.pop("_release"), r["split"]), []).append(r)
        if missing:
            raise SystemExit(f"{st}: {missing:,} articles (or targets) not scored yet; run `score` first")

        kept = []
        print(f"{st}: T={T:.3f}, top {args.top_frac:.0%} by margin within release x split")
        for (rel, sp), rs in sorted(groups.items()):
            rs.sort(key=lambda r: -r["politics"]["margin"])
            n = max(1, round(len(rs) * args.top_frac))
            for i, r in enumerate(rs[:n]):
                r["release"], r["politics"]["rank_in_group"] = rel, i
            kept += rs[:n]
            m = [r["politics"]["margin"] for r in rs]
            print(f"  {rel:9s} {sp:10s} {n:5,}/{len(rs):5,}  margin cut {rs[n - 1]['politics']['margin']:.3f}"
                  f"  (group median {np.median(m):.3f})  p_own kept min {min(r['politics']['p_own'] for r in rs[:n]):.3f}")
        with (ROOT / args.out.format(st=st)).open("w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  -> {args.out.format(st=st)}: {len(kept):,} records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["score", "select"])
    ap.add_argument("--releases", nargs="+", choices=list(RELEASES), default=list(RELEASES))
    ap.add_argument("--seeds", nargs="+", default=sorted(p.name.split("_")[0] for p in CLF.glob("seed*_best.pt")))
    ap.add_argument("--scores", help="default: runs/hard/scores.jsonl, tagged when not all releases and seeds")
    ap.add_argument("--chunk", type=int, default=512, help="articles per checkpoint write")
    ap.add_argument("--top-frac", type=float, default=0.2)
    ap.add_argument("--test-only", action="store_true",
                    help="drop every article that is in a train split of any loaded release")
    ap.add_argument("--target-scores", help="scores of the *_neutralised targets (same format); adds politics_target")
    ap.add_argument("--out", help="default: hard_{st}_neutral.jsonl, tagged like --scores")
    args = ap.parse_args()
    all_seeds = sorted(p.name.split("_")[0] for p in CLF.glob("seed*_best.pt"))
    tag = "".join(f"_{x}" for x in (args.releases if args.releases != list(RELEASES) else [])
                  + (args.seeds if sorted(args.seeds) != all_seeds else []))
    args.scores = args.scores or f"runs/hard/scores{tag}.jsonl"
    args.out = args.out or f"hard_{{st}}_neutral{tag}.jsonl"
    (score if args.step == "score" else select)(args)


if __name__ == "__main__":
    main()
