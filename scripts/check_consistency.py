"""Compare two runs of the generator, or re-run a random sample and compare.

    # two existing runs
    uv run python scripts/check_consistency.py --run A.jsonl --run-b B.jsonl

    # regenerate a random 30 of run A under its own recorded config
    uv run python scripts/check_consistency.py --run A.jsonl --in clusters.jsonl \
        --regenerate --sample 30 --seed 7 --similarity mymodule:score

Temperature is pinned and recorded, but Gemini is not bit-deterministic, so two
runs of the same config will differ in wording. This measures by how much, and
-- more usefully -- whether the metrics move when the wording does.

Similarity is supplied, not chosen here. --similarity takes any
`module:function` or `path/to/file.py:function` of the shape

    score(a: str, b: str) -> float          # per pair
    score(pairs: list[tuple[str, str]]) -> list[float]   # batched

The batched form is detected from the signature and preferred when present. The
built-in default is token Jaccard: crude, there only so this runs before your
scorer exists, and labelled as the default wherever it is reported.

Neither input run file is modified.
"""

import argparse
import inspect
import json
import random
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from neutralize import evaluate as ev                                  # noqa: E402
from neutralize.clusters import read_clusters, read_records            # noqa: E402
from neutralize.config import GenConfig                                # noqa: E402
from neutralize.plugins import load_callable                           # noqa: E402

_WORD = re.compile(r"[a-z0-9']+")


def jaccard(a, b):
    """The default. Deliberately crude -- swap it out with --similarity."""
    sa, sb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 1.0


def resolve_similarity(spec):
    """Returns (scorer(pairs) -> [float], label, is_default)."""
    if not spec:
        return (lambda pairs: [jaccard(a, b) for a, b in pairs],
                "token Jaccard", True)
    fn = load_callable(spec)
    required = [p for p in inspect.signature(fn).parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if len(required) == 1:
        return (lambda pairs: [float(x) for x in fn(list(pairs))], spec, False)
    return (lambda pairs: [float(fn(a, b)) for a, b in pairs], spec, False)


def config_from_record(rec, overrides=None):
    """Rebuild the generating config from a record, so a re-run matches exactly."""
    cfg = GenConfig(model=rec.get("model") or GenConfig().model)
    for k, v in (rec.get("params") or {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    for k, v in (overrides or {}).items():
        setattr(cfg, k, v)
    return cfg


def regenerate(run_recs, args):
    """Re-run a random sample under the config the original run recorded."""
    import generate_neutral as gn
    from neutralize.client import GeminiClient, RateGate
    from neutralize.clusters import Appender

    have = [r for r in run_recs if r.get("summary")]
    if not have:
        sys.exit("run A has no successful records to resample")
    ids = {r["story_id"] for r in have}
    sample = random.Random(args.seed).sample(sorted(ids), min(args.sample, len(ids)))

    cfg = config_from_record(have[0], {"workers": args.workers, "rpm": args.rpm})
    if args.model:
        cfg.model = args.model
    clusters = [c for c in read_clusters(args.inp, cfg) if c["id"] in set(sample)]
    missing = set(sample) - {c["id"] for c in clusters}
    if missing:
        print(f"  {len(missing)} sampled id(s) not found in {args.inp}", file=sys.stderr)
    if not clusters:
        sys.exit("none of the sampled ids are in the clusters file")

    out_b = args.run_b or str(Path(args.run).with_suffix(
        f".rerun-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"))
    print(f"regenerating {len(clusters)} cluster(s) -> {out_b}\n"
          f"  model={cfg.model} temperature={cfg.temperature} seed={cfg.seed} "
          f"(from the recorded config of {args.run})", file=sys.stderr)

    client = GeminiClient(cfg, gn.api_key(), RateGate(cfg.rpm))
    runner = gn.Runner(cfg, client, lambda s, c, cf: ev.evaluate(s, c, cf), out_b)
    from concurrent.futures import ThreadPoolExecutor
    recs = []
    with Appender(out_b) as out, ThreadPoolExecutor(cfg.workers) as pool:
        for rec in pool.map(runner.process, clusters):
            out.write(rec)
            recs.append(rec)
            print(f"  {rec['story_id']}: "
                  + (rec.get("error") or f"{rec['actual_words']}w"), file=sys.stderr)
    return recs, out_b


def mean(xs):
    return statistics.fmean(xs) if xs else None


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="first run's output JSONL")
    p.add_argument("--run-b", default=None,
                   help="second run's JSONL; with --regenerate, where to write it")
    p.add_argument("--regenerate", action="store_true",
                   help="re-generate a random sample of --run instead of reading --run-b")
    p.add_argument("--in", dest="inp", default=None,
                   help="clusters JSONL; required by --regenerate")
    p.add_argument("--sample", type=int, default=30)
    p.add_argument("--seed", type=int, default=7, help="which clusters to resample")
    p.add_argument("--model", default=None, help="override the recorded model id")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--rpm", type=int, default=60)
    p.add_argument("--similarity", default=None, metavar="MODULE:FUNC",
                   help="your similarity function; default is token Jaccard")
    p.add_argument("--out", default=None, help="where to write the JSON report")
    args = p.parse_args(argv)

    a_recs = read_records(args.run)
    if not a_recs:
        sys.exit(f"no records in {args.run}")

    if args.regenerate:
        if not args.inp:
            sys.exit("--regenerate needs --in <clusters.jsonl>")
        b_recs, b_path = regenerate(a_recs, args)
    else:
        if not args.run_b:
            sys.exit("give --run-b, or --regenerate with --in")
        b_recs, b_path = read_records(args.run_b), args.run_b
    if not b_recs:
        sys.exit(f"no records in {b_path}")

    a = {r["story_id"]: r for r in a_recs if r.get("summary")}
    b = {r["story_id"]: r for r in b_recs if r.get("summary")}
    shared = sorted(set(a) & set(b))
    if not shared:
        sys.exit("the two runs share no story with a summary in both")

    score, label, is_default = resolve_similarity(args.similarity)

    # Same story, two runs.
    sims = score([(a[i]["summary"], b[i]["summary"]) for i in shared])

    # Different stories, one run. Without this the number above is unreadable:
    # two summaries of one event share its proper nouns and figures whatever the
    # model does, so a high score may mean nothing at all.
    rng = random.Random(args.seed)
    control_pairs = []
    if len(shared) > 1:
        for i in shared:
            j = rng.choice([x for x in shared if x != i])
            control_pairs.append((a[i]["summary"], a[j]["summary"]))
    control = score(control_pairs) if control_pairs else []

    rows, metric_deltas = [], {}
    for i, s in zip(shared, sims):
        ma, mb = a[i].get("metrics") or {}, b[i].get("metrics") or {}
        d = {}
        for k in set(ma) & set(mb):
            if isinstance(ma[k], (int, float)) and isinstance(mb[k], (int, float)):
                d[k] = abs(ma[k] - mb[k])
                metric_deltas.setdefault(k, []).append(d[k])
        rows.append({
            "story_id": i, "similarity": round(s, 4),
            "words_a": a[i]["actual_words"], "words_b": b[i]["actual_words"],
            "word_delta": abs(a[i]["actual_words"] - b[i]["actual_words"]),
            "metric_delta": {k: round(v, 4) for k, v in sorted(d.items())},
        })

    report = {
        "run_a": args.run, "run_b": b_path, "n_pairs": len(shared),
        "similarity_fn": label + (" (default)" if is_default else ""),
        "model": a[shared[0]].get("model"),
        "params": a[shared[0]].get("params"),
        "same_story_similarity": {
            "mean": round(mean(sims), 4), "min": round(min(sims), 4),
            "max": round(max(sims), 4),
        },
        "different_story_similarity": (
            {"mean": round(mean(control), 4)} if control else None),
        "word_delta_mean": round(mean([r["word_delta"] for r in rows]), 2),
        "metric_delta_mean": {k: round(mean(v), 5)
                              for k, v in sorted(metric_deltas.items())},
        "pairs": rows,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    w = max(len("story"), max(len(r["story_id"]) for r in rows))
    print(f"\n{'story'.ljust(w)}  {'sim':>7}  {'words a/b':>11}  {'Δwords':>7}")
    print("-" * (w + 32))
    for r in rows:
        print(f"{r['story_id'].ljust(w)}  {r['similarity']:7.4f}  "
              f"{str(r['words_a']) + '/' + str(r['words_b']):>11}  {r['word_delta']:7d}")

    ss, ds = report["same_story_similarity"], report["different_story_similarity"]
    print(f"\nsimilarity fn      {report['similarity_fn']}")
    print(f"same story         {ss['mean']:.4f}  (min {ss['min']:.4f}, max {ss['max']:.4f})")
    if ds:
        print(f"different story    {ds['mean']:.4f}  <- the control")
        gap = ss["mean"] - ds["mean"]
        print(f"gap                {gap:+.4f}")
        if gap < 0.15:
            print("\n  The control is close to the same-story score, so this function is not")
            print("  separating 'same event' from 'same wording'. Read the metric deltas")
            print("  below instead, and consider a different --similarity.")
    if is_default:
        print("\n  Token Jaccard is the placeholder default. Pass --similarity "
              "module:func\n  to use your own scorer.")

    if report["metric_delta_mean"]:
        print(f"\nmean |Δ| per metric across {len(rows)} pairs "
              f"(small = the metrics are stable under rewording):")
        for k, v in report["metric_delta_mean"].items():
            print(f"  {k:<26} {v:.5f}")

    out = Path(args.out) if args.out else (
        ROOT / "runs" / f"consistency_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
