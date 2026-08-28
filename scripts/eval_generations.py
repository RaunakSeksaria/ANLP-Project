"""Score each run's generations: ROUGE, plus framing bias raw and normalised.

Reuses the metric from eda/framing_bias.py, and reports the per-VAD-token rate
beside the raw score because the raw score is a length proxy (correlation 0.88
with calibrated token count -- see eda/metric_gold_baseline.md). A run that wins
on raw framing bias while matching on the normalised rate has only learned to
say less.

ROUGE is implemented here rather than pulled in as a dependency: the `rouge`
package NeuS pins is unmaintained and the scores are a dozen lines.

    uv run python scripts/eval_generations.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eda"))

from framing_bias import (  # noqa: E402
    Aggregate, framing_bias, load_aux_lexicons, load_vad, tokenize,
)

RELEASED = ROOT / "data/framing-bias-metric-main/data/naacl2022_lrc_roundup_random_order_probe"
RUNS = ROOT / "runs"


def lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def f1(overlap, n_pred, n_ref):
    if not overlap or not n_pred or not n_ref:
        return 0.0
    p, r = overlap / n_pred, overlap / n_ref
    return 2 * p * r / (p + r)


def rouge(pred, ref):
    p, r = tokenize(pred), tokenize(ref)
    out = {}
    for n in (1, 2):
        pn, rn = ngrams(p, n), ngrams(r, n)
        counts = {}
        for g in rn:
            counts[g] = counts.get(g, 0) + 1
        overlap = 0
        for g in pn:
            if counts.get(g, 0) > 0:
                counts[g] -= 1
                overlap += 1
        out[f"rouge{n}"] = f1(overlap, len(pn), len(rn))
    out["rougeL"] = f1(lcs(p, r), len(p), len(r))
    return out


def strip_prefix(text):
    """Training targets carry NeuS's `TITLE=> . ARTICLE=> ` scaffolding."""
    return text.split("ARTICLE=>", 1)[1].strip() if "ARTICLE=>" in text else text.strip()


def main():
    vad, aux = load_vad(), load_aux_lexicons()
    refs = [strip_prefix(t) for t in (RELEASED / "test.target").read_text().splitlines()]

    results = {}
    run_dirs = sorted(d for d in RUNS.glob("*") if (d / "test_generations.txt").exists())
    if not run_dirs:
        print(f"no runs with generations under {RUNS}")
        return 1

    for run in run_dirs:
        gens = [strip_prefix(g) for g in
                (run / "test_generations.txt").read_text().splitlines()]
        n = min(len(gens), len(refs))
        if len(gens) != len(refs):
            print(f"  warning: {run.name} has {len(gens)} generations for {len(refs)} refs")

        agg = Aggregate()
        totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "words": 0}
        for gen, ref in zip(gens[:n], refs[:n]):
            for k, v in rouge(gen, ref).items():
                totals[k] += v
            totals["words"] += len(tokenize(gen))
            agg.add(framing_bias(gen, ref, vad, aux))

        means = agg.means()
        results[run.name] = {
            "n": n,
            "rouge1": round(totals["rouge1"] / n, 4),
            "rouge2": round(totals["rouge2"] / n, 4),
            "rougeL": round(totals["rougeL"] / n, 4),
            "gen_words": round(totals["words"] / n, 1),
            "framing_p_arousal": round(means["p_arousal"], 3),
            "framing_n_arousal": round(means["n_arousal"], 3),
            "p_arousal_per_vad_token": round(means["p_arousal_per_vad_token"], 4),
            "n_arousal_per_vad_token": round(means["n_arousal_per_vad_token"], 4),
            "calibrated_tokens": round(means["n_tokens"], 1),
        }

    keys = ["rouge1", "rouge2", "rougeL", "gen_words", "framing_p_arousal",
            "p_arousal_per_vad_token", "calibrated_tokens"]
    width = max(len(k) for k in results)
    print(f"\n{'run'.ljust(width)}  " + "  ".join(k[:12].rjust(12) for k in keys))
    print("-" * (width + 14 * len(keys)))
    for name, r in results.items():
        print(f"{name.ljust(width)}  " + "  ".join(f"{r[k]:12.4f}" for k in keys))
    print("\nRead framing_p_arousal against calibrated_tokens: a lower raw score at a")
    print("lower token count is shorter output, not more neutral output.")

    out = RUNS / "generation_metrics.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
