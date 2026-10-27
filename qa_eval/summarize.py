"""Print the QA content-preservation tables from results/ (stdlib only).

    python summarize.py
"""
import csv, json
from pathlib import Path
from statistics import mean

R = Path(__file__).resolve().parent / "results"
SYSTEMS = ["nemotron", "lambda0", "lambda035"]

for part, title in (("own_test", "QuestEval, side_neutral_full test split (1,775 articles)"),
                    ("long1k_qe300", "QuestEval, long1k sample (300 articles, 1,024-token LED input)")):
    print(f"\n{title}\n| System | QuestEval | Precision | Recall |\n|---|---|---|---|")
    for s in SYSTEMS:
        rows = [json.loads(l) for l in open(R / part / f"questeval_{s}.jsonl")]
        print(f"| {s} | {mean(r['questeval'] for r in rows):.3f} | {mean(r['precision'] for r in rows):.3f} | "
              f"{mean(r['recall'] for r in rows):.3f} |")

rows = list(csv.DictReader(open(R / "diversesumm" / "diversesumm_items.csv")))
cov = lambda r, s: float(r[f"p_{s}"]) > 0.5           # answer entailed by some chunk of the text
orig = [r for r in rows if cov(r, "original")]
print(f"\nDiverseSumm NLI coverage ({len(rows)} answers, {len({r['eid'] for r in rows})} stories)\n"
      "| Text | Answers covered | Kept, of those the original covers |\n|---|---|---|")
for s in ["original", "lambda0", "lambda035"]:
    kept = "—" if s == "original" else f"{100 * mean(cov(r, s) for r in orig):.1f}%"
    print(f"| {s} | {100 * mean(cov(r, s) for r in rows):.1f}% | {kept} |")
