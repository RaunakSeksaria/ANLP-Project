"""Aggregate POLITICS v2 seeds: per-seed and mean +- std metrics, per-epoch curves, per-class results,
temperature scaling (per seed and for the seed ensemble), reliability plots, and calibration.json.

Ensemble = mean over seeds of softmax(logits / T), the same rule `politics_classifier.predict` applies, with
one temperature fitted for it on the validation set.

    python aggregate_v2.py --runs runs/ --out v2_results/
"""
import argparse, json
from pathlib import Path

import numpy as np

from politics_classifier import LABELS, softmax, metrics, fit_temperature, reliability


def fit_mixture_temperature(logit_list, y):
    y = np.asarray(y)
    nll = lambda t: -np.log(np.clip(np.mean([softmax(l, t) for l in logit_list], 0)[np.arange(len(y)), y], 1e-12, 1)).mean()
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 400))
    t0 = grid[np.argmin([nll(t) for t in grid])]
    fine = np.linspace(t0 * 0.9, t0 * 1.1, 201)
    return float(fine[np.argmin([nll(t) for t in fine])])


def ms(xs):
    xs = np.asarray(xs, dtype=float)
    return f"{xs.mean():.2f} ± {xs.std(ddof=1) if len(xs) > 1 else 0:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    seeds = sorted(int(p.name[4:]) for p in runs.glob("seed*") if (p / f"{p.name}_best_logits.npz").exists())
    assert seeds, f"no finished seeds under {runs}"

    S, L = {}, {}
    for s in seeds:
        d = runs / f"seed{s}"
        S[s] = {"summary": json.load(open(d / f"seed{s}_summary.json")),
                "epochs": [json.loads(l) for l in open(d / f"seed{s}_epochs.jsonl")]}
        z = np.load(d / f"seed{s}_best_logits.npz")
        L[s] = {k: z[k] for k in z.files}
    yv, yt = L[seeds[0]]["valid_y"], L[seeds[0]]["test_y"]
    for s in seeds:  # every seed must be scored on the same articles in the same order
        assert (L[s]["valid_id"] == L[seeds[0]]["valid_id"]).all() and (L[s]["test_id"] == L[seeds[0]]["test_id"]).all()

    # ---- per seed, raw and temperature-scaled
    rep = {"seeds": seeds, "per_seed": {}, "temperature": {}}
    for s in seeds:
        t = fit_temperature(L[s]["valid"], yv)
        rep["temperature"][f"seed{s}"] = t
        rep["per_seed"][s] = {
            "best_epoch": S[s]["summary"]["best_epoch"], "epochs_run": S[s]["summary"]["epochs_run"],
            "valid": metrics(yv, softmax(L[s]["valid"])), "test": metrics(yt, softmax(L[s]["test"])),
            "valid_calibrated": metrics(yv, softmax(L[s]["valid"], t)), "test_calibrated": metrics(yt, softmax(L[s]["test"], t))}

    # ---- ensemble
    te = fit_mixture_temperature([L[s]["valid"] for s in seeds], yv)
    rep["temperature"]["ensemble"] = te
    ens = lambda part, t: np.mean([softmax(L[s][part], t) for s in seeds], 0)
    rep["ensemble"] = {"valid": metrics(yv, ens("valid", 1.0)), "test": metrics(yt, ens("test", 1.0)),
                       "valid_calibrated": metrics(yv, ens("valid", te)), "test_calibrated": metrics(yt, ens("test", te))}
    rep["reliability"] = {"test_raw": reliability(yt, ens("test", 1.0)), "test_calibrated": reliability(yt, ens("test", te))}
    json.dump(rep, open(out / "report.json", "w"), indent=2, default=str)
    json.dump({"temperature": rep["temperature"], "ensemble_rule": "mean over seeds of softmax(logits / T)",
               "fit_on": "Article-Bias-Prediction random-split validation set (6,996 articles), NLL"},
              open(out / "calibration.json", "w"), indent=2)

    # ---- per-epoch curves
    curves = []
    for s in seeds:
        for e in S[s]["epochs"]:
            curves.append({"seed": s, "epoch": e["epoch"], "train_loss": e["train_loss"],
                           "val_f1": e["valid"]["macro_f1"], "val_acc": e["valid"]["acc"],
                           "test_f1": e["test"]["macro_f1"], "test_acc": e["test"]["acc"],
                           "test_centre_recall": e["test"]["per_class"]["Centre"]["recall"]})
    json.dump(curves, open(out / "epoch_curves.json", "w"), indent=2)

    # ---- plots
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for s in seeds:
            c = [r for r in curves if r["seed"] == s]
            ax[0].plot([r["epoch"] for r in c], [r["val_f1"] for r in c], marker="o", label=f"seed {s}")
            ax[1].plot([r["epoch"] for r in c], [r["test_f1"] for r in c], marker="o", label=f"seed {s}")
        for x, t in zip(ax, ("Validation macro-F1", "Test macro-F1")):
            x.set_title(t); x.set_xlabel("epoch"); x.grid(alpha=.3); x.legend()
        fig.tight_layout(); fig.savefig(out / "epoch_curves.png", dpi=130); plt.close(fig)

        fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
        for x, key, title in ((ax[0], "test_raw", "Before (T = 1)"), (ax[1], "test_calibrated", f"After (T = {te:.2f})")):
            rows = [r for r in rep["reliability"][key] if r["n"]]
            x.plot([0, 1], [0, 1], "--", color="grey", lw=1)
            x.bar([(r["lo"] + r["hi"]) / 2 for r in rows], [r["accuracy"] for r in rows], width=1 / 15, alpha=.6,
                  edgecolor="black", label="accuracy")
            x.plot([r["confidence"] for r in rows], [r["accuracy"] for r in rows], "o", color="C3", ms=4)
            m = rep["ensemble"]["test_calibrated" if key == "test_calibrated" else "test"]
            x.set_title(f"{title}: ECE {m['ece']:.3f}"); x.set_xlabel("confidence"); x.set_ylabel("accuracy")
            x.set_xlim(0, 1); x.set_ylim(0, 1)
        fig.suptitle("Seed ensemble, Article-Bias-Prediction test set"); fig.tight_layout()
        fig.savefig(out / "reliability.png", dpi=130); plt.close(fig)
    except ImportError:
        print("matplotlib missing: skipped plots")

    # ---- markdown summary
    f = lambda m: f"{m['macro_f1']:.2f} | {m['acc']:.2f} | {m['mae']:.3f}"
    lines = ["| Model | Best epoch / run | Val F1 | Test F1 | Test acc | Test MAE | Test recall L / C / R | Test ECE raw → calibrated (T) |",
             "|---|---|---|---|---|---|---|---|"]
    for s in seeds:
        r = rep["per_seed"][s]; t = r["test"]; pc = t["per_class"]
        lines.append(f"| seed {s} | {r['best_epoch']} / {r['epochs_run']} | {r['valid']['macro_f1']:.2f} | {t['macro_f1']:.2f} | "
                     f"{t['acc']:.2f} | {t['mae']:.3f} | {pc['Left']['recall']:.1f} / {pc['Centre']['recall']:.1f} / {pc['Right']['recall']:.1f} | "
                     f"{t['ece']:.3f} → {r['test_calibrated']['ece']:.3f} ({rep['temperature'][f'seed{s}']:.2f}) |")
    tf = [rep["per_seed"][s]["test"] for s in seeds]
    lines.append(f"| **mean ± std** | | {ms([rep['per_seed'][s]['valid']['macro_f1'] for s in seeds])} | {ms([t['macro_f1'] for t in tf])} | "
                 f"{ms([t['acc'] for t in tf])} | {ms([t['mae'] for t in tf])} | "
                 f"{ms([t['per_class']['Left']['recall'] for t in tf])} / {ms([t['per_class']['Centre']['recall'] for t in tf])} / "
                 f"{ms([t['per_class']['Right']['recall'] for t in tf])} | {ms([t['ece'] for t in tf])} |")
    e = rep["ensemble"]; pc = e["test"]["per_class"]
    lines.append(f"| **ensemble of {len(seeds)}** | | {e['valid']['macro_f1']:.2f} | {e['test']['macro_f1']:.2f} | {e['test']['acc']:.2f} | "
                 f"{e['test']['mae']:.3f} | {pc['Left']['recall']:.1f} / {pc['Centre']['recall']:.1f} / {pc['Right']['recall']:.1f} | "
                 f"{e['test']['ece']:.3f} → {e['test_calibrated']['ece']:.3f} ({te:.2f}) |")
    cm = e["test"]["confusion"]
    lines += ["", "Ensemble confusion matrix, test (rows true, columns predicted):", "",
              "| | Left | Centre | Right |", "|---|---|---|---|"] + \
             [f"| **{LABELS[i]}** | {cm[i][0]} | {cm[i][1]} | {cm[i][2]} |" for i in range(3)]
    lines += ["", "Per-epoch macro-F1 (val / test):", "", "| Epoch | " + " | ".join(f"seed {s}" for s in seeds) + " |",
              "|---|" + "---|" * len(seeds)]
    for ep in range(1, max(r["epoch"] for r in curves) + 1):
        cells = []
        for s in seeds:
            c = [r for r in curves if r["seed"] == s and r["epoch"] == ep]
            cells.append(f"{c[0]['val_f1']:.2f} / {c[0]['test_f1']:.2f}" + (" ★" if ep == rep["per_seed"][s]["best_epoch"] else "") if c else "—")
        lines.append(f"| {ep} | " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
