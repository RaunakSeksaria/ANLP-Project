"""The NeuS framing-bias metric, reimplemented, plus a gold-data baseline.

The metric shipped with framing-bias-metric-main (`metric_code.ipynb`) does not
run: `other_bias_lexicon_analysis`, `pred_path` and `test_srcs` are never
defined, `blue_score` is printed but its assignment is commented out, `re` and
`wordnet_lemmatizer` are never imported, and four of the ten cells are empty.
This module reimplements the definition the notebook sketches.

Definition (Lee et al. 2022, NAACL):

    calibrated = set(tokens(generation)) - set(tokens(reference))
    for each calibrated token found in NRC-VAD:
        valence      += v
        v_positive   += |v - 0.5|   if v >= 0.5
        v_negative   += |v - 0.5|   if v <  0.5
        p_arousal    += a           if v >  0.65
        n_arousal    += a           if v <  0.35
    report the per-instance mean of each accumulator

Subtracting the reference's tokens is the "calibration": it charges a generation
only for affect-laden words the neutral reference did not itself use.

Two deviations from the notebook, both deliberate and both documented in
`eda/metric_gold_baseline.md`:

  * Tokenisation is stdlib. The notebook strips punctuation *before* calling
    `nltk.word_tokenize`, so on its own input `word_tokenize` degenerates to a
    whitespace split -- `re.findall(r"\\w+")` is equivalent and drops the
    dependency.
  * Lemmatisation resolves against the lexicon being queried rather than
    WordNet. WordNet's noun lemmatiser exists here only to make lexicon lookups
    hit; applying its `_morphy` noun rules and keeping a stripped form only when
    that form is in the lexicon does the same job deterministically, with no
    corpus download.

    uv run python eda/framing_bias.py --self-test
    uv run python eda/framing_bias.py                  # write the gold baseline
"""

import argparse
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEX = ROOT / "data/framing-bias-metric-main/data/lexicons"
NEUS = ROOT / "data/framing-bias-metric-main/data"
WCEP = ROOT / "data/WCEP-10"

POSITIVE_V = 0.65
NEGATIVE_V = 0.35
NEUTRAL_V = 0.5

# WordNet's noun _morphy rules, in the order it applies them.
MORPHY_NOUN = [("ses", "s"), ("xes", "x"), ("zes", "z"), ("ches", "ch"),
               ("shes", "sh"), ("men", "man"), ("ies", "y"), ("s", "")]

ACCUMULATORS = ("valence", "v_positive", "v_negative", "p_arousal", "n_arousal",
                "n_vad_tokens", "n_tokens")


# --------------------------------------------------------------------------- #
# lexicons
# --------------------------------------------------------------------------- #

def load_vad():
    """word -> (valence, arousal, dominance). 19,973 entries, tab-separated,
    no header, and no trailing newline on the last line."""
    vad = {}
    path = LEX / "NRC-VAD-Lexicon-Aug2018Release/NRC-VAD-Lexicon.txt"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        word, v, a, d = line.split("\t")
        vad[word] = (float(v), float(a), float(d))
    return vad


def _read_wordlist(path):
    """One word per line, '#' comment banners, blank lines."""
    return {
        w.strip().lower()
        for w in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if w.strip() and not w.startswith("#")
    }


def load_aux_lexicons():
    """The bias lexicons the notebook references but never implements."""
    rel = LEX / "bias_related_lexicons"
    lex = {
        "bias_lexicon": _read_wordlist(LEX / "bias-lexicon/bias-lexicon.txt"),
        "assertives": _read_wordlist(rel / "assertives_hooper1975.txt"),
        "factives": _read_wordlist(rel / "factives_hooper1975.txt"),
        "hedges": _read_wordlist(rel / "hedges_hyland2005.txt"),
        "implicatives": _read_wordlist(rel / "implicatives_karttunen1971.txt"),
        "report_verbs": _read_wordlist(rel / "report_verbs.txt"),
    }
    # MPQA subjectivity clues: "type=strongsubj ... word1=abandon ... priorpolarity=negative"
    tff = rel / "subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff"
    strong, weak = set(), set()
    for line in tff.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = dict(f.split("=", 1) for f in line.split() if "=" in f)
        word = fields.get("word1")
        if not word:
            continue
        (strong if fields.get("type") == "strongsubj" else weak).add(word.lower())
    lex["subj_strong"] = strong
    lex["subj_weak"] = weak
    return lex


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #

def tokenize(text):
    """Faithful to the notebook's preprocess_text, minus the nltk round-trip."""
    text = text.replace("U.S.", "USA").lower()
    return re.findall(r"\w+", text)


def lemma_for(token, vocab):
    """Strip a noun plural only when doing so lands inside `vocab`."""
    if token in vocab:
        return token
    for suffix, replacement in MORPHY_NOUN:
        if token.endswith(suffix):
            candidate = token[: -len(suffix)] + replacement
            if candidate in vocab:
                return candidate
    return token


def token_set(text, vocab):
    return {lemma_for(t, vocab) for t in tokenize(text)}


# --------------------------------------------------------------------------- #
# the metric
# --------------------------------------------------------------------------- #

def score_tokens(tokens, vad):
    """Accumulators for one calibrated token set."""
    out = dict.fromkeys(ACCUMULATORS, 0.0)
    out["n_tokens"] = len(tokens)
    for t in tokens:
        hit = vad.get(t)
        if hit is None:
            continue
        v, a, _ = hit
        out["n_vad_tokens"] += 1
        out["valence"] += v
        if v < NEUTRAL_V:
            out["v_negative"] += abs(v - NEUTRAL_V)
        else:
            out["v_positive"] += abs(v - NEUTRAL_V)
        if v > POSITIVE_V:
            out["p_arousal"] += a
        elif v < NEGATIVE_V:
            out["n_arousal"] += a
    return out


def framing_bias(generation, reference, vad, aux=None):
    """The calibrated metric for one (generation, reference) pair."""
    gen = token_set(generation, vad)
    ref = token_set(reference, vad)
    calibrated = gen - ref
    out = score_tokens(calibrated, vad)
    if aux:
        for name, words in aux.items():
            out[name] = sum(1 for t in calibrated if lemma_for(t, words) in words)
    return out


class Aggregate:
    """Sums accumulators over instances and reports per-instance means."""

    def __init__(self):
        self.total = defaultdict(float)
        self.n = 0

    def add(self, scores):
        for k, v in scores.items():
            self.total[k] += v
        self.n += 1

    def means(self):
        if not self.n:
            return {}
        m = {k: v / self.n for k, v in self.total.items()}
        # arousal per calibrated token, to separate "more biased" from "longer"
        vad_tokens = self.total["n_vad_tokens"] or 1
        m["p_arousal_per_vad_token"] = self.total["p_arousal"] / vad_tokens
        m["n_arousal_per_vad_token"] = self.total["n_arousal"] / vad_tokens
        m["vad_coverage"] = self.total["n_vad_tokens"] / (self.total["n_tokens"] or 1)
        return m


HEADLINE = ("p_arousal", "n_arousal", "v_positive", "v_negative",
            "p_arousal_per_vad_token", "n_arousal_per_vad_token",
            "n_tokens", "n_vad_tokens", "vad_coverage")


def table(rows, keys=HEADLINE):
    """rows: list of (label, means dict)."""
    label_w = max(len(r[0]) for r in rows)
    head = "  ".join(["".ljust(label_w)] + [k[:11].rjust(11) for k in keys])
    out = [head, "-" * len(head)]
    for label, m in rows:
        out.append("  ".join([label.ljust(label_w)]
                             + [f"{m.get(k, 0):11.3f}" for k in keys]))
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# gold baselines
# --------------------------------------------------------------------------- #

def sentences(seq):
    return " ".join(s for _, s in seq)


def neus_gold(vad, aux):
    """Score every gold L/C/R article as if it were a generation, against the
    human roundup. Establishes the metric's range on real framing before any
    model optimises against it."""
    rows = [json.loads(l) for l in
            (NEUS / "raw_crawled/all_crawled_combined_filtered_RoundupLeftCenterRight.jsonl").open()]

    by_stance = defaultdict(Aggregate)
    concat, identity, center_only = Aggregate(), Aggregate(), Aggregate()

    for r in rows:
        roundup = sentences(r["roundup"])
        if not roundup.strip():
            continue
        bodies = {}
        for stance, v in r["news"].items():
            body = sentences(v["newBody"])
            bodies[stance] = body
            by_stance[stance].add(framing_bias(body, roundup, vad, aux))
        concat.add(framing_bias(" ".join(bodies.values()), roundup, vad, aux))
        identity.add(framing_bias(roundup, roundup, vad, aux))
        if "center" in bodies:
            center_only.add(framing_bias(bodies["center"], roundup, vad, aux))

    return {
        "n_stories": len(rows),
        "rows": [(f"gold {s} article", by_stance[s].means()) for s in ("left", "center", "right")]
                + [("concat L+C+R", concat.means()),
                   ("copy-centre baseline", center_only.means()),
                   ("identity (roundup vs itself)", identity.means())],
    }


def length_confound(vad):
    """Is the metric measuring affect, or just how many novel tokens survived
    calibration? Correlate the raw accumulators against the calibrated token
    count, and compare per-token affect rates across stances."""
    rows = [json.loads(l) for l in
            (NEUS / "raw_crawled/all_crawled_combined_filtered_RoundupLeftCenterRight.jsonl").open()]

    per_stance = defaultdict(list)
    scores = []
    for r in rows:
        roundup = sentences(r["roundup"])
        if not roundup.strip():
            continue
        for stance, v in r["news"].items():
            s = framing_bias(sentences(v["newBody"]), roundup, vad)
            scores.append(s)
            per_stance[stance].append(s)

    def pearson(xs, ys):
        mx, my = st.mean(xs), st.mean(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
        return num / den if den else float("nan")

    vad_tokens = [s["n_vad_tokens"] for s in scores]
    correlations = {
        "p_arousal vs n_tokens": pearson([s["n_tokens"] for s in scores],
                                         [s["p_arousal"] for s in scores]),
        "p_arousal vs n_vad_tokens": pearson(vad_tokens, [s["p_arousal"] for s in scores]),
        "n_arousal vs n_vad_tokens": pearson(vad_tokens, [s["n_arousal"] for s in scores]),
        "p+n arousal vs n_vad_tokens": pearson(vad_tokens,
                                               [s["p_arousal"] + s["n_arousal"] for s in scores]),
    }

    rates = {}
    for stance, ss in per_stance.items():
        per_token = [s["p_arousal"] / s["n_vad_tokens"] for s in ss if s["n_vad_tokens"]]
        rates[stance] = {
            "n": len(ss),
            "p_arousal_raw": st.mean(s["p_arousal"] for s in ss),
            "p_arousal_per_vad_token": st.mean(per_token),
            "sd_per_vad_token": st.pstdev(per_token),
        }
    return {"n_instances": len(scores), "correlations": correlations, "per_stance": rates}


def wcep_gold(vad, aux, split="test", limit=None):
    """Same metric on WCEP-10, whose 25-word target was never written to be
    neutral. Checks whether the metric means anything there."""
    agg, identity = Aggregate(), Aggregate()
    n = 0
    with (WCEP / f"{split}.txt").open() as f:
        for line in f:
            r = json.loads(line)
            summary = r["summary"]
            for d in r["document"]:
                agg.add(framing_bias(d, summary, vad, aux))
            identity.add(framing_bias(summary, summary, vad, aux))
            n += 1
            if limit and n >= limit:
                break
    return {"n_clusters": n,
            "rows": [(f"WCEP-10 {split} document", agg.means()),
                     ("identity (summary vs itself)", identity.means())]}


# --------------------------------------------------------------------------- #

def self_test(vad, aux):
    checks = []

    known = {"murder": "low valence", "joy": "high valence", "table": "mid valence"}
    for w in known:
        assert w in vad, f"{w} missing from NRC-VAD"
    assert vad["joy"][0] > POSITIVE_V, vad["joy"]
    assert vad["murder"][0] < NEGATIVE_V, vad["murder"]
    checks.append(f"NRC-VAD loaded: {len(vad)} entries; joy={vad['joy'][0]:.3f} murder={vad['murder'][0]:.3f}")

    text = "The brutal massacre triggered outrage and grief across the nation."
    zero = framing_bias(text, text, vad, aux)
    assert zero["p_arousal"] == 0 and zero["n_arousal"] == 0 and zero["n_tokens"] == 0, zero
    checks.append("calibration: identity pair scores exactly zero")

    biased = framing_bias("A brutal massacre and a horrific atrocity.", "An incident occurred.", vad, aux)
    neutral = framing_bias("An event took place on Tuesday.", "An incident occurred.", vad, aux)
    assert biased["n_arousal"] > neutral["n_arousal"], (biased, neutral)
    checks.append(f"ordering: loaded text n_arousal={biased['n_arousal']:.3f} > "
                  f"flat text {neutral['n_arousal']:.3f}")

    assert lemma_for("massacres", vad) == "massacre"
    assert lemma_for("atrocities", vad) == "atrocity"
    assert lemma_for("gas", vad) == "gas", "must not strip a real -s word present in the lexicon"
    checks.append("lemmatiser: plurals resolve, lexicon-present -s words survive")

    sizes = {k: len(v) for k, v in aux.items()}
    assert all(sizes.values()), sizes
    checks.append(f"aux lexicons: {sizes}")

    for c in checks:
        print("  ok  " + c)
    print("\nself-test passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--wcep-limit", type=int, default=None,
                    help="cap WCEP clusters (default: all of the test split)")
    ap.add_argument("--out", default=str(ROOT / "eda"))
    args = ap.parse_args()

    vad = load_vad()
    aux = load_aux_lexicons()

    if args.self_test:
        self_test(vad, aux)
        return

    neus_res = neus_gold(vad, aux)
    wcep_res = wcep_gold(vad, aux, limit=args.wcep_limit)

    print(f"\n=== NeuS gold articles vs human roundup  (n={neus_res['n_stories']} stories) ===")
    print(table(neus_res["rows"]))
    print(f"\n=== WCEP-10 documents vs 25-word summary  (n={wcep_res['n_clusters']} clusters) ===")
    print(table(wcep_res["rows"]))

    aux_keys = sorted(aux)
    print("\n=== auxiliary lexicon hits per instance ===")
    print(table(neus_res["rows"] + wcep_res["rows"], keys=aux_keys))

    conf = length_confound(vad)
    print(f"\n=== is the metric measuring affect or length?  (n={conf['n_instances']} gold articles) ===")
    for k, v in conf["correlations"].items():
        print(f"  pearson  {k:32} {v: .4f}")
    print()
    for stance in ("left", "center", "right"):
        s = conf["per_stance"][stance]
        print(f"  {stance:7} n={s['n']:5}  raw p_arousal={s['p_arousal_raw']:.3f}"
              f"   per VAD token={s['p_arousal_per_vad_token']:.4f} (sd {s['sd_per_vad_token']:.4f})")
    print("  -> raw scores separate the stances; per-token affect rates do not.")

    out = Path(args.out) / "metric_gold_baseline.json"
    out.write_text(json.dumps(
        {"neus": {"n_stories": neus_res["n_stories"],
                  "rows": {k: v for k, v in neus_res["rows"]}},
         "wcep": {"n_clusters": wcep_res["n_clusters"],
                  "rows": {k: v for k, v in wcep_res["rows"]}},
         "length_confound": conf},
        indent=2, sort_keys=True))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
