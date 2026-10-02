"""Score one summary against its source cluster.

Two plain functions, no class hierarchy:

    evaluate(summary, cluster) -> {metric_name: number}
    failures(metrics, cfg)     -> ["human-readable problem", ...]   empty == passed

The driver calls the first, then the second, and if the second returns anything
it asks Gemini to rewrite with those strings in the prompt. Swap the whole thing
out with `--evaluator mymodule:my_evaluate`; it only has to return a dict of
numbers whose keys line up with the thresholds in GenConfig.

The framing-bias metric is eda/framing_bias.py, reused rather than
reimplemented. Coverage uses a local content-word recall instead of ROUGE F1:
the question is whether each source's contribution survived, and a summary is
expected to be far shorter than its sources combined.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eda", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Optional: data/ is gitignored, so the lexicons may simply not be on this
# machine. That degrades the metric set rather than killing the run.
try:
    from framing_bias import (  # noqa: E402
        framing_bias, load_aux_lexicons, load_vad, token_set, tokenize,
    )
    _HAVE_FRAMING = True
except Exception:                                        # pragma: no cover
    _HAVE_FRAMING = False

    def tokenize(text):
        return re.findall(r"[a-z0-9']+", text.lower())

POSITIVE_V, NEGATIVE_V = 0.65, 0.35

STOP = set("""a an the and or but if while of to in on at by for with from as is are was were be
been being it its this that these those he she they we you i his her their our your them him us
me my not no nor so than then there here which who whom whose what when where why how all any
both each few more most other some such only own same too very can will just should now said
say says over under after before during about into out up down off again further once""".split())

_NUM = re.compile(r"\b\d[\d,]*(?:\.\d+)?%?\b")

_resources = None


def load_resources():
    """Lexicons are loaded once per process; they are a few MB of dict."""
    global _resources
    if _resources is None:
        _resources = ({"vad": load_vad(), "aux": load_aux_lexicons()}
                      if _HAVE_FRAMING else {})
    return _resources


def _content(text):
    return [t for t in tokenize(text) if t not in STOP and len(t) > 2]


def _recall(summary, source):
    """Share of the source's content words that survive into the summary.

    Recall, not F1: the question is whether this source's contribution made it
    in, and a summary is expected to be much shorter than the sources combined.
    """
    src = set(_content(source))
    if not src:
        return None
    return len(src & set(_content(summary))) / len(src)


def _stance_group(stance, names):
    return any(stance == n or stance.startswith(n) for n in names)


def evaluate(summary, cluster, cfg=None, resources=None):
    res = resources if resources is not None else load_resources()
    arts = cluster["articles"]
    source_text = "\n".join(f"{a.get('title', '')} {a['body']}" for a in arts)

    m = {}
    m["words"] = len(summary.split())
    # Must match prompt.target_words: the longest article, not the mean.
    target = max((len(a["body"].split()) for a in arts), default=0)
    m["target_words"] = target
    m["length_ratio"] = m["words"] / target if target else 0.0

    # --- vocabulary the summary INTRODUCED -------------------------------
    # framing_bias calibrates as token_set(gen) - token_set(ref). With the
    # sources as the reference that is exactly the quantity of interest:
    # affect-laden words the summary added that no source used.
    if res:
        fb = framing_bias(summary, source_text, res["vad"], res["aux"])
        new_vad = fb["n_vad_tokens"] or 1
        m["new_tokens"] = fb["n_tokens"]
        m["new_vad_tokens"] = fb["n_vad_tokens"]
        m["new_p_arousal"] = round(fb["p_arousal"], 4)
        m["new_n_arousal"] = round(fb["n_arousal"], 4)
        # Per introduced VAD token, never raw alone: the raw score is a length
        # proxy (r = 0.88 with calibrated token count, eda/metric_gold_baseline.md).
        m["new_arousal_rate"] = round((fb["p_arousal"] + fb["n_arousal"]) / new_vad, 4)
        m["new_affect_tokens"] = len(_introduced_affect(summary, source_text, res["vad"]))
        m["bias_lexicon_hits"] = sum(
            int(v) for k, v in fb.items()
            if k not in {"valence", "v_positive", "v_negative", "p_arousal",
                         "n_arousal", "n_tokens", "n_vad_tokens"})

    # --- coverage, per source --------------------------------------------
    recalls = [(a, _recall(summary, f"{a.get('title', '')} {a['body']}")) for a in arts]
    vals = [r for _, r in recalls if r is not None]
    if vals:
        mean = sum(vals) / len(vals)
        m["coverage_mean"] = round(mean, 4)
        m["coverage_min"] = round(min(vals), 4)
        m["coverage_ratio"] = round(min(vals) / mean, 4) if mean else 0.0

    # --- stance parity ----------------------------------------------------
    # evaluate() sees the stance labels even when the prompt does not.
    if cfg is not None:
        left = [r for a, r in recalls
                if r is not None and _stance_group(a["stance"], cfg.left_stances)]
        right = [r for a, r in recalls
                 if r is not None and _stance_group(a["stance"], cfg.right_stances)]
        if left and right:
            m["recall_left"] = round(sum(left) / len(left), 4)
            m["recall_right"] = round(sum(right) / len(right), 4)
            m["stance_parity_gap"] = round(abs(m["recall_left"] - m["recall_right"]), 4)

    # --- shared numbers ---------------------------------------------------
    shared = _shared_numbers(arts)
    if shared:
        kept = {n for n in shared if n in set(_NUM.findall(summary))}
        m["numbers_shared"] = len(shared)
        m["number_retention"] = round(len(kept) / len(shared), 4)

    return m


def _introduced_affect(summary, source_text, vad):
    """High-arousal words present in the summary and in no source, by name."""
    new = token_set(summary, vad) - token_set(source_text, vad)
    out = []
    for t in new:
        hit = vad.get(t)
        if hit and (hit[0] > POSITIVE_V or hit[0] < NEGATIVE_V) and hit[1] > 0.5:
            out.append(t)
    return sorted(out)


def _shared_numbers(arts, min_sources=2):
    counts = {}
    for a in arts:
        for n in set(_NUM.findall(f"{a.get('title', '')} {a['body']}")):
            counts[n] = counts.get(n, 0) + 1
    return {n for n, c in counts.items() if c >= min_sources}


def failures(metrics, cfg, cluster=None, summary=None):
    """Threshold checks, phrased as instructions.

    A bare number tells a language model nothing it can act on, so each string
    carries the measurement *and* the edit it implies. Metrics the evaluator did
    not produce are skipped rather than treated as failures -- a partial
    replacement evaluator still works.
    """
    out = []
    g = metrics.get

    # The rate is a ratio over introduced affect words, so a handful of them makes
    # it wildly noisy -- one charged word can read as 0.6. Only judge it once the
    # denominator is big enough to mean something.
    if (g("new_arousal_rate") is not None
            and g("new_vad_tokens", 0) >= cfg.min_vad_tokens_for_rate
            and g("new_arousal_rate") > cfg.max_new_arousal_rate):
        out.append(
            f"The summary's emotional intensity is too high: {g('new_arousal_rate'):.3f} "
            f"arousal per newly-introduced affect word, against a ceiling of "
            f"{cfg.max_new_arousal_rate}. Replace charged wording with plain description.")

    if g("new_affect_tokens") is not None and g("new_affect_tokens") > cfg.max_new_affect_tokens:
        names = ""
        if cluster is not None and summary is not None and load_resources():
            src = "\n".join(f"{a.get('title', '')} {a['body']}" for a in cluster["articles"])
            words = _introduced_affect(summary, src, load_resources()["vad"])[:12]
            if words:
                names = " They include: " + ", ".join(words) + "."
        out.append(
            f"The summary introduces {g('new_affect_tokens')} emotionally charged words that "
            f"appear in none of the source articles (limit {cfg.max_new_affect_tokens}). "
            f"Rewrite them in neutral terms.{names}")

    if g("bias_lexicon_hits") is not None and g("bias_lexicon_hits") > cfg.max_bias_lexicon_hits:
        out.append(
            f"The summary adds {g('bias_lexicon_hits')} words from known media-bias lexicons "
            f"that no source used (limit {cfg.max_bias_lexicon_hits}). Remove the loaded "
            f"vocabulary.")

    if g("coverage_ratio") is not None and g("coverage_ratio") < cfg.min_source_coverage_ratio:
        out.append(
            f"One source is under-represented: its facts appear at "
            f"{g('coverage_ratio'):.2f} of the average source's rate (floor "
            f"{cfg.min_source_coverage_ratio}). This is bias by omission. Go back through "
            f"each source and add the substantive facts only it reports.")

    if g("stance_parity_gap") is not None and g("stance_parity_gap") > cfg.max_stance_parity_gap:
        heavier = ("left-leaning" if g("recall_left", 0) > g("recall_right", 0)
                   else "right-leaning")
        out.append(
            f"The summary tracks the {heavier} sources more closely than the others "
            f"(gap {g('stance_parity_gap'):.3f}, ceiling {cfg.max_stance_parity_gap}). "
            f"Give every source's factual contribution equal weight.")

    if g("number_retention") is not None and g("number_retention") < cfg.min_number_retention:
        out.append(
            f"Only {g('number_retention'):.0%} of the figures that two or more sources agree "
            f"on made it into the summary (floor {cfg.min_number_retention:.0%}). Restore the "
            f"missing numbers exactly as the sources give them.")

    if cfg.enforce_length and g("target_words"):
        lo, hi = 1 - cfg.length_tolerance, 1 + cfg.length_tolerance
        if not lo <= g("length_ratio", 1.0) <= hi:
            direction = "longer" if g("length_ratio") < lo else "shorter"
            out.append(
                f"The summary is {g('words')} words against a target of about "
                f"{g('target_words')}. Make it {direction} -- by adding or trimming detail, "
                f"not by dropping or padding facts.")

    return out
