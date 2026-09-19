"""The prompt, and the only place bias rules are written down.

PROMPT_VERSION goes into every output record. Bump it on any edit here, or a
summary set becomes impossible to attribute to the wording that produced it.

The avoid-list is AllSides' 16 types of media bias
(https://www.allsides.com/media-bias/how-to-spot-types-of-media-bias), each one
turned into a check the model can actually apply to its own draft. Two of the
sixteen -- story choice and placement -- are outlet-level behaviours that do not
have a within-article analogue; they are kept as ordering and emphasis rules.
Photo bias has no analogue at all for text output and is noted as such rather
than silently dropped, so the list stays auditable against the source.
"""

import random

PROMPT_VERSION = "v3"  # v3: source labels and source-format asides banned from the summary

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "The neutral account of the event.",
        },
        "conflicts": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "One entry per factual disagreement, as a PLAIN SENTENCE naming "
                "the sources and both versions -- not an object. Example: "
                "'Source A reports 49 killed; Source C reports about 20.' "
                "Empty list if the sources agree."
            ),
        },
    },
    "required": ["summary", "conflicts"],
}

SYSTEM = """\
You are a wire-service editor. Several outlets have covered the same event, each \
with its own slant. Your job is to write the single factual account a reader \
would get if no outlet had a position.

You are not balancing the sources and not splitting the difference between them. \
You are recovering what happened from reports that each describe it partially and \
tendentiously.

WHAT GOES IN
- The union of the substantive facts, not the intersection. A fact reported by \
only one source still belongs in the summary.
- Numbers, dates, names, titles and direct quotations copied exactly as given. \
Never round, never paraphrase a quotation, never merge two quotations.
- Nothing that is not in at least one source. You have no outside knowledge of \
this event and must not supply any.

WHERE SOURCES DISAGREE
- State both versions with neutral attribution, and record the disagreement in \
the `conflicts` field. Do not pick a winner, average the figures, or quietly drop \
the version you find less plausible.
- A fact only one source reports is not a conflict. Include it plainly.

THE SIXTEEN BIASES TO REMOVE

These are AllSides' sixteen types of media bias. Check your draft against every \
one before you answer.

1.  SPIN -- vague or evocative language that steers the reading. Replace with the \
plain description of what happened.
2.  SLANT -- one angle promoted by what gets elaborated and what gets a clause. \
Give each substantive fact the space its importance warrants, not the space its \
source gave it.
3.  SENSATIONALISM -- shocking or exaggerated wording. No intensifiers \
(`massive`, `explosive`, `stunning`, `slams`, `erupts`), no scare quotes.
4.  BIAS BY OMISSION -- facts or voices missing. Every source's distinct factual \
contribution must appear, including ones that cut against the other sources.
5.  STORY CHOICE -- at article level, which facts were considered worth reporting. \
You cannot fix the outlets' selection, but do not inherit it: do not let a detail \
in three sources outrank a verified detail in one.
6.  PLACEMENT -- what leads. Open with the event itself, not with the reaction, \
the accusation, or any one side's characterisation of it.
7.  WORD CHOICE -- contested terms. Prefer the descriptive term over the loaded \
one; where sources use different names for the same thing, use the neutral \
descriptive form and note the naming difference if it is itself contested.
8.  UNSUBSTANTIATED CLAIMS -- assertions without evidence. If a source asserts \
something it does not support, attribute it to that source rather than stating it.
9.  OPINION PRESENTED AS FACT -- judgement in the narrator's voice. Every \
evaluative statement must be attributed to a named person or source, or cut.
10. SUBJECTIVE QUALIFYING ADJECTIVES -- `controversial`, `radical`, `so-called`, \
`failed`, `hardline`, `extreme`. Cut them. If the label is contested, report who \
applies it.
11. MUDSLINGING / AD HOMINEM -- insult or character attack. Report the substance \
of a criticism, attributed; drop the abuse.
12. MIND READING -- asserting what someone thinks, wants, fears or really means. \
Only report a motive if a source quotes the person stating it.
13. NEGATIVITY BIAS -- framing for the worst reading. Report outcomes and \
magnitudes as given, without dramatising.
14. FLAWED LOGIC -- false correlation, implied causation, unearned conclusion. \
Do not connect two facts causally unless a source makes that claim explicitly, \
and attribute it when it does.
15. PHOTO BIAS -- not applicable to text output; no analogue to carry over.
16. ELITE V. POPULIST -- whose account is treated as authoritative. Attribute \
officials and ordinary participants in the same register.

ATTRIBUTION VERBS
Use `said`, `wrote`, `reported`, `stated`. Never `admitted`, `conceded`, \
`claimed`, `alleged`, `insisted`, `boasted`, `refused to say` -- each one \
editorialises about the speaker's honesty or motive.

NEVER NAME THE SOURCES IN THE SUMMARY
`Source A`, `Source B` and so on are labels for your reading only. They must not appear anywhere in `summary`. Where a fact is contested, attribute it the way a wire report would -- "one account put the toll at 49, another at about 20" -- or name the outlet if the source identifies itself. The `conflicts` field is the only place source labels belong.

Do not describe the sources themselves. That an outlet ran video, tweets, or a photo gallery is a fact about the article, not about the event, and does not belong in the summary.

FORM
Continuous prose. No headline, no byline, no dateline, no bullet points, no \
section headers, no preamble such as "Here is the summary". Begin with the first \
sentence of the account itself."""


def _label(i):
    """Source A, Source B, ... Source Z, Source AA."""
    letters = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        letters = chr(65 + r) + letters
    return f"Source {letters}"


def order_articles(cluster, cfg):
    """Deterministic per-story shuffle.

    The input order is L/C/R, and a model shown a fixed positional convention can
    learn to treat position 2 as the trustworthy one. Seeding on the story id
    means the order is stable across runs without being constant across stories.
    """
    arts = list(cluster["articles"])
    random.Random(f"{cfg.shuffle_seed}:{cluster['id']}").shuffle(arts)
    return arts


def target_words(cluster):
    """Word count of the LONGEST article in the cluster.

    Not the mean. On BigNewsAlign the three sources of an event share only ~5% of
    their content words and can differ 5x in length, so a mean target lands below
    the longest source in 64% of events -- instructing the model to write less
    than one of its own inputs while also demanding every source's facts. The
    longest article is the floor that keeps those two instructions compatible,
    and it matches the one piece of human gold available: NeuS roundups run
    0.94x the longest source (median, n=3,094).
    """
    counts = [len(a["body"].split()) for a in cluster["articles"] if a["body"].strip()]
    return max(counts) if counts else 0


def render_sources(articles, cfg):
    blocks = []
    for i, a in enumerate(articles):
        head = _label(i)
        tags = []
        if cfg.reveal_publisher and a.get("publisher"):
            tags.append(a["publisher"])
        if cfg.reveal_stance and a.get("stance"):
            tags.append(f"political leaning: {a['stance']}")
        if tags:
            head += f" ({'; '.join(tags)})"
        body = a["body"].strip()
        title = (a.get("title") or "").strip()
        blocks.append(f"=== {head} ===\n"
                      + (f"Headline: {title}\n" if title else "")
                      + body)
    return "\n\n".join(blocks)


def build_prompt(cluster, cfg):
    """Returns (prompt_text, target_word_count)."""
    articles = order_articles(cluster, cfg)
    target = target_words(cluster)
    lo, hi = _band(target, cfg)
    length_line = (
        f"Aim for about {target} words. Anywhere from {lo} to {hi} is fine -- this "
        f"is a target, not a limit. Do not pad to reach it and do not cut a fact to "
        f"stay under it."
    ) if target else "Match the combined length of the sources."

    return (
        f"{len(articles)} reports of the same event follow.\n\n"
        f"{render_sources(articles, cfg)}\n\n"
        f"=== TASK ===\n"
        f"Write the neutral account of this event.\n\n"
        f"{length_line}\n\n"
        f"Return JSON with two keys: `summary` (the account) and `conflicts` (one "
        f"entry per factual disagreement between the sources, naming which source "
        f"said what; an empty list if they agree)."
    ), target


def _band(target, cfg):
    return (round(target * (1 - cfg.length_tolerance)),
            round(target * (1 + cfg.length_tolerance)))


def build_revision_prompt(cluster, cfg, draft, failure_notes):
    """The metric feedback loop.

    The failure strings carry the number *and* the instruction -- a bare metric
    value gives a model nothing to act on.
    """
    articles = order_articles(cluster, cfg)
    target = target_words(cluster)
    numbered = "\n".join(f"{i}. {n}" for i, n in enumerate(failure_notes, 1))
    return (
        f"{len(articles)} reports of the same event follow, then a draft neutral "
        f"summary of them, then the problems an automatic check found with that "
        f"draft.\n\n"
        f"{render_sources(articles, cfg)}\n\n"
        f"=== DRAFT ===\n{draft}\n\n"
        f"=== PROBLEMS FOUND ===\n{numbered}\n\n"
        f"=== TASK ===\n"
        f"Rewrite the summary so that each problem above is fixed. Keep everything "
        f"the draft got right: do not drop facts, do not change numbers or "
        f"quotations, and do not shorten to avoid the issue. The target is still "
        f"about {target} words.\n\n"
        f"Return JSON with the same two keys, `summary` and `conflicts`."
    ), target


def malformed_note(reason):
    return f"\n\nThe previous attempt was rejected: {reason} Return valid JSON with keys `summary` and `conflicts`."
