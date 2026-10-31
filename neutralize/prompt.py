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

import json
import random
from pathlib import Path

PROMPT_VERSION = "v3"  # v3: source labels and source-format asides banned from the summary

# How a factual disagreement is handled. The rest of the prompt is identical, so
# the two styles are directly comparable on the same events.
DISAGREE_ATTRIBUTE = """WHERE SOURCES DISAGREE
- State both versions with neutral attribution, and record the disagreement in \
the `conflicts` field. Do not pick a winner, average the figures, or quietly drop \
the version you find less plausible.
- A fact only one source reports is not a conflict. Include it plainly."""

DISAGREE_RANGE = """WHERE SOURCES DISAGREE
- When sources give DIFFERENT NUMBERS for the same quantity, report a RANGE \
rather than listing each figure. One source saying 400 deaths and another 350 \
becomes "an estimated 350-400 deaths". Use the lowest and highest figures \
reported, smallest first, and mark it as an estimate or as a figure that varies \
by account.
- Never average the figures, never pick one, and never invent a bound that no \
source reported.
- If the sources agree on a number, state it plainly -- a single figure is not a \
range.
- When the disagreement is NOT numeric -- two accounts of what happened, of \
someone's motive, of whether an event was planned -- do not force a range and do \
not name the outlets. Report the uncertainty itself: "accounts differ on whether \
the interviews were conducted live or recorded earlier". Where one account is a \
hedged official statement and another is a flat assertion, preserve that \
difference in strength.
- Record every disagreement you reconciled in the `conflicts` field, including \
the individual figures, so the reconciliation stays auditable.
- A fact only one source reports is not a conflict. Include it plainly."""

CONTESTED_ATTRIBUTE = ('Where a fact is contested, attribute it the way a wire '
                       'report would -- "one account put the toll at 49, another '
                       'at about 20" -- or name the outlet if the source '
                       'identifies itself.')

# The worked example here is what the model actually imitates. Leaving the
# attribute-style example in place while asking for ranges produced no ranges at
# all: the model copied the example and ignored the rule above it.
CONTESTED_RANGE = ('Where a contested fact is NUMERIC, give the range and nothing '
                   'else: "an estimated 350-400 deaths", "between 12 and 15 '
                   'arrests". Do NOT write "one account says X, another says Y" '
                   'for a number -- the range replaces that phrasing entirely. '
                   'Do not name outlets.')

_STYLES = {"attribute": DISAGREE_ATTRIBUTE, "range": DISAGREE_RANGE}
_CONTESTED = {"attribute": CONTESTED_ATTRIBUTE, "range": CONTESTED_RANGE}
_STYLE_VERSION = {"attribute": "v3", "range": "v4-range"}


def _neutralise_version(cfg):
    v = getattr(cfg, "neutralise_prompt", NEUTRALISE_VERSION) or NEUTRALISE_VERSION
    if v not in NEUTRALISE_PROMPTS:
        raise ValueError(f"neutralise_prompt must be one of {sorted(NEUTRALISE_PROMPTS)}, got {v!r}")
    return v


def system_prompt(cfg):
    if getattr(cfg, "task", "fuse") == "neutralise":
        return NEUTRALISE_PROMPTS[_neutralise_version(cfg)]
    style = (getattr(cfg, "conflict_style", "attribute") or "attribute").lower()
    if style not in _STYLES:
        raise ValueError(f"conflict_style must be one of {sorted(_STYLES)}, got {style!r}")
    return (SYSTEM.replace("{DISAGREE}", _STYLES[style])
                  .replace("{CONTESTED}", _CONTESTED[style]))


NEUTRALISE_VERSION = "n2"  # n2: headline facts kept, identifying descriptors kept, no invented quote settings


def prompt_version(cfg):
    """Recorded in every output record; task and style are part of the identity."""
    if getattr(cfg, "task", "fuse") == "neutralise":
        return _neutralise_version(cfg)
    style = (getattr(cfg, "conflict_style", "attribute") or "attribute").lower()
    return _STYLE_VERSION.get(style, PROMPT_VERSION)


def response_schema(cfg):
    return (NEUTRALISE_SCHEMA if getattr(cfg, "task", "fuse") == "neutralise"
            else RESPONSE_SCHEMA)

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

{DISAGREE}

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
`Source A`, `Source B` and so on are labels for your reading only. They must not appear anywhere in `summary`. {CONTESTED} The `conflicts` field is the only place source labels belong.

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
    """Returns (prompt_text, target_word_count). Dispatches on cfg.task."""
    if getattr(cfg, "task", "fuse") == "neutralise":
        return build_neutralise_prompt(cluster, cfg)
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


# ---------------------------------------------------------------------------
# Single-article neutralisation
#
# A different task from fusion: one article in, the same article out with its
# framing removed. Downstream the model will only ever see one source, so the
# target must be reachable from that one source and nothing else. The v3 fusion
# corpus measurably failed this -- 9.6% of its outputs stated a calendar date no
# source gave, because the model resolved "Friday" from world knowledge. The
# hard rule below exists specifically to stop that.
# ---------------------------------------------------------------------------

NEUTRALISE_SYSTEM = """\
You are a copy editor. You are given ONE news article. Rewrite it so that it \
reports the same events with the framing removed.

This is a rewrite, not a summary and not a fact-check. The output should be \
about as long as the input.

THE ONE HARD RULE: ADD NOTHING
Everything in your rewrite must come from the article in front of you. You have \
no other knowledge of this story, these people, or this period, and you must \
write as though you have none.

Specifically, never add:
- a date, day or year the article does not state. If it says "Friday", write \
"Friday". Do not work out the calendar date. If it gives no date, give none.
- a location, job title, party, age or affiliation the article does not give.
- background, context or explanation of who someone is or why something matters.
- a consequence, cause or outcome the article does not report.
- a number the article does not contain, including totals you could compute.
- a speaker, audience, medium or setting for a quotation. If the article gives \
a quotation or a fragment without saying who said it, where, or in what form, \
your rewrite must not say either. Write it as: The phrase "..." was also \
reported. Nothing more. Never write "a message read", "a sign said", "a witness \
said", "he told reporters" or "in a statement" unless the article itself says \
so. Joining fragments into continuous prose is not permission to invent how \
they connect.

If you find yourself about to write something you know but the article does not \
say, stop and leave it out. An article that says less than you know is correct.

KEEP EVERY FACT
Removing framing is not removing content. Every event, actor, figure, date, \
place and quotation in the article must survive into your rewrite. If you cannot \
neutralise a sentence without dropping a fact, keep the fact and neutralise the \
wording around it.

- The headline is part of the article. Every fact in it that the body does not \
repeat -- who, where, how many, what kind of place or group -- must appear in \
your prose. Drop only the headline's form: labels such as "Watch:" or \
"BREAKING", title-case capitalisation, teaser phrasing.
- Copy numbers, names, titles and dates exactly as the article gives them.
- Copy direct quotations character for character. Never paraphrase inside \
quotation marks, never merge two quotations, never trim a quotation to change \
its sense. If a quotation is itself loaded, keep it and attribute it -- what a \
person said is a fact about them.

WHAT TO REMOVE

These are AllSides' types of media bias. Work through the article and strip each \
one, changing nothing else.

1.  SPIN -- vague or evocative phrasing that steers the reading. Replace with \
the plain description of what happened.
2.  SLANT -- one angle promoted by what gets elaborated. Give each fact the \
space its importance warrants, not the space the article gave it.
3.  SENSATIONALISM -- shocking or exaggerated wording. Cut intensifiers \
(`massive`, `explosive`, `stunning`, `slams`, `erupts`, `blasts`), scare quotes, \
and exclamation.
4.  BIAS BY OMISSION -- you cannot fix what the article left out, and you must \
not fill the gap from your own knowledge. Leave the gap.
5.  STORY CHOICE and PLACEMENT -- open with the event itself, not with the \
reaction to it, the accusation about it, or one side's characterisation of it.
6.  WORD CHOICE -- prefer the descriptive term to the loaded one. Where a term \
is contested, use the neutral descriptive form. A descriptor that IDENTIFIES \
who or what was involved -- sexual orientation, religion, nationality, \
ethnicity, party, occupation, as in "a gay nightclub" or "a Republican \
senator" -- is a fact, not bias. Keep it. Remove a descriptor only when it \
EVALUATES rather than identifies.
7.  UNSUBSTANTIATED CLAIMS -- an assertion the article does not support becomes \
attributed to whoever made it, or goes.
8.  OPINION PRESENTED AS FACT -- judgement in the reporter's voice must be \
attributed to a named person, or cut.
9.  SUBJECTIVE QUALIFYING ADJECTIVES -- `controversial`, `so-called`, `radical`, \
`failed`, `hardline`, `extreme`, `disgraced`. Cut them. If a label is contested, \
report who applies it.
10. MUDSLINGING / AD HOMINEM -- keep the substance of a criticism, attributed; \
drop the abuse.
11. MIND READING -- no sentence may assert what a person thinks, wants, fears or \
really means unless the article quotes them saying it.
12. NEGATIVITY BIAS -- report outcomes and magnitudes as given, without \
dramatising.
13. FLAWED LOGIC -- do not connect two facts causally unless the article makes \
that claim explicitly; where it does, attribute it.
14. ELITE V. POPULIST -- attribute officials and ordinary people in the same \
register.

ATTRIBUTION VERBS
Use `said`, `wrote`, `reported`, `stated`, `told`. Never `admitted`, `conceded`, \
`claimed`, `alleged`, `insisted`, `boasted`, `refused to say` -- each editorialises \
about the speaker's honesty or motive.

FORM
Continuous prose. No headline, no byline, no dateline, no bullets, no section \
headers, no preamble. Do not mention the article, the outlet, or that you are \
rewriting anything. Begin with the first sentence of the account itself."""


NEUTRALISE_SCHEMA = {
    "type": "object",
    "properties": {
        "neutral": {
            "type": "string",
            "description": "The article rewritten with its framing removed.",
        },
        "removed": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "One short PLAIN SENTENCE per edit, naming the bias type and the "
                "wording changed, e.g. 'Sensationalism: \"slammed\" -> \"criticised\"'. "
                "The audit trail for the rewrite. Empty list if nothing needed "
                "changing."
            ),
        },
    },
    "required": ["neutral", "removed"],
}


# ---------------------------------------------------------------------------
# Prompt family (generation-setting experiments). Each variant is n2 with ONE
# change, so a difference in the outputs can be pinned on that change. n2
# itself is untouched and stays the default.
# ---------------------------------------------------------------------------

def _n2_with(old, new):
    assert NEUTRALISE_SYSTEM.count(old) == 1, old
    return NEUTRALISE_SYSTEM.replace(old, new)


_FORM = "\n\nFORM\n"
assert NEUTRALISE_SYSTEM.count(_FORM) == 1

# n3: fixes for the two defects measured on n2. "The article notes/argues ..."
# (13% of left outputs) is rule 8 misfiring -- with no named person to attribute
# a judgement to, the model attributes it to the article, so the opinion
# survives under a new label. And titles/abbreviations get expanded ("Sen." ->
# "Senator", "R-Texas" -> "Republican"), which adds words the article lacks.
_RULE8 = ("8.  OPINION PRESENTED AS FACT -- judgement in the reporter's voice must be "
          "attributed to a named person, or cut.")
NEUTRALISE_N3 = _n2_with(
    _RULE8,
    _RULE8 + " The article, the outlet, the reporter, \"the report\" and \"observers\" are not "
    "people: never write \"The article notes\", \"The article argued\", \"The piece "
    "described\" or \"According to the report\". If the only source of a judgement is "
    "the writer, cut the judgement and keep any fact inside it."
).replace(
    "- Copy numbers, names, titles and dates exactly as the article gives them.",
    "- Copy numbers, names, titles and dates exactly as the article gives them. Keep "
    "abbreviations as written: do not expand \"Sen.\" to \"Senator\", \"R-Texas\" to "
    "\"Republican of Texas\", or a surname to a full name.")

# Aggression ladder: n2 plus an EDIT STRENGTH section placed just before FORM.
NEUTRALISE_LIGHT = NEUTRALISE_SYSTEM.replace(_FORM, """

EDIT STRENGTH: LIGHT
Change words and phrases only: loaded and sensational wording, intensifiers, \
scare quotes, subjective adjectives, editorialising attribution verbs, and \
opinion stated in the reporter's voice. Keep the article's sentence order, \
paragraph structure and emphasis as they are: do not move, merge, split or \
rebalance sentences, and do not reorder the opening. This overrides items 2, 5 \
and 12 above. A sentence with no loaded wording is copied unchanged.""" + _FORM)

NEUTRALISE_HEAVY = NEUTRALISE_SYSTEM.replace(_FORM, """

EDIT STRENGTH: THOROUGH
Do not stop at word swaps. Rebuild the article as wire copy: open with the core \
event stated plainly; order the remaining facts by news importance, not by the \
order the article chose; where the article reports more than one side, give \
each the same treatment and the same register; and move every statement of \
motive, significance or likely consequence into attributed speech, or cut it. \
Every sentence should read as something a wire service would print. The hard \
rule and KEEP EVERY FACT still apply in full.""" + _FORM)

# Minimal: the same task in one paragraph, without the AllSides taxonomy.
# Tests whether the long rule list earns its tokens.
NEUTRALISE_MIN = """\
You are a copy editor. Rewrite the news article you are given so that it reports \
the same events neutrally, as a wire service would: remove loaded, evaluative \
and sensational wording, opinion in the writer's voice, and slanted emphasis. \
Keep every fact, name, number, date and quotation (quotations character for \
character). Add nothing the article does not say -- no dates, titles, background \
or context from your own knowledge. Keep it about as long as the input. Write \
continuous prose: no headline, no preamble, and do not mention the article or \
that it was rewritten."""

# Paraphrase control: same LLM, same constraints, but told to KEEP the framing.
# The classifier drop original -> paraphrase is the "new wording" effect; the
# extra drop original -> neutral rewrite is what neutralisation itself buys.
NEUTRALISE_PARA = """\
You are a copy editor. You are given ONE news article. Reword it so that it says \
the same things in different words while keeping its point of view exactly.

This is a control condition for an experiment: the rewording must KEEP the \
article's framing. Keep its slant, its emphasis, the order of its material, its \
tone, its opinions stated in the writer's voice, and what it chooses to \
elaborate. Replace a loaded word with an equally loaded one ("slammed" -> \
"blasted"), never with a neutral one. Do not soften, balance, hedge or \
attribute anything the article states in its own voice.

Change the wording of nearly every sentence: different vocabulary and sentence \
structure, the same meaning and the same slant.

ADD NOTHING
Everything in your rewording must come from the article in front of you: no \
date, title, location, number, background or context it does not give.

KEEP EVERY FACT
Every event, actor, figure, date, place and quotation must survive, including \
the facts in the headline. Copy numbers, names and dates exactly. Copy direct \
quotations character for character.

FORM
Continuous prose. No headline, no byline, no dateline, no bullets, no preamble. \
Do not mention the article, the outlet, or that you are rewording anything. \
Begin with the first sentence of the account itself."""

# AllSides "How to Spot Types of Media Bias" (allsides.com/media-bias/how-to-spot-types-of-media-bias),
# used two ways on top of n3, both placed just before FORM.
# n3_lex: the page's word lists, as signals to check rather than a ban list.
_ALLSIDES_LEX = """

WATCH LIST
AllSides' examples of each bias type. These words are signals to check, not a \
ban list: in the writer's own voice, replace or cut them unless they state a \
literal fact ("in critical condition", "a significant difference" in a study, \
"a landmark" as a building). Inside a direct quotation, keep them.
- Spin: emerge, serious, refuse, crucial, high-stakes, tirade, landmark, "latest \
in a string of", major, "turn up the heat", critical, decrying, offend, "stern \
talks", offensive, "facing calls to", meaningful, "even though", monumental, \
significant.
- Implying bad behaviour without evidence: finally, surfaced, acknowledged, \
emerged, "refusing to say", conceded, dodged, admission, "came to light", \
"admit to".
- Dramatic substitutes for "said": mocked, raged, bragged, fumed, "lashed out", \
incensed, scoffed, erupted, rant, boasted, gloated.
- Sensationalism: shocking, remarkable, chaotic, scathing, explosive, fiery, \
desperate, rip, blast, rebuke, slam, onslaught, showdown, "breaking".
- Subjective judgement in the writer's voice: good/better/best, bad/worse/worst, \
"is considered to be", seemingly, apparently, "it's likely that", "may mean \
that", "would seem", suggests, possibly, dangerous, extreme.
- Subjective qualifying adjectives: "disturbing rise", "serious accusations", \
"troubling trend", "sinister warning", "extreme law", "baseless claim", \
"debunked theory", "critical bill", "offensive statement", "harsh rebuke", \
"extremist group", "far-right"/"far-left" as a label. Report the allegation, \
claim or law itself and let the reader judge it.
- Polarising terms, where each side has its own word: pro-choice/anti-choice, \
pro-abortion/anti-abortion, gun rights/gun control, riot/protest, illegal \
immigrants/migrants, illegal alien/asylum-seeking migrants, voting \
rights/voting security, sex reassignment surgery/gender-affirming care. In the \
writer's voice, describe what happened rather than adopting either side's \
term; if a term is the official name of a law, group or policy, keep it."""

# n3_fs: worked before -> after pairs built from the excerpts AllSides marks up
# on that page (the page shows them as screenshots; the "after" text is ours).
_ALLSIDES_SHOTS = """

WORKED EXAMPLES
Short excerpts that AllSides uses to illustrate media bias, each with an \
acceptable neutral rewrite. They show the edits at sentence scale; apply the \
same edits across the whole article. They are not about your article: never \
carry a name, fact or phrase from them into your rewrite.

Example 1 (dramatic substitute for "said")
Before: President Donald Trump gloated over mass layoffs at multiple news \
outlets on Saturday, tweeting that "many others will follow" and attributing \
the loss to "Fake News."
After: President Donald Trump tweeted on Saturday about mass layoffs at \
multiple news outlets, writing that "many others will follow" and attributing \
the loss to "Fake News."

Example 2 (spin verbs)
Before: Headline: Biden slams SCOTUS presidential immunity ruling, ignores \
questions about dropping out. Biden's address lasted four minutes and forty \
seconds before he bolted and dodged questions from reporters.
After: Biden criticized the SCOTUS presidential immunity ruling in an address \
that lasted four minutes and forty seconds. He left without answering \
reporters' questions, including questions about dropping out.

Example 3 (subjective qualifying adjectives, mudslinging)
Before: Headline: Rep. Madison Cawthorn Issues Sinister Warning To Anyone \
Opposing Him. The extremist Republican ranted about liberals trying to make \
people "sexless," then threatened people who "stand in the way."
After: Rep. Madison Cawthorn, a Republican, said liberals were trying to make \
people "sexless" and issued a warning to anyone opposing him, referring to \
people who "stand in the way."

Example 4 (mind reading)
Before: Donald Trump's hatred of looking foolish and Democrats' conviction \
that they have a winning hand is leaving the President with no way out of the \
stalemate over his border wall.
After: President Donald Trump and Democrats remain in a stalemate over his \
border wall.

Example 5 (opinion presented as fact)
Before: Twitter banned or suspended several high-profile journalists Thursday \
evening, a move that further reveals the seemingly arbitrary decision-making \
of Elon Musk, a self-avowed "free speech absolutist."
After: Twitter banned or suspended several high-profile journalists Thursday \
evening. Elon Musk has described himself as a "free speech absolutist."

Example 6 (slant and placement: lead with the event, keep the facts)
Before: Vice President Harris mourned the Palestinians who were killed in a \
successful Israeli hostage rescue operation this weekend during an address \
delivered Saturday. The Israeli operation retrieved four hostages who were \
taken into Gaza by Hamas on Oct. 7. Harris, speaking before the Michigan \
Democratic Party, celebrated their retrieval but spent more time lamenting \
the deaths of Palestinians killed in the operation, many of whom Israel has \
described as terrorists.
After: An Israeli operation this weekend retrieved four hostages who were \
taken into Gaza by Hamas on Oct. 7. In an address to the Michigan Democratic \
Party on Saturday, Vice President Harris welcomed their retrieval and spoke at \
greater length about the Palestinians killed in the operation, many of whom \
Israel has described as terrorists.

Example 7 (sensational headline words, editorialising attribution)
Before: Headline: Trump claims he ordered 'obliteration' of Iran if he's \
assassinated in chilling warning: 'I've left instructions'
After: Trump said he has ordered the "obliteration" of Iran if he is \
assassinated. "I've left instructions," he said.\""""

assert NEUTRALISE_N3.count(_FORM) == 1
NEUTRALISE_N3_LEX = NEUTRALISE_N3.replace(_FORM, _ALLSIDES_LEX + _FORM)
NEUTRALISE_N3_FS = NEUTRALISE_N3.replace(_FORM, _ALLSIDES_SHOTS + _FORM)

NEUTRALISE_PROMPTS = {
    "n2": NEUTRALISE_SYSTEM,
    "n3": NEUTRALISE_N3,
    "n3_lex": NEUTRALISE_N3_LEX,
    "n3_fs": NEUTRALISE_N3_FS,
    "n2_light": NEUTRALISE_LIGHT,
    "n2_heavy": NEUTRALISE_HEAVY,
    "n2_min": NEUTRALISE_MIN,
    "para": NEUTRALISE_PARA,
    # Whole-article few-shot: the n3 system prompt unchanged, plus the first k
    # pairs of fewshot_full1024.json sent as earlier user/assistant turns.
    "n3_k2": NEUTRALISE_N3,
    "n3_k4": NEUTRALISE_N3,
    # Human-written few-shot: 4 NeuS excerpts, each with its AllSides roundup as
    # the target (see fewshot_neus.json for why those roundups qualify).
    "n3_neus_k4": NEUTRALISE_N3,
}
FEW_SHOT = {"n3_k2": ("fewshot_full1024.json", 2), "n3_k4": ("fewshot_full1024.json", 4),
            "n3_neus_k4": ("fewshot_neus.json", 4)}


def few_shot_messages(cfg):
    """The example turns that go between the system prompt and the article.

    Examples come from fewshot_full1024.json (train-split n2 rewrites) or
    fewshot_neus.json (human AllSides roundups), and are shown exactly as a real
    request and a real reply: the same user prompt build_neutralise_prompt makes,
    and the JSON the model is asked to return. An example's target_words, when
    set, replaces the input length in its prompt (a roundup is shorter than its
    excerpt). Only the NVIDIA client sends them.
    """
    if getattr(cfg, "task", "fuse") != "neutralise":
        return []
    if _neutralise_version(cfg) not in FEW_SHOT:
        return []
    name, k = FEW_SHOT[_neutralise_version(cfg)]
    msgs = []
    for ex in json.loads(Path(__file__).with_name(name).read_text(encoding="utf-8"))[:k]:
        user, _ = build_neutralise_prompt({"articles": [ex], "_target_words": ex.get("target_words")}, cfg)
        reply = json.dumps({"neutral": ex["neutral"], "removed": ex["removed"]}, ensure_ascii=False)
        msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]
    return msgs


def build_neutralise_prompt(cluster, cfg):
    """One article in. Returns (prompt_text, target_word_count).

    The 'cluster' carries exactly one article; the target length is that
    article's own length, since this is a rewrite rather than a compression.
    """
    a = cluster["articles"][0]
    target = cluster.get("_target_words") or len(a["body"].split())
    title = (a.get("title") or "").strip()
    if _neutralise_version(cfg) == "para":
        head, task, removed = ("Reword the following article, keeping its framing.",
                               "Reword it. Keep every fact and the article's slant",
                               "one short sentence per significant rewording")
    else:
        head, task, removed = ("Rewrite the following article with its framing removed.",
                               "Rewrite it. Keep every fact",
                               "one short sentence per bias edit you made")
    return (
        f"{head}\n\n"
        "=== ARTICLE ===\n"
        + (f"Headline: {title}\n" if title else "")
        + a["body"].strip()
        + "\n\n=== TASK ===\n"
        f"{task}, add nothing that is not above, and aim "
        f"for roughly {target} words.\n\n"
        "Return JSON with two keys: `neutral` (the rewritten article) and "
        f"`removed` ({removed})."
    ), target
