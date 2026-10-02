"""AllSides "16 types of media bias", scored per text with Jev.

One request per text carries every question (answers are independent, so batching them
changes cost, not results). Covers the types a single text can show; story choice, story
placement and photo bias are not observable here. Presence is a Noul per type; direction is
a Choice for the types that have a side.

    python jev/bias_types.py probe --n 4
    python jev/bias_types.py run --stories 750 --budget-usd 0.25
"""
import argparse, asyncio, json, random
from pathlib import Path

from typesafe_sdk import Choice, Noul

from jev_classify import OUT, Ledger, api_key, load_items, score_all, QUESTIONS as CLASSIFY_Q
import jev_classify

N = lambda q: Noul(instructions=q)

PRESENCE = {
    "spin": N("Does the article use spin: vague or dramatic wording (e.g. 'lashed out', 'refused', "
              "'came to light') that pushes an impression of the people involved beyond the facts reported?"),
    "sensationalism": N("Does the article use sensational, exaggerated language (e.g. 'shocking', 'explosive', "
                        "'slams', 'chaos') to make the story seem more dramatic than the facts?"),
    "loaded_words": N("Does the article use politically loaded terms that one side prefers (e.g. 'illegal alien' vs "
                      "'undocumented migrant', 'gun rights' vs 'gun control', 'riot' vs 'protest')?"),
    "subjective_adjectives": N("Does the article tell the reader how to judge something with subjective qualifying "
                               "adjectives (e.g. 'troubling trend', 'baseless claim', 'extreme law')?"),
    "opinion_as_fact": N("Does the article state opinions or interpretations as if they were facts, without "
                         "attributing them to anyone?"),
    "unsubstantiated": N("Does the article make factual claims without giving any evidence, data or source for them?"),
    "unattributed_sources": N("Does the article cite vague unnamed sources such as 'critics say', 'experts warn' or "
                              "'some argue' without saying who they are?"),
    "mudslinging": N("Does the article insult or attack a person's character or motives instead of their arguments?"),
    "mind_reading": N("Does the article claim to know what someone thinks, feels or intends without evidence such "
                      "as a quote?"),
    "negativity": N("Does the article frame events in a more negative light than the facts reported require?"),
    "one_sided": N("Does the article present only one side of a contested issue, quoting or explaining just one "
                   "side's view?"),
    "flawed_logic": N("Does the article draw a conclusion or causal link that does not follow from the evidence it gives?"),
}

SIDE = {"left": "Favours the political left: Democrats, progressives or liberal positions",
        "right": "Favours the political right: Republicans, conservatives or conservative positions",
        "neither": "Favours neither side, or there is no political side to favour"}

DIRECTION = {
    "word_choice_side": Choice(
        instructions="Whose vocabulary does the article use for contested political topics?",
        criteria={"left": "Terms the political left prefers (e.g. 'undocumented', 'gun safety', 'gender-affirming care')",
                  "right": "Terms the political right prefers (e.g. 'illegal alien', 'gun rights', 'pro-life')",
                  "neither": "Neutral terms, or no contested terms at all"}),
    "slant_side": Choice(instructions="Which side does the article's selection and emphasis of facts favour?",
                         criteria=SIDE),
    "viewpoint_first_side": Choice(
        instructions="Whose viewpoint does the article give the most prominent position, such as the first quotes "
                     "or the opening framing?", criteria=SIDE),
    "attack_target": Choice(
        instructions="If the article attacks or disparages people, which side are they on?",
        criteria={"left": "People on the left: Democrats, progressives, liberals",
                  "right": "People on the right: Republicans, conservatives, Trump",
                  "none": "The article attacks or disparages no one"}),
    "elite_populist": Choice(
        instructions="Whose perspectives does the article defer to?",
        criteria={"elite": "Prestigious institutions: government agencies, universities, experts, large companies",
                  "populist": "Ordinary people or dissenters outside prestigious institutions",
                  "neither": "Neither emphasis"}),
}

QUESTIONS = {**PRESENCE, **DIRECTION, "outlet_style": CLASSIFY_Q["outlet_style"]}


def flatten(name, ans):
    return ans.noul if hasattr(ans, "noul") else dict(ans.probabilities)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "run"])
    ap.add_argument("--stories", type=int, default=750)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--budget-usd", type=float, default=0.25, help="spend allowed for this invocation")
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()

    jev_classify.QUESTIONS.update(QUESTIONS)          # score_all looks questions up by name
    jev_classify.probs = flatten                      # store every answer's full distribution
    ledger = Ledger(0)
    ledger.cap = (ledger.usd + a.budget_usd) / jev_classify.PRICE_PER_TOKEN
    print(f"spent before ${ledger.usd:.4f}; this run may spend up to ${a.budget_usd:.2f}", flush=True)

    labelled, summaries = load_items()
    by_story = {}
    for it in labelled + summaries:
        by_story.setdefault(it["story_id"], []).append(it)
    test_stories = sorted(s for s, v in by_story.items() if v[0]["split"] == "test")
    pick = random.Random(7).sample(test_stories, a.stories if a.mode == "run" else 1)
    items = [it for s in pick for it in by_story[s]]
    if a.mode == "probe":
        items = items[: a.n]
    f = OUT / ("bias_types_probe.jsonl" if a.mode == "probe" else "bias_types.jsonl")
    asyncio.run(score_all(items, list(QUESTIONS), True, f, ledger, 2600, a.concurrency))


if __name__ == "__main__":
    main()
