"""Zero-shot political-ideology classification with TypeSafe Jev (System One).

Side experiment: can a calibrated judgment model, with no training on our labels, classify
news articles as Left / Centre / Right as well as the fine-tuned BERT classifiers?

  dev   score a balanced sample of labelled articles with every prompt design at once
        (all questions share one request, so the article is paid for once)
  run   score every text in stories_10k with one chosen design

Spend is tracked in results/spend.json across all runs and hard-capped (default $0.95).
The API key is read from ../.env (JEV_API) and never printed.
"""
import argparse, asyncio, json, random, time
from pathlib import Path

from typesafe_sdk import AsyncTypeSafeClient, Choice, Score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "stories_10k.jsonl"
OUT = Path(__file__).resolve().parent / "results"
PRICE_PER_TOKEN = 0.042 / 1e6          # jev-latest, input tokens only; output is free
SLOT2ID = {"left": 0, "center": 1, "right": 2}
CLASSES = ["left", "centre", "right"]

# ------------------------------------------------------------------ prompt designs
QUESTIONS = {
    "choice_simple": Choice(
        instructions="What is the political leaning of this news article?",
        criteria={"left": "Left-leaning or progressive",
                  "centre": "Centrist or balanced, no clear partisan leaning",
                  "right": "Right-leaning or conservative"}),

    "choice_rich": Choice(
        instructions={
            "question": "Which US political leaning does the `article` reflect, in the left / centre / "
                        "right sense used by media-bias raters such as AllSides?",
            "focus": "Judge how the story is told: which facts are emphasised, whose views are quoted "
                     "and how sympathetically, and loaded or neutral word choice. The topic alone does "
                     "not decide the leaning.",
        },
        criteria={
            "left": {"what": "Framing sympathetic to Democrats or progressive positions, or critical of "
                             "Republicans and conservatives",
                     "cues": "emphasis on inequality, climate, civil rights, gun control, immigrants' "
                             "welfare; critical framing of Trump or the GOP"},
            "centre": {"what": "Straight news: neutral wording, balanced sourcing, reports what each side "
                               "says without endorsing either",
                       "not_for": "Articles that quote both sides but clearly frame one favourably"},
            "right": {"what": "Framing sympathetic to Republicans or conservative positions, or critical "
                              "of Democrats and progressives",
                      "cues": "emphasis on border security, crime, taxes, religious liberty, gun rights; "
                              "critical framing of Democrats, the media or 'the left'"},
        }),

    "score5": Score(
        instructions="Where does the political slant of this news article sit on the US left-right spectrum?",
        criteria=["Strongly left: openly partisan for progressives or Democrats",
                  "Leans left: mostly neutral but framing favours the left",
                  "Centre: balanced, neutral straight news",
                  "Leans right: mostly neutral but framing favours the right",
                  "Strongly right: openly partisan for conservatives or Republicans"]),

    "outlet_style": Choice(
        instructions="Judging from its writing style, story selection and framing, what kind of news outlet "
                     "most likely published this article?",
        criteria={"left": "A left-leaning outlet",
                  "centre": "A centrist outlet or neutral wire service",
                  "right": "A right-leaning outlet"}),
}


def probs(name, ans):
    """Map any answer to a [p_left, p_centre, p_right] vector."""
    p = ans.probabilities
    if name == "score5":
        q = [p.get(i, p.get(str(i), 0.0)) for i in range(5)]   # SDK keys are ints
        return [q[0] + q[1], q[2], q[3] + q[4]]
    return [p.get(c, 0.0) for c in CLASSES]


# ------------------------------------------------------------------ data
def load_items():
    labelled, summaries = [], []
    for line in DATA.open(encoding="utf-8"):
        s = json.loads(line)
        for slot in ("left", "center", "right"):
            a = s[slot]
            labelled.append({"id": a["article_id"], "story_id": s["story_id"], "split": s.get("split"),
                             "slot": slot, "label": SLOT2ID[slot], "publisher": a["publisher"],
                             "title": a["title"], "text": a["body"]})
        summaries.append({"id": f'{s["story_id"]}:neutral', "story_id": s["story_id"], "split": s.get("split"),
                          "slot": "neutral", "label": None, "publisher": None, "title": None, "text": s["neutral"]})
    return labelled, summaries


def state_for(item, with_title):
    if with_title and item["title"]:
        return {"title": item["title"], "article": item["text"]}
    return {"article": item["text"]}


# ------------------------------------------------------------------ spend ledger
class Ledger:
    def __init__(self, cap_usd):
        self.path = OUT / "spend.json"
        self.cap = cap_usd / PRICE_PER_TOKEN
        d = json.loads(self.path.read_text()) if self.path.exists() else {"input_tokens": 0, "requests": 0}
        self.tokens, self.requests, self.reserved = d["input_tokens"], d["requests"], 0
        self.lock = asyncio.Lock()

    @property
    def usd(self):
        return self.tokens * PRICE_PER_TOKEN

    async def reserve(self, est):
        async with self.lock:
            if self.tokens + self.reserved + est > self.cap:
                return False
            self.reserved += est
            return True

    async def settle(self, est, actual):
        async with self.lock:
            self.reserved -= est
            self.tokens += actual
            self.requests += 1
            self.path.write_text(json.dumps({"input_tokens": self.tokens, "requests": self.requests,
                                             "usd": round(self.usd, 4)}))


def api_key():
    for line in (ROOT / ".env").read_text().splitlines():
        k, sep, v = line.partition("=")
        if sep and k.strip() == "JEV_API":
            return v.strip().strip('"').strip("'")
    raise SystemExit("JEV_API not found in .env")


# ------------------------------------------------------------------ runner
async def score_all(items, qnames, with_title, out_file, ledger, est_tokens, concurrency):
    done = set()
    if out_file.exists():
        done = {json.loads(l)["id"] for l in out_file.open()}
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(done)} already scored, {len(todo)} to go; spent so far ${ledger.usd:.4f}", flush=True)
    questions = {q: QUESTIONS[q] for q in qnames}
    sem = asyncio.Semaphore(concurrency)
    stop = asyncio.Event()
    n_ok = n_fail = 0
    t0 = time.time()

    async with AsyncTypeSafeClient(api_key=api_key(), model="jev-latest") as client:
        with out_file.open("a") as fh:
            async def one(it):
                nonlocal n_ok, n_fail
                if stop.is_set():
                    return
                async with sem:
                    if stop.is_set() or not await ledger.reserve(est_tokens):
                        stop.set(); return
                    try:
                        r = await client.system_one(state_for(it, with_title), questions)
                    except Exception as e:
                        await ledger.settle(est_tokens, 0)
                        n_fail += 1
                        if n_fail <= 5:
                            print(f"  fail {it['id']}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                        return
                    await ledger.settle(est_tokens, r.usage.input_tokens)
                    rec = {k: it[k] for k in ("id", "story_id", "split", "slot", "label", "publisher")}
                    rec["input_tokens"] = r.usage.input_tokens
                    for q in qnames:
                        a = r.answers[q]
                        rec[q] = {"p": probs(q, a), "confidence": getattr(a, "confidence", None)}
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                    n_ok += 1
                    if n_ok % 500 == 0:
                        rate = n_ok / (time.time() - t0)
                        print(f"  {n_ok}/{len(todo)}  {rate:.1f}/s  spent ${ledger.usd:.4f}", flush=True)

            await asyncio.gather(*(one(it) for it in todo))

    print(f"scored {n_ok}, failed {n_fail}, stopped_on_budget={stop.is_set()}; "
          f"total spent ${ledger.usd:.4f} over {ledger.requests} requests", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dev", "probe", "run"])
    ap.add_argument("--per-class", type=int, default=150, help="dev: articles per class")
    ap.add_argument("--method", default="choice_rich", help="run/probe: question name (comma-separated ok)")
    ap.add_argument("--title", action="store_true", help="include the article title in state")
    ap.add_argument("--cap-usd", type=float, default=0.95)
    ap.add_argument("--est-tokens", type=int, default=900, help="per-request reservation for the cap")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--n", type=int, default=20, help="probe: number of requests")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(a.cap_usd)
    labelled, summaries = load_items()

    if a.mode == "dev":
        rng = random.Random(0)
        val = [x for x in labelled if x["split"] == "val"]
        dev = [x for c in range(3) for x in rng.sample([v for v in val if v["label"] == c], a.per_class)]
        for with_title in (False, True):
            f = OUT / f"dev_{'title' if with_title else 'body'}.jsonl"
            print(f"== dev, title={with_title}, {len(dev)} articles, all {len(QUESTIONS)} designs ==", flush=True)
            asyncio.run(score_all(dev, list(QUESTIONS), with_title, f, ledger, a.est_tokens * 2, a.concurrency))
    elif a.mode == "probe":
        items = random.Random(1).sample(labelled, a.n // 2) + random.Random(1).sample(summaries, a.n - a.n // 2)
        asyncio.run(score_all(items, a.method.split(","), a.title, OUT / "probe.jsonl", ledger,
                              a.est_tokens, a.concurrency))
    else:
        # dev articles were used to choose the design and fit the prior correction: keep them out.
        # Labelled first (the evaluation); summaries shuffled so a budget cut leaves a random sample.
        dev_ids = {json.loads(l)["id"] for l in (OUT / "dev_title.jsonl").open()}
        summaries = random.Random(0).sample(summaries, len(summaries))
        items = [x for x in labelled if x["id"] not in dev_ids] + summaries
        f = OUT / f"run_{a.method.replace(',', '+')}{'_title' if a.title else ''}.jsonl"
        asyncio.run(score_all(items, a.method.split(","), a.title, f, ledger, a.est_tokens, a.concurrency))


if __name__ == "__main__":
    main()
