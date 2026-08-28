"""Build every input the Ada jobs need, from the NeuS crawl. CPU, seconds.

Emits into data/prepared/:

  articles.jsonl        9,285 articles: story_id, stance, publisher, title, body,
                        roundup, and which released split the story belongs to
  probe_random.json     article ids split 80/10/10 at random
  probe_media.json      article ids split so NO publisher appears in more than
                        one side -- the unseen-outlet test the released split
                        cannot give us
  triplets.jsonl        (anchor, positive, negative) article-id triples:
                        positive = same story, different publisher
                        negative = same publisher, different story
  gen/{train,val,test}.{source,target}
                        NeuS's own format, regenerated with a fixed seed so the
                        L/C/R order is reproducible (the original shuffle was
                        unseeded)

    uv run python scripts/prepare_data.py
"""

import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEUS = ROOT / "data/framing-bias-metric-main/data"
OUT = ROOT / "data/prepared"
SEED = 42

CRAWLS = (
    "raw_crawled/all_crawled_combined_filtered_RoundupLeftCenterRight.jsonl",
    "raw_crawled/all_crawled_combined.jsonl",
    "raw_crawled/headline_11_2021-10-19_crawled_all.jsonl",
)
PROBE_SPLIT = "naacl2022_lrc_roundup_random_order_probe"
PROBE_ARTICLE = re.compile(r"TITLE=>(.*)\.\s*ARTICLE=>(.*)$", re.S)

# AllSides' own staff blog appears in the `newsSource` field as an author name
# rather than an outlet. Left in the data it becomes a degenerate probe class.
STAFF_BLOG = re.compile(r"^[A-Z][a-z]+ [A-Z]\.? ?[A-Z][a-z]+$")


def norm(s):
    return re.sub(r"\W+", "", unicodedata.normalize("NFKD", s or "").lower())


def sentences(seq):
    return " ".join(s for _, s in seq)


def remove_dotdotdot(text):
    """NeuS's own cleanup (data_preprocessing.ipynb): AllSides excerpts are
    truncated with an ellipsis."""
    return text.replace("...", " ").replace("…", " ").strip()


def load_split_membership():
    """story title -> released split, via the union of all three crawl files.
    The filtered file alone covers only ~87%; the headline crawl has the rest."""
    title_to_story = {}
    for fn in CRAWLS:
        for r in (json.loads(l) for l in (NEUS / fn).open()):
            for stance, v in (r.get("news") or {}).items():
                if isinstance(v, dict) and v.get("newsTitle"):
                    title_to_story.setdefault(norm(v["newsTitle"]), r["id"])

    membership = {}
    for split in ("train", "val", "test"):
        for line in (NEUS / PROBE_SPLIT / f"{split}.source").read_text().splitlines():
            for part in line.split("[SEP]"):
                m = PROBE_ARTICLE.search(part)
                if not m:
                    continue
                sid = title_to_story.get(norm(m.group(1)))
                if sid:
                    membership[sid] = split
    return membership


def build_articles():
    rows = [json.loads(l) for l in (NEUS / CRAWLS[0]).open()]
    membership = load_split_membership()

    articles, skipped = [], Counter()
    for r in rows:
        roundup = remove_dotdotdot(sentences(r["roundup"]))
        if not roundup:
            skipped["empty roundup"] += 1
            continue
        split = membership.get(r["id"])
        if split is None:
            skipped["story not in a released split"] += 1
        for stance, v in r["news"].items():
            body = remove_dotdotdot(sentences(v["newBody"]))
            publisher = (v.get("newsSource") or "").strip()
            if not body or not publisher:
                skipped["missing body or publisher"] += 1
                continue
            articles.append({
                "article_id": f"{r['id']}:{stance}",
                "story_id": r["id"],
                "split": split,
                "stance": stance,
                "stance_specific": v.get("stance_specific"),
                "publisher": publisher,
                "is_staff_blog": bool(STAFF_BLOG.match(publisher)),
                "title": v.get("newsTitle", "").strip(),
                "body": body,
                "roundup": roundup,
                "date": r["date"][:10],
                "topic": r.get("topic"),
            })
    return articles, skipped


# --------------------------------------------------------------------------- #
# probe splits
# --------------------------------------------------------------------------- #

def probe_random(articles, rng):
    ids = [a["article_id"] for a in articles]
    rng.shuffle(ids)
    n = len(ids)
    a, b = int(0.8 * n), int(0.9 * n)
    return {"train": ids[:a], "val": ids[a:b], "test": ids[b:]}


def probe_media(articles, rng, min_articles=8):
    """Publisher-disjoint, stratified by stance.

    Whole publishers go to one side, so no outlet is ever seen on two sides.
    Stratification matters more here than it looks: in this corpus stance is a
    property of the outlet, not the article, so bucketing publishers by size
    alone lands 85% of one stance in test and the accuracies become unreadable.
    Each publisher is assigned its dominant stance and round-robined within that
    stance group.

    Only publishers with enough articles are eligible -- a two-example class
    teaches nothing and inflates the label space."""
    by_pub = defaultdict(list)
    pub_stance = defaultdict(Counter)
    for a in articles:
        if not a["is_staff_blog"]:
            by_pub[a["publisher"]].append(a["article_id"])
            pub_stance[a["publisher"]][a["stance"]] += 1

    eligible = {p: ids for p, ids in by_pub.items() if len(ids) >= min_articles}

    by_stance = defaultdict(list)
    for p in eligible:
        by_stance[pub_stance[p].most_common(1)[0][0]].append(p)

    buckets = {"train": [], "val": [], "test": []}
    order = ["train"] * 8 + ["val"] + ["test"]
    for s_idx, stance in enumerate(sorted(by_stance)):
        group = sorted(by_stance[stance], key=lambda p: (-len(eligible[p]), p))
        # offset each stance group so the largest outlets don't all land in train
        for i, p in enumerate(group):
            buckets[order[(i + s_idx * 3) % len(order)]].append(p)

    split = {k: [i for p in v for i in eligible[p]] for k, v in buckets.items()}
    return split, {k: sorted(v) for k, v in buckets.items()}, eligible


# --------------------------------------------------------------------------- #
# triplets
# --------------------------------------------------------------------------- #

def build_triplets(articles, rng, per_anchor=2):
    """positive: same story, different publisher.
    negative: same publisher, different story.

    The shared publisher between anchor and negative is the whole point -- it
    is the only way to satisfy the margin without abandoning publisher style as
    the organising signal."""
    by_story = defaultdict(list)
    by_pub = defaultdict(list)
    for a in articles:
        by_story[a["story_id"]].append(a)
        by_pub[a["publisher"]].append(a)

    triplets = []
    for anchor in articles:
        positives = [x for x in by_story[anchor["story_id"]]
                     if x["article_id"] != anchor["article_id"]
                     and x["publisher"] != anchor["publisher"]]
        negatives = [x for x in by_pub[anchor["publisher"]]
                     if x["story_id"] != anchor["story_id"]]
        if not positives or not negatives:
            continue
        for _ in range(per_anchor):
            triplets.append({
                "anchor": anchor["article_id"],
                "positive": rng.choice(positives)["article_id"],
                "negative": rng.choice(negatives)["article_id"],
                "split": anchor["split"],
            })
    return triplets


# --------------------------------------------------------------------------- #
# generation files
# --------------------------------------------------------------------------- #

def write_generation_files(articles, rng):
    """NeuS's source format, order shuffled with a fixed seed. Their own
    preprocessing shuffled unseeded, so the released files are not reproducible;
    ours are."""
    by_story = defaultdict(list)
    for a in articles:
        by_story[a["story_id"]].append(a)

    gen = OUT / "gen"
    gen.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    handles = {s: ((gen / f"{s}.source").open("w"), (gen / f"{s}.target").open("w"))
               for s in ("train", "val", "test")}
    ids = {s: [] for s in ("train", "val", "test")}

    for story_id, arts in sorted(by_story.items()):
        split = arts[0]["split"]
        if split is None or len(arts) != 3:
            counts["skipped"] += 1
            continue
        shuffled = sorted(arts, key=lambda a: a["stance"])
        rng.shuffle(shuffled)
        source = " [SEP] ".join(
            f"TITLE=> {a['title']}. ARTICLE=> {a['body']}" for a in shuffled
        ).replace("\n", "")
        target = f"TITLE=> . ARTICLE=> {arts[0]['roundup']}".replace("\n", "")
        src_f, tgt_f = handles[split]
        src_f.write(source + "\n")
        tgt_f.write(target + "\n")
        ids[split].append(story_id)
        counts[split] += 1

    for src_f, tgt_f in handles.values():
        src_f.close()
        tgt_f.close()
    (gen / "story_ids.json").write_text(json.dumps(ids, indent=2))
    return counts


def main():
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    articles, skipped = build_articles()
    print(f"articles: {len(articles)}")
    for k, v in skipped.items():
        print(f"  skipped ({k}): {v}")
    staff = sum(a["is_staff_blog"] for a in articles)
    print(f"  flagged as AllSides staff bylines rather than outlets: {staff}")

    with (OUT / "articles.jsonl").open("w") as f:
        for a in articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    rnd = probe_random(articles, rng)
    (OUT / "probe_random.json").write_text(json.dumps(rnd, indent=2))
    print(f"probe_random:  " + "  ".join(f"{k}={len(v)}" for k, v in rnd.items()))

    med, pubs, eligible = probe_media(articles, rng)
    (OUT / "probe_media.json").write_text(json.dumps(
        {"articles": med, "publishers": pubs}, indent=2))
    print(f"probe_media:   " + "  ".join(f"{k}={len(v)}" for k, v in med.items()))
    print(f"               eligible publishers (>=8 articles): {len(eligible)}"
          f"  train/val/test outlets = "
          f"{len(pubs['train'])}/{len(pubs['val'])}/{len(pubs['test'])}")
    overlap = set(pubs["train"]) & set(pubs["test"])
    assert not overlap, overlap
    print("               publisher overlap train∩test: 0  (verified)")

    trips = build_triplets(articles, rng)
    with (OUT / "triplets.jsonl").open("w") as f:
        for t in trips:
            f.write(json.dumps(t) + "\n")
    print(f"triplets: {len(trips)}  " +
          "  ".join(f"{k}={v}" for k, v in Counter(t["split"] for t in trips).items()))

    counts = write_generation_files(articles, rng)
    print("gen files: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"\nwrote {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    sys.exit(main())
