#!/usr/bin/env python3
"""Download the three source datasets from Hugging Face into dataset/raw/.

Writes left_neutral_dedup.jsonl, right_neutral_dedup.jsonl and
neutral_articles_10k.jsonl: every record of the HF config, verbatim, with the
record's own `split` field. The repo is private, so export HF_TOKEN first.
"""
import argparse
import gzip
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = Path(__file__).resolve().parents[1]
HF_DATASET = "RaunakSeksaria/anlp-project-datasets"
CONFIGS = {"reframing_left": "left_neutral_dedup.jsonl",
           "reframing_right": "right_neutral_dedup.jsonl",
           "neutralised_across_articles": "neutral_articles_10k.jsonl"}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=REPO / "dataset/raw")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
for config, name in CONFIGS.items():
    with (args.out / name).open("wb") as out:
        for split in ("train", "validation", "test"):
            path = hf_hub_download(HF_DATASET, f"data/{config}/{split}.jsonl.gz", repo_type="dataset")
            with gzip.open(path, "rb") as handle:
                out.write(handle.read())
    print(f"{config} -> {args.out / name}")
