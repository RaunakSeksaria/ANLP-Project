#!/usr/bin/env python3
"""Drop records whose input or target exceeds the model's token limit.

Counts tokens exactly as training encodes them: input `title </s> body`,
target the neutral rewrite, both with special tokens. Writes the kept records
verbatim under the file names make_pairable_subset.py reads.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="gets left_neutral_dedup.jsonl and right_neutral_dedup.jsonl")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--tokenizer", default="allenai/led-base-16384")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for side, path in (("left", args.left), ("right", args.right)):
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = [json.loads(l) for l in lines]
        sources = [f"{r[f'{side}_original']['title'].strip()} {tok.sep_token} "
                   f"{r[f'{side}_original']['body'].strip()}" for r in rows]
        targets = [r[f"{side}_neutralised"].strip() for r in rows]
        src_len = [len(x) for x in tok(sources)["input_ids"]]
        tgt_len = [len(x) for x in tok(targets)["input_ids"]]
        keep = [s <= args.max_tokens and t <= args.max_tokens for s, t in zip(src_len, tgt_len)]
        with (args.output_dir / f"{side}_neutral_dedup.jsonl").open("w", encoding="utf-8") as out:
            for line, k in zip(lines, keep):
                if k:
                    out.write(line + "\n")
        summary[side] = {"kept": dict(Counter(r["split"] for r, k in zip(rows, keep) if k)),
                         "dropped": keep.count(False)}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
