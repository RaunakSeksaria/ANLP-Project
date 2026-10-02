"""Build the out.csv (content, bias) the ChunkyBERT notebook expects.

The paper states the dataset (37,554 articles; columns link/title/content/
bias/source/author; ~25.7% authors missing — see Table 2) is the one released
at https://github.com/ramybaly/Article-Bias-Prediction. This script reads that
repo's data/jsons/*.json files (one article per file) and flattens them into a
single CSV, matching the notebook's input format exactly.
"""
import argparse
import json
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Build out.csv from Article-Bias-Prediction jsons")
    parser.add_argument('--jsons-dir', required=True,
                         help='Path to Article-Bias-Prediction/data/jsons')
    parser.add_argument('--output', required=True, help='Output CSV path')
    args = parser.parse_args()

    rows = []
    filenames = sorted(f for f in os.listdir(args.jsons_dir) if f.endswith('.json'))
    for fname in filenames:
        with open(os.path.join(args.jsons_dir, fname), encoding='utf-8') as f:
            rec = json.load(f)
        rows.append({
            'link': rec.get('url'),
            'title': rec.get('title'),
            'content': rec.get('content'),
            'bias': rec.get('bias'),
            'source': rec.get('source'),
            'author': rec.get('authors'),
        })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=['content', 'bias'])
    df['bias'] = df['bias'].astype(int)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Wrote {len(df)} rows to {args.output}")
    print("Class distribution:")
    print(df['bias'].value_counts().sort_index())


if __name__ == "__main__":
    main()
