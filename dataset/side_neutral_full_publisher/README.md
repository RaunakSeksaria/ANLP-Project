# side_neutral_full_publisher

`side_neutral_full` with **publisher negatives** (proposal variant a): every
triplet's negative is an article from the anchor's own publisher, about a
different event group, in the same split. Everything else is byte-identical to
`dataset/side_neutral_full`: the baseline files, anchors, positives
(opposite-side article about the same event) and targets. A run on this split
differs from a run on `side_neutral_full` in its negatives alone.

Built with `src/scripts/make_negative_variants.py --policy publisher` (seed
`42:publisher`, on the `manas` branch) from `side_neutral_full` and the
publisher labels in `left/right_neutral_dedup.jsonl`. Checks run at build time:
Manas's `validate_prepared` passes, 100% of negatives share the anchor's
publisher, no negative is in the anchor's event group, and the trainer's loader
(`LEDTripletDataset`) accepts every negative as given (0 replacements).
`prepared/manifest.json` records the policy, the source triplet hashes and
per-publisher anchor counts.

Models trained on it: HF `RaunakSeksaria/anlp-neutral-led`,
`side-neutral-full-publisher/<run>`.

Use with `--data-dir dataset/side_neutral_full_publisher/prepared`.
