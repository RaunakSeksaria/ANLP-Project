# splits/

The exact train / validation / test splits every model was trained and tested
on, without the data itself.

| file | contents |
| --- | --- |
| `manifest.json` | For each prepared set: its path under `dataset/`, row counts, and the SHA-256 of every `baseline.jsonl` / `triplets.jsonl` |
| `side_neutral_full/{train,validation,test}_example_ids.txt` | E2 example ids per split (the publisher variant shares them) |
| `neutral_articles_10k_prepared/{train,validation,test}_example_ids.txt` | E1 example ids per split |
| `build.sh` | Rebuilds `dataset/` from Hugging Face and checks it against `manifest.json` |
| `download_hf.py`, `check.py` | The download and check steps on their own |

```bash
export HF_TOKEN=...      # while the dataset repo is private
splits/build.sh          # ends with "all splits match"
```

The rebuilt files are byte-identical to the ones used for training: the E2 sets
match the copies on the `data` branch, and the E1 set matches the hashes logged
with the E1 runs. The splits come from the `split` field of the source datasets,
which assigns whole connected components of events to one split, so no article
body appears in two splits.
