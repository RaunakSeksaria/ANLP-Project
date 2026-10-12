# Paired original/neutral LED embeddings

Rows follow the left JSONL, then the right JSONL. Row `i` in every matrix is row `i` of `metadata.parquet`, identified by `sample_id`. Both texts were encoded independently through each checkpoint's LED encoder. Original input uses stripped title, tokenizer separator, stripped body; neutral input uses stripped rewrite text. Pooling and normalization match the triplet training path. No distances or other evaluation were computed.

```python
import numpy as np
import pyarrow.parquet as pq
rows = pq.read_table('neutral_pair_embedding_data/metadata.parquet')
original = np.load('neutral_pair_embedding_data/lambda_0_original.npy')
neutral = np.load('neutral_pair_embedding_data/lambda_0_neutral.npy')
```

The same filenames use `lambda_0_35` for the second checkpoint. Lambda 0.1 was not extracted because its checkpoint is not currently in the cited Hugging Face repository.

Reproduce from the repository root:

```bash
python scripts/extract_neutral_pair_embeddings.py \
  --left-file left_neutral_dedup.jsonl \
  --right-file right_neutral_dedup.jsonl \
  --output-dir neutral_pair_embedding_data \
  --backup-dir /home2/manas.agrawal/anlp/embedding_data_backup \
  --revision 03598e556d4f5bd9abd2e7ddb304b80498037e24 \
  --cache-dir .embedding_hf_cache \
  --device cuda --batch-size 8
```
