# Original-article LED embeddings

Rows are ordered as the left JSONL, then the right JSONL. `sample_id` and `original_row_index` identify each source row. Row `i` of every `*_embeddings.npy` is row `i` of `metadata.parquet`.

The `original_article` column contains only the original `body`. The encoder input follows training: stripped title, tokenizer separator, stripped original body, truncated to the configured input length. The first token has global attention. Embeddings use training's float32 attention-mask-aware mean and L2 normalization. Neutral rewrites are ignored.

```python
import numpy as np
import pyarrow.parquet as pq
metadata = pq.read_table('embedding_data/metadata.parquet').to_pandas()
z0 = np.load('embedding_data/lambda_0_embeddings.npy')
z035 = np.load('embedding_data/lambda_0_35_embeddings.npy')
```

Reproduce from the repository root with `python scripts/extract_embedding_dataset.py --revision 03598e556d4f5bd9abd2e7ddb304b80498037e24 --device cuda --batch-size 4 --cache-dir .embedding_hf_cache --output-dir embedding_data`. Use `--smoke-only` to validate ten articles first.
