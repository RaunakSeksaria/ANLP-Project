# 2,000 paired article embeddings across three LED checkpoints

A seeded sample of 1,000 left and 1,000 right original–neutral pairs is stored in source order. Row `i` of all six matrices is row `i` of `metadata.parquet`. Both texts were encoded directly through each checkpoint's LED encoder with the training pooling helper. Base means pretrained LED before ANLP task training. The dataset can be regenerated from the root JSONL files without the earlier full embedding dataset. No distance metric was computed.

Reproduce from the repository root with `sbatch scripts/run_three_model_2k.sbatch` on the GPU cluster, or run `python scripts/prepare_three_model_2k.py --device cuda --batch-size 8` on a GPU node.
