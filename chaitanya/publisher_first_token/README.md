# Publisher-anchored first-token runs

Same models as Manas's ideology-anchored first-token runs (branch `manas` @ `03e2b1b`), with **publisher**
negatives: the triplet negative is an article from the anchor's own outlet about a different event (instead of a
same-side article from any outlet). Data, code, hyperparameters and seed are otherwise identical to his runs.

Weights: [tchaitanya0109/anlp-neutral-led-publisher-first-token](https://huggingface.co/tchaitanya0109/anlp-neutral-led-publisher-first-token)
(folders `standard-lambda035/`, `attraction-lambda035/`; the model card also has the embedding evaluation).

## Data

`dataset/raw/{left,right}_neutral_dedup.jsonl` = the `reframing_left` / `reframing_right` configs of
[`RaunakSeksaria/anlp-project-datasets`](https://huggingface.co/datasets/RaunakSeksaria/anlp-project-datasets)
(train + validation + test), as written by `splits/download_hf.py` on the `mid-eval` branch, at dataset revision
`240b93952356e66754f76a9c1d3ff47bb852fef4` (the revision that reproduces Manas's runs). Then:

```bash
python src/scripts/prepare_side_neutral.py \
    --left dataset/raw/left_neutral_dedup.jsonl --right dataset/raw/right_neutral_dedup.jsonl \
    --output-dir dataset/side_neutral --seed 42
python src/scripts/make_negative_variants.py \
    --base dataset/side_neutral/prepared --output-dir dataset/side_neutral_publisher --policy publisher \
    --left dataset/raw/left_neutral_dedup.jsonl --right dataset/raw/right_neutral_dedup.jsonl --seed 42
python chaitanya/publisher_first_token/check_publisher_dataset.py --raw dataset/raw \
    --ideology dataset/side_neutral/prepared --publisher dataset/side_neutral_publisher/prepared \
    --report dataset/side_neutral_publisher/check_report.json
```

`check_publisher_dataset.py` checks that the rebuilt `side_neutral` set matches the six file hashes logged in Manas's
W&B runs, and that the publisher set differs from it in the negatives only (16,447 / 1,778 / 1,775 rows; 100 % of
negatives from the anchor's publisher and from a different event group; anchors, positives, targets and baseline
files unchanged).

## Training

Manas's launchers `src/scripts/train_joint_first_token_attraction_2gpu.sh` / `train_joint_first_token_2gpu.sh` with
only the data directory, run name and W&B team changed (2× RTX 2080 Ti on Ada; Python 3.12, torch 2.7.0+cu118,
transformers 5.16.1, wandb 0.29.0):

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 src/scripts/train_joint.py \
    --data-dir dataset/side_neutral_publisher/prepared --output-dir outputs/<run name> \
    --model-name allenai/led-base-16384 --train-batch-size 1 --eval-batch-size 1 \
    --gradient-accumulation-steps 4 --max-epochs 10 --learning-rate 3e-5 --weight-decay 0.01 --warmup-ratio 0.10 \
    --max-grad-norm 1.0 --early-stopping-patience 3 --early-stopping-min-delta 0.0 \
    --max-input-length 512 --max-target-length 512 --generation-max-new-tokens 512 \
    --triplet-lambda 0.35 --triplet-margin 0.2 --attention-window 512 \
    --triplet-objective attraction --positive-distance-threshold 0.5 --positive-beta 0.3 \
    --pooling first_token --seed 42 --num-workers 0 --device cuda --precision fp32 \
    --wandb-entity tchaitanya0109-iiit-hyderabad --wandb-project anlp-project \
    --wandb-run-name side-neutral-publisher-first-token-attraction-lambda035 --require-wandb
```

The standard-triplet run uses `--triplet-objective standard --positive-beta 0.0` (no `--positive-distance-threshold`)
and the run name `side-neutral-publisher-first-token-lambda035`.

## Results

| Run | Triplet objective | W&B | Best epoch (of 6) | Best val generation loss | Val token accuracy | Val token-overlap F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `standard-lambda035` | standard triplet, margin 0.2 | [xsdb9688](https://wandb.ai/tchaitanya0109-iiit-hyderabad/anlp-project/runs/xsdb9688) | 3 | 0.4152 | 0.8929 | 0.8145 |
| `attraction-lambda035` | triplet + attraction (τ 0.5, β 0.3), margin 0.2 | [6179sx9m](https://wandb.ai/tchaitanya0109-iiit-hyderabad/anlp-project/runs/6179sx9m) | 3 | 0.4154 | 0.8930 | 0.8141 |

Full validation split (1,778 articles), greedy decoding, best checkpoint. Manas's matching ideology-anchored runs:
0.4150 / 0.4151 validation loss, 0.8144 / 0.8147 token-overlap F1. Both runs stopped early after epoch 6.
