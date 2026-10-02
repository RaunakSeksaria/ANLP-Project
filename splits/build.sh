#!/bin/bash
# Rebuild every prepared split under dataset/ from the Hugging Face datasets,
# then check them against splits/manifest.json. Run from anywhere; ~5 min on CPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
uv run python splits/download_hf.py                                   # -> dataset/raw/
uv run python training/src/scripts/make_pairable_subset.py            # -> dataset/pairable/
uv run python training/src/scripts/prepare_side_neutral.py            # -> dataset/side_neutral_full/ (E2)
uv run python training/src/scripts/make_negative_variants.py \
    --policy publisher --output-dir dataset/side_neutral_full_publisher
uv run python training/src/scripts/prepare_data.py                    # -> dataset/neutral_articles_10k_prepared/ (E1)
uv run python splits/check.py
