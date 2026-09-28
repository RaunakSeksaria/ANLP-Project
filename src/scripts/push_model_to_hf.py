#!/usr/bin/env python3
"""Upload a saved LED checkpoint using HF_TOKEN (a token with write access).

Example:
    python src/scripts/push_model_to_hf.py \
        --checkpoint outputs/joint_mean_lambda035/best \
        --repo-id YOUR_USERNAME/led-neutral-lambda035 --dry-run

Remove --dry-run to upload. New repositories are private unless --public is set.
Existing repository visibility is unchanged. Model files at matching remote
paths are updated; unrelated remote files are not deleted.
API: https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from huggingface_hub import HfApi
from huggingface_hub.utils import validate_repo_id


# Local credential supplied by the user. Do not commit or share this file.
# Set HF_TOKEN to override this token without editing the script.
DEFAULT_HF_TOKEN = "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"


# Explicit inference-only file list: exclude optimizer state, datasets, logs,
# and local training metadata. These checkpoints have unsharded safetensors.
MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)
REQUIRED_FILES = (
    "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Directory containing model.safetensors, e.g. outputs/.../best")
    parser.add_argument("--repo-id", required=True, help="HF username/model-name or org/model-name")
    parser.add_argument("--public", action="store_true", help="Create a public repository (default: private)")
    parser.add_argument("--dry-run", action="store_true", help="Preview files without network access or a token")
    args = parser.parse_args(argv)
    validate_repo_id(args.repo_id)
    if args.repo_id.count("/") != 1:
        parser.error("--repo-id must include a namespace: username/model-name")
    checkpoint = args.checkpoint.resolve()
    missing = [name for name in REQUIRED_FILES if not (checkpoint / name).is_file()]
    if missing:
        parser.error(f"Incomplete checkpoint {checkpoint}: missing {', '.join(missing)}")
    files = [name for name in MODEL_FILES if (checkpoint / name).is_file()]
    print(f"Checkpoint: {checkpoint}")
    print(f"Destination: {args.repo_id}")
    print(f"New repository visibility: {'public' if args.public else 'private'}")
    for name in files:
        print(f"  {name}: {(checkpoint / name).stat().st_size:,} bytes")
    if args.dry_run:
        print("Dry run complete; nothing uploaded.")
        return 0

    token = os.environ.get("HF_TOKEN", DEFAULT_HF_TOKEN).strip()
    if not token:
        parser.error("Set HF_TOKEN to a Hugging Face token with write access before uploading")
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id, repo_type="model", private=not args.public, exist_ok=True
    )
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(checkpoint),
        allow_patterns=files,
        commit_message=f"Upload LED checkpoint: {checkpoint.parent.name}/{checkpoint.name}",
    )
    print(f"Uploaded: https://huggingface.co/{args.repo_id}")
    print(f"Commit: {commit.oid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
