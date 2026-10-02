"""Resolve a checkpoint argument to a local directory, downloading from Hugging Face if needed."""

from __future__ import annotations

from pathlib import Path

HF_REPO = "RaunakSeksaria/anlp-neutral-led"


def resolve_checkpoint(spec: str | Path, repo: str = HF_REPO) -> Path:
    """Return `spec` if it is a local directory, else download `<experiment>/<run>` from `repo`.

    Example: ``resolve_checkpoint("side-neutral-full/lambda035")``.
    """
    local = Path(spec)
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    folder = str(spec).strip("/")
    root = Path(snapshot_download(repo, allow_patterns=[f"{folder}/*"]))
    path = root / folder
    if not (path / "model.safetensors").is_file():
        raise FileNotFoundError(f"{folder} is neither a local directory nor a run in {repo}")
    return path
