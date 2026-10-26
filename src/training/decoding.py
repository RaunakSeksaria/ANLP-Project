"""Resolve the decoding choice without changing other generation settings."""

from __future__ import annotations

from typing import Any, Mapping


def decoding_settings(base: Mapping[str, Any], strategy: str) -> dict[str, Any]:
    if strategy not in {"greedy", "beam"}:
        raise ValueError("decoding strategy must be greedy or beam")
    settings = dict(base)
    if strategy == "beam":
        settings["do_sample"] = False
        settings["num_beams"] = 4
    return settings
