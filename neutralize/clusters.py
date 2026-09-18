"""Reading input clusters and writing output records.

Named clusters.py rather than io.py so nothing in the package accidentally
shadows the stdlib module.
"""

import json
import os
import sys


class ClusterError(ValueError):
    pass


def normalise(raw, cfg, lineno):
    """Map a raw input line onto the internal shape, using the configured keys.

    Internal shape is fixed: {"id", "articles": [{"stance","publisher","title","body"}]}.
    Everything downstream reads that, so a different input layout is a config
    change and never a code change.
    """
    if cfg.id_field not in raw:
        raise ClusterError(f"line {lineno}: no {cfg.id_field!r} field")
    arts_raw = raw.get(cfg.articles_field)
    if not isinstance(arts_raw, list) or not arts_raw:
        raise ClusterError(
            f"line {lineno}: {cfg.articles_field!r} missing or not a non-empty list")

    articles = []
    for a in arts_raw:
        body = (a.get(cfg.body_field) or "").strip()
        if not body:
            continue
        articles.append({
            "stance": (a.get(cfg.stance_field) or "").strip().lower(),
            "publisher": (a.get(cfg.publisher_field) or "").strip(),
            "title": (a.get(cfg.title_field) or "").strip(),
            "body": body,
        })
    if not articles:
        raise ClusterError(f"line {lineno}: no article had a non-empty "
                           f"{cfg.body_field!r}")

    return {"id": str(raw[cfg.id_field]), "articles": articles, "raw": raw}


def read_clusters(path, cfg):
    """Yield normalised clusters. A bad line warns and is skipped, so one
    malformed record does not abort a thousand-cluster run."""
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  skipping line {lineno}: bad JSON ({e})", file=sys.stderr)
                continue
            try:
                yield normalise(raw, cfg, lineno)
            except ClusterError as e:
                print(f"  skipping {e}", file=sys.stderr)


def read_records(path):
    """Read an output JSONL back. Tolerates a torn final line from a killed run."""
    records = []
    if not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines):
                print(f"  ignoring torn final line in {path} "
                      f"(a previous run was killed mid-write)", file=sys.stderr)
            else:
                print(f"  ignoring unparseable line {i} in {path}", file=sys.stderr)
    return records


def done_ids(path):
    return {str(r.get("story_id")) for r in read_records(path)
            if r.get("story_id") is not None and r.get("summary")}


class Appender:
    """Append-only JSONL with a flush and fsync per record.

    A killed run therefore loses at most the record in flight, and read_records
    tolerates that line being torn.
    """

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._f = open(path, "a", encoding="utf-8")

    def write(self, record):
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
