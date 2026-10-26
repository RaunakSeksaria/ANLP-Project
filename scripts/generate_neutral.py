"""Generate a neutral summary per cluster with Gemini.

    uv run python scripts/generate_neutral.py --in clusters.jsonl \
        --out runs/gemini/summaries.jsonl --model <model-id>

Input is one JSON object per line: a story id and a list of articles covering the
same event, one per outlet. Output is one JSON object per line, appended as each
cluster finishes, so a killed run resumes where it stopped.

There is no response cache. A rerun over an existing output file skips the ids
already in it and costs nothing; to sample the model again, write to a different
--out and compare the two with scripts/check_consistency.py.

Every sampling parameter is recorded in each record. --temperature is a flag, not
a constant buried in the client.
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neutralize import evaluate as ev                                  # noqa: E402
from neutralize.client import (                                        # noqa: E402
    FatalAPIError, GeminiClient, RateGate, TransientExhausted,
)
from neutralize.clusters import Appender, done_ids, read_clusters      # noqa: E402
from neutralize.config import GenConfig                                # noqa: E402
from neutralize.plugins import load_callable                           # noqa: E402
from neutralize.prompt import (                                        # noqa: E402
    PROMPT_VERSION, build_prompt, build_revision_prompt, malformed_note,
    prompt_version,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = GenConfig()
    p.add_argument("--in", dest="inp", required=True, help="input clusters JSONL")
    p.add_argument("--out", required=True, help="output JSONL (appended to)")
    p.add_argument("--provider", choices=("gemini", "nvidia"), default="gemini",
                   help="nvidia = NVIDIA's OpenAI-compatible endpoint "
                        "(Nemotron, Kimi); reads NVIDIA_API_KEY or --api-key-env")
    p.add_argument("--api-key-env", default=None,
                   help="environment variable holding the key "
                        "(default GEMINI_API_KEY / NVIDIA_API_KEY)")
    p.add_argument("--stop-after", type=int, default=d.stop_after_consecutive_failures,
                   help="abort after this many consecutive failures; 0 disables")
    p.add_argument("--model", default=os.environ.get("GEMINI_MODEL") or d.model)
    p.add_argument("--temperature", type=float, default=d.temperature)
    p.add_argument("--top-p", type=float, default=d.top_p)
    p.add_argument("--top-k", type=int, default=None,
                   help="unset by default; for --provider nvidia it is only sent when given")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--max-output-tokens", type=int, default=d.max_output_tokens)
    p.add_argument("--thinking-budget", type=int, default=d.thinking_budget,
                   help="0 disables thinking (default). Gemini 2.x only; 3.x ignores it")
    p.add_argument("--thinking-level", default=d.thinking_level,
                   choices=("", "MINIMAL", "LOW", "MEDIUM", "HIGH"),
                   help="Gemini 3.x thinking control; overrides --thinking-budget")
    p.add_argument("--max-revisions", type=int, default=d.max_revisions,
                   help="eval-metric feedback loop; 0 turns it off")
    p.add_argument("--max-network-retries", type=int, default=d.max_network_retries)
    p.add_argument("--max-malformed-retries", type=int, default=d.max_malformed_retries)
    p.add_argument("--workers", type=int, default=d.workers)
    p.add_argument("--rpm", type=int, default=d.rpm)
    p.add_argument("--task", choices=("fuse", "neutralise"), default=d.task,
                   help="fuse = many articles -> one neutral article; "
                        "neutralise = ONE article -> same article, framing removed")
    p.add_argument("--conflict-style", choices=("attribute", "range"),
                   default=d.conflict_style,
                   help="attribute = state both versions (v3); "
                        "range = reconcile numbers into a range (v4)")
    p.add_argument("--reveal-stance", action="store_true",
                   help="show the model each article's leaning (ablation)")
    p.add_argument("--reveal-publisher", action="store_true")
    p.add_argument("--evaluator", default=None,
                   help="module:function returning {metric: number}; "
                        "defaults to neutralize.evaluate:evaluate")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ids", default=None, help="comma-separated story ids")
    p.add_argument("--overwrite", action="store_true",
                   help="regenerate ids already present in --out")
    p.add_argument("--dry-run", action="store_true",
                   help="print the rendered prompt for the first cluster and exit")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override any other GenConfig field, e.g. "
                        "--set max_stance_parity_gap=0.05")
    return p.parse_args(argv)


def build_config(args):
    cfg = GenConfig(
        model=args.model, temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k if args.top_k is not None or args.provider == "nvidia" else GenConfig.top_k,
        seed=args.seed,
        max_output_tokens=args.max_output_tokens,
        thinking_budget=args.thinking_budget,
        thinking_level=args.thinking_level,
        max_revisions=args.max_revisions,
        max_network_retries=args.max_network_retries,
        max_malformed_retries=args.max_malformed_retries,
        workers=args.workers, rpm=args.rpm,
        stop_after_consecutive_failures=args.stop_after,
        reveal_stance=args.reveal_stance, reveal_publisher=args.reveal_publisher,
        conflict_style=args.conflict_style,
        task=args.task,
    )
    for item in args.set:
        if "=" not in item:
            sys.exit(f"--set expects KEY=VALUE, got {item!r}")
        key, val = item.split("=", 1)
        if not hasattr(cfg, key):
            sys.exit(f"--set: no config field {key!r}. "
                     f"Fields: {', '.join(cfg.field_names())}")
        cur = getattr(cfg, key)
        try:
            cast = (json.loads(val) if isinstance(cur, (list, tuple, dict))
                    else type(cur)(val) if not isinstance(cur, bool)
                    else val.lower() in ("1", "true", "yes"))
        except (TypeError, ValueError):
            sys.exit(f"--set {key}: cannot read {val!r} as {type(cur).__name__}")
        setattr(cfg, key, cast)
    return cfg


def api_key(var="GEMINI_API_KEY"):
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    key = os.environ.get(var)
    if not key:
        sys.exit(f"{var} is not set. Put it in .env at the repo root "
                 f"({var}=...) or export it.")
    return key


def check_malformed(reply, target, cfg):
    """Reasons to ask for the whole thing again, rather than to revise it."""
    if reply.blocked:
        return f"blocked:{reply.blocked}"
    if not reply.text:
        return "the response was empty."
    finish = (reply.finish_reason or "").upper()
    if finish and "STOP" not in finish:
        if "MAX_TOKEN" in finish:
            return "the response was cut off before it finished."
        return f"the response ended with {finish}."
    words = len(reply.text.split())
    if target:
        if words < target * cfg.hard_min_ratio:
            return f"the summary was {words} words, far short of the ~{target} asked for."
        if words > target * cfg.hard_max_ratio:
            return f"the summary was {words} words, far beyond the ~{target} asked for."
    return None


class Runner:
    def __init__(self, cfg, client, evaluator, out, verbose=True):
        self.cfg, self.client, self.evaluator = cfg, client, evaluator
        self.out = out
        self.verbose = verbose
        self.lock = threading.Lock()
        self.abort = threading.Event()
        self.fatal = None
        self.n_done = self.n_failed = 0
        self.consecutive_failures = 0
        self.stopped_early = None
        self.tokens = 0

    def process(self, cluster):
        prompt, target = build_prompt(cluster, self.cfg)
        before = self.client.calls
        drafts, history, notes = [], [], []
        attempts = {"malformed": 0, "revision": 0}
        summary, conflicts, reply = "", [], None

        for revision in range(self.cfg.max_revisions + 1):
            reply, bad = self._generate_valid(prompt, target, attempts, cluster["id"])
            if bad:
                return self._record(cluster, target, error=bad, attempts=attempts,
                                    api_calls=self.client.calls - before)
            summary, conflicts = reply.text, reply.conflicts
            metrics = self.evaluator(summary, cluster, self.cfg)
            notes = ev.failures(metrics, self.cfg, cluster, summary)
            history.append(metrics)
            if not notes or revision == self.cfg.max_revisions:
                break
            drafts.append(summary)
            attempts["revision"] += 1
            self._log(f"[{cluster['id']}] revision {attempts['revision']}: "
                      f"{len(notes)} issue(s)")
            prompt, target = build_revision_prompt(cluster, self.cfg, summary, notes)

        return self._record(
            cluster, target, summary=summary, conflicts=conflicts, reply=reply,
            metrics=history[-1] if history else {}, history=history,
            passed=not notes, notes=notes, drafts=drafts, attempts=attempts,
            api_calls=self.client.calls - before)

    def _generate_valid(self, prompt, target, attempts, sid):
        """The malformed-output loop. Returns (reply, error_or_None)."""
        note = ""
        budget = self.cfg.max_malformed_retries
        for attempt in range(budget + 1):      # one try, then `budget` retries
            if self.abort.is_set():
                return None, "__skip__"
            try:
                reply = self.client.generate(prompt + note, label=f"[{sid}] ")
            except FatalAPIError as e:
                with self.lock:
                    self.fatal = str(e)
                self.abort.set()
                return None, f"fatal: {e}"
            except TransientExhausted as e:
                return None, f"network: {e}"

            with self.lock:
                self.tokens += reply.usage.get("total_tokens") or 0
            bad = check_malformed(reply, target, self.cfg)
            if not bad:
                return reply, None
            if bad.startswith("blocked:"):
                # Safety blocks do not improve on retry; record and move on.
                return None, bad
            if attempt == budget:
                return None, f"malformed after {budget} retries: {bad}"
            attempts["malformed"] += 1
            self._log(f"[{sid}] malformed, retry {attempts['malformed']}/{budget}: {bad}")
            note = malformed_note(bad)
        return None, "malformed"

    def _record(self, cluster, target, summary="", conflicts=None, reply=None,
                metrics=None, history=None, passed=None, notes=None, drafts=None,
                attempts=None, error=None, api_calls=0):
        rec = {
            "story_id": cluster["id"],
            "summary": summary,
            "conflicts": conflicts or [],
            "target_words": target,
            "actual_words": len(summary.split()),
            "n_sources": len(cluster["articles"]),
            "model": self.cfg.model,
            "params": {**self.cfg.params(), "prompt_version": prompt_version(self.cfg)},
            # what was actually sent, where the client can say (params is the config)
            "sent": (self.client.decoding_params() if hasattr(self.client, "decoding_params")
                     else None),
            "attempts": {**(attempts or {}), "api_calls": api_calls},
            "metrics": metrics or {},
            "metrics_history": history or [],
            "passed": passed,
            "unresolved": notes or [],
            "drafts": drafts or [],
            "finish_reason": getattr(reply, "finish_reason", None),
            "model_version": getattr(reply, "model_version", None),
            "response_id": getattr(reply, "response_id", None),
            "usage": getattr(reply, "usage", {}) or {},
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if error:
            rec["error"] = error
        return rec

    def _log(self, msg):
        if self.verbose:
            print(f"  {msg}", file=sys.stderr, flush=True)


def main(argv=None):
    args = parse_args(argv)
    cfg = build_config(args)

    clusters = list(read_clusters(args.inp, cfg))
    if not clusters:
        sys.exit(f"no usable clusters in {args.inp}")

    if args.ids:
        want = {s.strip() for s in args.ids.split(",") if s.strip()}
        clusters = [c for c in clusters if c["id"] in want]
    if not args.overwrite:
        already = done_ids(args.out)
        skipped = sum(1 for c in clusters if c["id"] in already)
        clusters = [c for c in clusters if c["id"] not in already]
        if skipped:
            print(f"skipping {skipped} cluster(s) already in {args.out}", file=sys.stderr)
    if args.limit:
        clusters = clusters[:args.limit]

    if args.dry_run:
        prompt, target = build_prompt(clusters[0], cfg)
        print(f"--- cluster {clusters[0]['id']}  "
              f"({len(clusters[0]['articles'])} sources, target {target} words, "
              f"prompt {prompt_version(cfg)}) ---\n")
        print(prompt)
        print(f"\n--- {len(clusters)} cluster(s) would be generated ---")
        return 0

    if not clusters:
        print("nothing to do: every cluster is already in the output file.",
              file=sys.stderr)
        return 0

    evaluator = load_callable(args.evaluator) if args.evaluator else (
        lambda s, c, cf: ev.evaluate(s, c, cf))
    if cfg.max_revisions and not args.evaluator:
        ev.load_resources()          # fail early if the lexicons are missing

    gate = RateGate(cfg.rpm)
    if args.provider == "nvidia":
        from neutralize.nvidia_client import NvidiaClient
        client = NvidiaClient(cfg, api_key(args.api_key_env or "NVIDIA_API_KEY"), gate)
    else:
        client = GeminiClient(cfg, api_key(args.api_key_env or "GEMINI_API_KEY"), gate)
    runner = Runner(cfg, client, evaluator, args.out)

    print(f"{len(clusters)} cluster(s) -> {args.out}\n"
          f"provider={args.provider} model={cfg.model} temperature={cfg.temperature} top_p={cfg.top_p} "
          f"top_k={cfg.top_k} seed={cfg.seed} thinking={cfg.thinking_budget}\n"
          f"workers={cfg.workers} rpm={cfg.rpm} max_revisions={cfg.max_revisions} "
          f"prompt={prompt_version(cfg)}", file=sys.stderr)

    started = time.time()
    with Appender(args.out) as out, ThreadPoolExecutor(cfg.workers) as pool:
        for rec in pool.map(runner.process, clusters):
            with runner.lock:
                # Once abort trips, ThreadPoolExecutor.map still drains every
                # remaining cluster. Writing those as failures buried the real
                # errors under 188k rows and made the file un-resumable.
                if rec.get("error") == "__skip__":
                    continue
                out.write(rec)
                if rec.get("error"):
                    runner.n_failed += 1
                    runner.consecutive_failures += 1
                    limit = cfg.stop_after_consecutive_failures
                    if limit and runner.consecutive_failures >= limit:
                        runner.stopped_early = (
                            f"{runner.consecutive_failures} consecutive failures; "
                            f"last: {rec['error'][:120]}")
                        runner.abort.set()
                else:
                    runner.n_done += 1
                    runner.consecutive_failures = 0
                n = runner.n_done + runner.n_failed
                print(f"  [{n}/{len(clusters)}] {rec['story_id']}: "
                      + (f"FAILED ({rec['error']})" if rec.get("error") else
                         f"{rec['actual_words']}w"
                         f"{'' if rec['passed'] is not False else ' (metrics not met)'}"),
                      file=sys.stderr, flush=True)

    elapsed = time.time() - started
    print(f"\n{runner.n_done} written, {runner.n_failed} failed, "
          f"{client.calls} API calls, ~{runner.tokens} tokens, {elapsed:.0f}s",
          file=sys.stderr)
    if runner.stopped_early:
        print(f"\nSTOPPED EARLY: {runner.stopped_early}", file=sys.stderr)
        print(f"clusters already written are safe; rerun the same command to "
              f"resume once the endpoint is healthy.", file=sys.stderr)
        return 3
    if runner.fatal:
        print(f"\nrun aborted: {runner.fatal}", file=sys.stderr)
        return 2
    return 1 if runner.n_done == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
