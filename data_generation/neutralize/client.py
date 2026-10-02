"""Gemini wrapper: one call, plus the transient-failure retry loop.

This module owns exactly one of the three retry budgets -- the transient one.
Malformed output and metric-driven revision are the driver's business, because
both of them need to change the prompt and this does not.
"""

import json
import random
import re
import sys
import threading
import time

import httpx
from google import genai
from google.genai import errors, types

from .prompt import RESPONSE_SCHEMA, SYSTEM, response_schema, system_prompt

# 429 and 5xx are worth waiting out. 400/401/403/404 are a bad key, a bad model
# id or a malformed request -- retrying those three times only wastes time.
TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504}


class FatalAPIError(RuntimeError):
    """Not worth retrying: bad credentials, unknown model, malformed request."""


class TransientExhausted(RuntimeError):
    """Still failing after max_network_retries."""


class Reply:
    __slots__ = ("text", "conflicts", "finish_reason", "usage", "blocked",
                 "model_version", "response_id")

    def __init__(self, text, conflicts, finish_reason, usage, blocked=None,
                 model_version=None, response_id=None):
        self.text = text
        self.conflicts = conflicts
        self.finish_reason = finish_reason
        self.usage = usage
        self.blocked = blocked
        # The id we ask for is not necessarily the version that answers: an alias
        # moves under you, and a dataset built over weeks can silently span two
        # model versions with nothing in the records to show it.
        self.model_version = model_version
        self.response_id = response_id


class RateGate:
    """Token bucket, plus a run-wide pause that any 429 can trigger.

    A 429 means the *account* is over quota, so backing off only the thread that
    saw it just lets the other three walk into the same wall. The gate is shared.
    """

    def __init__(self, rpm):
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0
        self._paused_until = 0.0

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                if now < self._paused_until:
                    pause, delay = self._paused_until - now, None
                else:
                    start = max(now, self._next)
                    self._next = start + self.interval
                    pause, delay = None, start - now
            if pause is not None:
                time.sleep(pause)
                continue    # re-check: another 429 may have extended the pause
            if delay > 0:
                time.sleep(delay)
            return

    def pause(self, seconds):
        with self._lock:
            self._paused_until = max(self._paused_until,
                                     time.monotonic() + seconds)


def _retry_after(exc):
    """Honour a server-supplied delay when there is one."""
    for src in (getattr(exc, "details", None), getattr(exc, "message", "")):
        text = json.dumps(src) if isinstance(src, dict) else str(src or "")
        m = re.search(r'"?retry[-_ ]?(?:after|delay)"?[":\s]+"?(\d+(?:\.\d+)?)', text, re.I)
        if m:
            return float(m.group(1))
    return None


def _usage(resp):
    u = getattr(resp, "usage_metadata", None)
    if u is None:
        return {}
    return {
        "prompt_tokens": getattr(u, "prompt_token_count", None),
        "output_tokens": getattr(u, "candidates_token_count", None),
        "thoughts_tokens": getattr(u, "thoughts_token_count", None),
        "total_tokens": getattr(u, "total_token_count", None),
    }


def _parse(resp):
    """Pull `summary` and `conflicts` out of a structured response.

    `.parsed` is the happy path; `.text` is the fallback for when the SDK hands
    back raw JSON, and a plain-prose reply degrades to being the summary itself
    rather than being thrown away.
    """
    parsed = getattr(resp, "parsed", None)
    if isinstance(parsed, dict) and parsed.get("summary"):
        return str(parsed["summary"]).strip(), list(parsed.get("conflicts") or [])

    text = (getattr(resp, "text", None) or "").strip()
    if not text:
        return "", []
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return text, []
    if isinstance(obj, dict) and obj.get("summary"):
        return str(obj["summary"]).strip(), list(obj.get("conflicts") or [])
    return text, []


class GeminiClient:
    def __init__(self, cfg, api_key, gate=None, verbose=True):
        self.cfg = cfg
        self.gate = gate or RateGate(cfg.rpm)
        self.verbose = verbose
        self._client = genai.Client(api_key=api_key)
        # Not every model accepts a thinking budget. Discovered once, then
        # remembered, so the whole run does not pay for the same 400 repeatedly.
        self._thinking_ok = cfg.thinking_budget is not None
        self._lock = threading.Lock()
        self.calls = 0

    def _config(self, use_thinking):
        kw = dict(
            system_instruction=system_prompt(self.cfg),
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            seed=self.cfg.seed,
            max_output_tokens=self.cfg.max_output_tokens,
            response_mime_type="application/json",
            response_schema=response_schema(self.cfg),
        )
        if use_thinking:
            level = (self.cfg.thinking_level or "").strip().upper()
            kw["thinking_config"] = (
                types.ThinkingConfig(thinking_level=level) if level
                else types.ThinkingConfig(thinking_budget=self.cfg.thinking_budget))
        return types.GenerateContentConfig(**kw)

    def generate(self, prompt, label=""):
        """One generation, retrying only transient failures.

        Raises FatalAPIError for anything a retry cannot fix, and
        TransientExhausted once the budget is spent.
        """
        last = None
        for attempt in range(self.cfg.max_network_retries + 1):
            self.gate.acquire()
            use_thinking = self._thinking_ok
            try:
                with self._lock:
                    self.calls += 1
                resp = self._client.models.generate_content(
                    model=self.cfg.model,
                    contents=prompt,
                    config=self._config(use_thinking),
                )
            except errors.APIError as e:
                status = getattr(e, "code", None) or getattr(e, "status", None)
                msg = str(getattr(e, "message", "") or e)
                if use_thinking and status == 400 and "thinking" in msg.lower():
                    with self._lock:
                        self._thinking_ok = False
                    self._log(f"{label}model rejects thinking_budget; "
                              f"disabling it for the rest of the run")
                    continue
                if not isinstance(status, int) or status not in TRANSIENT_STATUS:
                    raise FatalAPIError(f"HTTP {status}: {msg}") from e
                last = e
                if status == 429:
                    self.gate.pause(_retry_after(e) or self._backoff(attempt))
            except (httpx.TransportError, TimeoutError, ConnectionError, OSError) as e:
                if _fatal_transport(e):
                    raise FatalAPIError(str(e)) from e
                last = e
            else:
                return self._reply(resp)

            if attempt == self.cfg.max_network_retries:
                break
            delay = _retry_after(last) or self._backoff(attempt)
            self._log(f"{label}transient failure "
                      f"({attempt + 1}/{self.cfg.max_network_retries}), "
                      f"retrying in {delay:.1f}s: {_short(last)}")
            time.sleep(delay)

        raise TransientExhausted(
            f"{self.cfg.max_network_retries} retries exhausted"
            + (f": {_short(last)}" if last is not None else ""))

    def _backoff(self, attempt):
        # Jittered so four workers that hit the same 429 do not resume in lockstep.
        return min(self.cfg.backoff_base ** (attempt + 1), self.cfg.backoff_cap) * (
            0.7 + 0.6 * random.random())

    def _reply(self, resp):
        text, conflicts = _parse(resp)
        cands = getattr(resp, "candidates", None) or []
        finish = str(getattr(cands[0], "finish_reason", "")) if cands else ""
        fb = getattr(resp, "prompt_feedback", None)
        blocked = str(getattr(fb, "block_reason", "") or "") or None
        return Reply(text, conflicts, finish, _usage(resp), blocked,
                     getattr(resp, "model_version", None),
                     getattr(resp, "response_id", None))

    def _log(self, msg):
        if self.verbose:
            print(f"  {msg}", file=sys.stderr, flush=True)


# A genuinely broken TLS setup will not fix itself in four seconds. A connection
# dropped mid-stream will. Both can surface as an SSL-named exception, so match on
# the message: UNEXPECTED_EOF_WHILE_READING is connection churn and must be retried.
_FATAL_TLS = ("certificate_verify_failed", "certificate verify failed",
              "self signed certificate", "hostname mismatch",
              "certificate has expired", "unable to get local issuer")


def _fatal_transport(e):
    msg = str(e).lower()
    return any(s in msg for s in _FATAL_TLS)


def _short(e, n=160):
    s = str(getattr(e, "message", "") or e).replace("\n", " ")
    return s[:n] + ("..." if len(s) > n else "")
