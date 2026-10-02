"""NVIDIA's OpenAI-compatible endpoint (Nemotron, Kimi, ...).

Same surface as GeminiClient -- `generate(prompt) -> Reply`, the same transient
retry budget, the same fatal-vs-transient split -- so scripts/generate_neutral.py
does not care which backend is in use.
"""

import json
import random
import re
import sys
import threading
import time

import requests

from .client import (
    FatalAPIError, RateGate, Reply, TransientExhausted, _fatal_transport, _short,
)
from .prompt import SYSTEM, system_prompt

URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# A bad key, an unknown model or a malformed request will not fix itself.
FATAL_STATUS = {400, 401, 403, 404}


def parse_message(msg):
    """Pull summary/conflicts out of an OpenAI-style message.

    Nemotron does not reliably honour the response schema: `conflicts` comes back
    as a list of dicts on some events and plain strings on others. Normalise to
    strings here rather than letting two shapes reach the output file.
    """
    content = (msg.get("content") or "").strip()
    if not content:
        return "", []
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return _salvage(body), []
    # the neutralise task returns {"neutral", "removed"}; fusion returns
    # {"summary", "conflicts"}. Normalise to the same two slots.
    if isinstance(obj, dict) and obj.get("neutral") and not obj.get("summary"):
        rem = obj.get("removed") or []
        # The model sometimes returns `removed` as one string instead of a list.
        # Iterating that yields one entry per CHARACTER, which is how the first
        # smoke test logged "C", "h", "a", "n", "g".
        if isinstance(rem, str):
            rem = [rem]
        obj = {"summary": obj["neutral"], "conflicts": rem}
    if not isinstance(obj, dict) or not obj.get("summary"):
        return content, []

    # Nemotron sometimes emits the whole object a second time INSIDE the summary
    # field, so `summary` comes back as a serialised {"summary": ..., "conflicts":
    # ...} blob. Unwrap it rather than shipping the JSON as the summary text --
    # 87 of the first 10,000 records were corrupted this way, and regenerating
    # does not reliably clear it because it is a formatting slip, not a sampling
    # one. Bounded loop in case it nests more than once.
    for _ in range(3):
        s = obj.get("summary")
        if not isinstance(s, str):
            break
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip()).strip()
        if not stripped.startswith("{"):
            break
        try:
            inner = json.loads(stripped)
        except json.JSONDecodeError:
            # Truncated or malformed nested object: recover the summary string
            # with a regex rather than losing the record entirely.
            m = re.search(r'"summary"\s*:\s*"(.*?)"\s*,\s*"conflicts"', stripped, re.S)
            if m:
                obj = {"summary": m.group(1).encode().decode("unicode_escape"),
                       "conflicts": obj.get("conflicts") or []}
            break
        if not isinstance(inner, dict) or not inner.get("summary"):
            break
        obj = {"summary": inner["summary"],
               "conflicts": inner.get("conflicts") or obj.get("conflicts") or []}

    raw_conf = obj.get("conflicts") or []
    if isinstance(raw_conf, str):
        raw_conf = [raw_conf]
    conflicts = []
    for c in raw_conf:
        conflicts.append(c if isinstance(c, str)
                         else json.dumps(c, ensure_ascii=False))
    return str(obj["summary"]).strip(), conflicts


# The model sometimes omits the opening `{"summary": "` but still emits the
# closing JSON syntax, so the content is prose with a tail like
#   ... convention.",\n  "conflicts": []\n}
# json.loads fails on that and the raw text -- braces included -- would become
# the summary. Strip the scaffolding instead of shipping it.
_JSON_TAIL = re.compile(
    r'"?\s*,?\s*"conflicts"\s*:\s*\[.*?\]\s*\}?\s*`*\s*$', re.S)
_JSON_HEAD = re.compile(r'^\s*\{?\s*"summary"\s*:\s*"')


def _salvage(text):
    out = _JSON_HEAD.sub("", text)
    out = _JSON_TAIL.sub("", out)
    out = out.strip().rstrip("`").strip()
    # a lone trailing quote left by the stripped tail
    if out.endswith('"') and out.count('"') % 2 == 1:
        out = out[:-1].rstrip()
    return out.strip() or text


class NvidiaClient:
    def __init__(self, cfg, api_key, gate=None, verbose=True):
        self.cfg = cfg
        self.gate = gate or RateGate(cfg.rpm)
        self.verbose = verbose
        self._key = api_key
        self._lock = threading.Lock()
        self.calls = 0
        self._session = requests.Session()

    def _body(self, prompt):
        return {
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": system_prompt(self.cfg)},
                         {"role": "user", "content": prompt}],
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_output_tokens,
            "stream": False,
            # Nemotron takes thinking through the chat template, not a flag.
            "chat_template_kwargs": {
                "enable_thinking": bool(self.cfg.thinking_level
                                        or self.cfg.thinking_budget)},
        }

    def _backoff(self, attempt):
        return min(self.cfg.backoff_base ** (attempt + 1),
                   self.cfg.backoff_cap) * (0.7 + 0.6 * random.random())

    def generate(self, prompt, label=""):
        last = None
        for attempt in range(self.cfg.max_network_retries + 1):
            self.gate.acquire()
            with self._lock:
                self.calls += 1
            try:
                r = self._session.post(
                    URL,
                    headers={"Authorization": f"Bearer {self._key}",
                             "Accept": "application/json"},
                    json=self._body(prompt),
                    timeout=self.cfg.request_timeout)
            except (requests.RequestException, OSError) as e:
                if _fatal_transport(e):
                    raise FatalAPIError(str(e)) from e
                last = e
            else:
                if r.status_code == 200:
                    return self._reply(r.json())
                if r.status_code in FATAL_STATUS:
                    raise FatalAPIError(f"HTTP {r.status_code}: {r.text[:200]}")
                last = RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
                if r.status_code == 429:
                    self.gate.pause(self._backoff(attempt))

            if attempt == self.cfg.max_network_retries:
                break
            delay = self._backoff(attempt)
            self._log(f"{label}transient ({attempt + 1}/"
                      f"{self.cfg.max_network_retries}), retry in "
                      f"{delay:.1f}s: {_short(last)}")
            time.sleep(delay)

        raise TransientExhausted(
            f"{self.cfg.max_network_retries} retries exhausted: {_short(last)}")

    def _reply(self, d):
        choice = (d.get("choices") or [{}])[0]
        text, conflicts = parse_message(choice.get("message") or {})
        u = d.get("usage") or {}
        return Reply(text, conflicts,
                     str(choice.get("finish_reason") or ""),
                     {"prompt_tokens": u.get("prompt_tokens"),
                      "output_tokens": u.get("completion_tokens"),
                      "total_tokens": u.get("total_tokens")},
                     None,
                     d.get("model"),
                     d.get("id"))

    def _log(self, msg):
        if self.verbose:
            print(f"  {msg}", file=sys.stderr, flush=True)
