"""Every knob in one place.

Sampling parameters are fields here rather than literals at the call site so that
each output record can carry the exact settings that produced it. There is no
response cache: a rerun calls the API again and is expected to differ in wording.
`scripts/check_consistency.py` measures how much.
"""

from dataclasses import asdict, dataclass, field, fields


@dataclass
class GenConfig:
    # --- model and sampling -------------------------------------------------
    model: str = "gemini-flash-latest"
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 1
    seed: int = 42
    max_output_tokens: int = 4096
    # Gemini 3.x controls thinking with thinking_level (MINIMAL/LOW/MEDIUM/HIGH);
    # thinking_budget is the older knob and is silently IGNORED there -- budget=2048
    # measurably produced 0 thought tokens on gemini-3.7-flash. When thinking_level
    # is set it wins; otherwise the budget is sent, which still works on 2.x.
    # Default off: unpinned thinking ran ~918 tok/call elsewhere, which would roughly
    # triple this job's output bill. Whether it improves the summaries is UNTESTED --
    # the A/B was blocked by quota, not answered.
    thinking_budget: int = 0        # 0 disables, -1 automatic (2.x models)
    thinking_level: str = ""        # "", MINIMAL, LOW, MEDIUM, HIGH (3.x models)

    # --- length target ------------------------------------------------------
    # Soft: mean word count over the cluster's articles, +/- tolerance. The hard
    # ratios are a sanity guard that triggers a malformed-output retry, not a
    # style preference.
    # Wide on purpose: human summary length correlates only r=0.58 with source
    # length, so a tight band manufactures revisions that are not real failures.
    length_tolerance: float = 0.25
    hard_min_ratio: float = 0.4
    hard_max_ratio: float = 2.5

    # --- three separate retry budgets --------------------------------------
    max_network_retries: int = 3     # timeouts, 429, 5xx. Prompt unchanged.
    max_malformed_retries: int = 3   # empty/unparseable/truncated/absurd length.
    max_revisions: int = 0           # eval-metric feedback loop. 0 = off.
    backoff_base: float = 2.0
    backoff_cap: float = 60.0

    # --- prompt -------------------------------------------------------------
    # Sources are anonymised by default: telling the model which article is the
    # left one invites it to balance (give each side its turn) rather than
    # neutralise. Revealing them is an ablation, not the default.
    reveal_stance: bool = False
    reveal_publisher: bool = False
    shuffle_seed: int = 42

    # --- input field names --------------------------------------------------
    # A different key layout is a config change, not a code change.
    id_field: str = "story_id"
    articles_field: str = "articles"
    stance_field: str = "stance"
    title_field: str = "title"
    body_field: str = "body"
    publisher_field: str = "publisher"

    # --- stance grouping, for the parity metric only ------------------------
    # evaluate() sees the stances even when the prompt does not.
    left_stances: tuple = ("left", "far-left", "lean-left", "left-center", "leaning left")
    right_stances: tuple = ("right", "far-right", "lean-right", "right-center", "leaning right")

    # --- evaluate() thresholds ---------------------------------------------
    # Deliberately loose; revisions are off by default. Calibrate these on the
    # NeuS test split, where a human neutral roundup exists to score against,
    # before turning --max-revisions up.
    max_new_arousal_rate: float = 0.35     # arousal per introduced VAD token
    min_vad_tokens_for_rate: int = 5       # below this the rate is noise, so skip it
    max_new_affect_tokens: int = 8         # high-arousal words no source used
    max_bias_lexicon_hits: int = 3         # introduced hits in NeuS's bias lexicons
    min_source_coverage_ratio: float = 0.55  # worst source's recall / mean recall
    max_stance_parity_gap: float = 0.12    # |ROUGE-1 vs left - vs right|
    min_number_retention: float = 0.80     # shared numbers that survive
    enforce_length: bool = True

    # --- throughput ---------------------------------------------------------
    workers: int = 4
    rpm: int = 60

    def params(self):
        """The subset that defines a generation, for the output record."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "max_output_tokens": self.max_output_tokens,
            "thinking_budget": self.thinking_budget,
            "thinking_level": self.thinking_level,
            "reveal_stance": self.reveal_stance,
            "reveal_publisher": self.reveal_publisher,
        }

    def to_dict(self):
        return asdict(self)

    @classmethod
    def field_names(cls):
        return [f.name for f in fields(cls)]
