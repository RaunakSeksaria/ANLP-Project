"""Fine-tune BART on NeuS, optionally with the event-aware triplet loss.

One script, two runs, one comparison:

    python scripts/train_neus.py --run-name baseline --triplet-weight 0
    python scripts/train_neus.py --run-name triplet  --triplet-weight 1.0

Each run writes into runs/<run-name>/:
    test_generations.txt   decoded neutral summaries for the test split
    vectors.npz            pooled encoder representation of all 9,282 articles,
                           so scripts/probe.py can measure publisher leakage
                           on this model rather than on a pretrained one
    metrics.json           losses, timings, config

The triplet objective, following the interim proposal: for an anchor article,
the positive is a *different publisher's* report of the same story and the
negative is the *same publisher's* report of a different story. Anchor and
negative sharing a publisher is the whole mechanism -- it is the only way to
satisfy the margin without abandoning publisher style as the organising signal.

Structurally this mirrors NeuS's own auxiliary-task hook
(src/transformers/models/bart/modeling_bart.py:1377-1533 and
src/finetune.py:203-224), which adds `task_loss_ratio * extra_task_loss` to the
LM loss over a pooled encoder state. Same attachment point, different loss,
current transformers instead of their 4.3.3 fork.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "data/prepared"
RELEASED = ROOT / "data/framing-bias-metric-main/data/naacl2022_lrc_roundup_random_order_probe"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

class GenerationSet(Dataset):
    """NeuS's released source/target files: three articles in randomised
    left/centre/right order, one human roundup."""

    def __init__(self, data_dir, split):
        self.src = (Path(data_dir) / f"{split}.source").read_text().splitlines()
        self.tgt = (Path(data_dir) / f"{split}.target").read_text().splitlines()
        assert len(self.src) == len(self.tgt), (len(self.src), len(self.tgt))

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        return self.src[i], self.tgt[i]


class TripletSet(Dataset):
    def __init__(self, split):
        articles = {a["article_id"]: a for a in
                    (json.loads(l) for l in (PREP / "articles.jsonl").open())}
        self.articles = articles
        keep = {split, None} if split == "train" else {split}
        self.rows = [
            t for t in (json.loads(l) for l in (PREP / "triplets.jsonl").open())
            if t["split"] in keep
            and all(t[k] in articles for k in ("anchor", "positive", "negative"))
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        t = self.rows[i]
        return tuple(self.articles[t[k]]["body"] for k in ("anchor", "positive", "negative"))


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


# --------------------------------------------------------------------------- #
# model helpers
# --------------------------------------------------------------------------- #

def pool(hidden, attention_mask, input_ids, eos_id):
    """NeuS pools the encoder state at the EOS position. Fall back to a masked
    mean when a sequence has no EOS (truncation can remove it)."""
    eos = input_ids.eq(eos_id)
    has_eos = eos.any(dim=1)
    mask = attention_mask.unsqueeze(-1).float()
    mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
    if not has_eos.any():
        return mean
    # last EOS per sequence
    idx = (eos.float() * torch.arange(eos.size(1), device=eos.device)).argmax(dim=1)
    gathered = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
    return torch.where(has_eos.unsqueeze(-1), gathered, mean)


def encode_pooled(model, tokenizer, texts, device, max_len, autocast_dtype):
    enc = tokenizer(list(texts), return_tensors="pt", padding=True,
                    truncation=True, max_length=max_len).to(device)
    with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                        enabled=autocast_dtype is not None):
        out = model.get_encoder()(input_ids=enc.input_ids,
                                  attention_mask=enc.attention_mask)
    return pool(out.last_hidden_state, enc.attention_mask,
                enc.input_ids, tokenizer.eos_token_id)


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--model", default="facebook/bart-large")
    ap.add_argument("--data-dir", default=str(RELEASED))
    ap.add_argument("--triplet-weight", type=float, default=0.0,
                    help="0 reproduces the generation-only baseline")
    ap.add_argument("--triplet-margin", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--triplet-batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-source", type=int, default=512)
    ap.add_argument("--max-target", type=int, default=250)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--limit-train", type=int, default=None, help="smoke-test switch")
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                      "fp32": None}[args.precision]
    if device.type == "cpu":
        autocast_dtype = None
    print(f"device={device} precision={args.precision} triplet_weight={args.triplet_weight}",
          flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    # transformers >=5 rejects generation settings written onto model.config
    model.generation_config.forced_bos_token_id = tokenizer.bos_token_id
    model.config.forced_bos_token_id = None

    train_set = GenerationSet(args.data_dir, "train")
    if args.limit_train:
        train_set.src = train_set.src[: args.limit_train]
        train_set.tgt = train_set.tgt[: args.limit_train]
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: b, drop_last=False)

    triplet_iter = None
    if args.triplet_weight > 0:
        tset = TripletSet("train")
        print(f"triplet stream: {len(tset)} triples", flush=True)
        triplet_iter = cycle(DataLoader(tset, batch_size=args.triplet_batch_size,
                                        shuffle=True, collate_fn=lambda b: b,
                                        drop_last=True))

    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(optim, args.warmup, total_steps)
    scaler = torch.amp.GradScaler(enabled=(args.precision == "fp16"))
    print(f"{len(train_set)} examples · {steps_per_epoch} optimiser steps/epoch · "
          f"{total_steps} total", flush=True)

    history = []
    t0 = time.time()
    model.train()
    for epoch in range(args.epochs):
        running = {"lm": 0.0, "trip": 0.0, "n": 0}
        optim.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            sources = [s for s, _ in batch]
            targets = [t for _, t in batch]
            enc = tokenizer(sources, return_tensors="pt", padding=True,
                            truncation=True, max_length=args.max_source).to(device)
            lab = tokenizer(targets, return_tensors="pt", padding=True,
                            truncation=True, max_length=args.max_target).to(device)
            labels = lab.input_ids.masked_fill(lab.input_ids == tokenizer.pad_token_id, -100)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=autocast_dtype is not None):
                lm_loss = model(input_ids=enc.input_ids,
                                attention_mask=enc.attention_mask,
                                labels=labels).loss

            trip_loss = torch.zeros((), device=device)
            if triplet_iter is not None:
                trip = next(triplet_iter)
                anchors, positives, negatives = zip(*trip)
                embeds = encode_pooled(model, tokenizer,
                                       list(anchors) + list(positives) + list(negatives),
                                       device, args.max_source, autocast_dtype)
                a, p, n = embeds.chunk(3, dim=0)
                trip_loss = F.triplet_margin_loss(a.float(), p.float(), n.float(),
                                                  margin=args.triplet_margin)

            loss = lm_loss + args.triplet_weight * trip_loss
            scaler.scale(loss / args.grad_accum).backward()

            running["lm"] += lm_loss.item()
            running["trip"] += trip_loss.detach().item()
            running["n"] += 1

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                sched.step()
                optim.zero_grad(set_to_none=True)

            if running["n"] % 100 == 0:
                print(f"  epoch {epoch} step {step + 1}/{len(train_loader)} "
                      f"lm={running['lm'] / running['n']:.4f} "
                      f"trip={running['trip'] / running['n']:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)

        entry = {"epoch": epoch,
                 "lm_loss": running["lm"] / max(1, running["n"]),
                 "triplet_loss": running["trip"] / max(1, running["n"]),
                 "elapsed_s": round(time.time() - t0, 1)}
        history.append(entry)
        print(f"epoch {epoch} done: {entry}", flush=True)

    # ---- generate on test ----
    model.eval()
    test_set = GenerationSet(args.data_dir, "test")
    generations = []
    with torch.inference_mode():
        for i in range(0, len(test_set), args.batch_size):
            chunk = [test_set[j] for j in range(i, min(i + args.batch_size, len(test_set)))]
            enc = tokenizer([s for s, _ in chunk], return_tensors="pt", padding=True,
                            truncation=True, max_length=args.max_source).to(device)
            ids = model.generate(**enc, num_beams=args.num_beams,
                                 max_length=args.max_target,
                                 early_stopping=args.num_beams > 1)
            generations += tokenizer.batch_decode(ids, skip_special_tokens=True)
    (out_dir / "test_generations.txt").write_text("\n".join(g.replace("\n", " ")
                                                            for g in generations))
    print(f"wrote {len(generations)} generations", flush=True)

    # ---- dump pooled encoder vectors for the leakage probe ----
    articles = [json.loads(l) for l in (PREP / "articles.jsonl").open()]
    vectors, ids = [], []
    with torch.inference_mode():
        for i in range(0, len(articles), 32):
            chunk = articles[i:i + 32]
            emb = encode_pooled(model, tokenizer, [a["body"] for a in chunk],
                                device, args.max_source, autocast_dtype)
            vectors.append(emb.float().cpu().numpy())
            ids += [a["article_id"] for a in chunk]
    np.savez_compressed(out_dir / "vectors.npz",
                        vectors=np.concatenate(vectors), ids=np.array(ids))
    print(f"wrote vectors for {len(ids)} articles", flush=True)

    (out_dir / "metrics.json").write_text(json.dumps(
        {"config": vars(args), "history": history,
         "total_s": round(time.time() - t0, 1)}, indent=2))
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
