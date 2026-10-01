"""Generate neutral rewrites with RaunakSeksaria/anlp-neutral-led (side-neutral-full/lambda{0,035}).

Settings are the model card's / training config's: input "title </s> body" truncated to 512 tokens,
global attention on token 0, greedy, max 512 new tokens. Appends per batch, so a rerun resumes.

    python led_eval/generate.py --set test         # 2,000 test-split left/right articles of stories_10k
    python led_eval/generate.py --set diversesumm  # 2,450 DiverseSumm articles (QA coverage eval)
"""
import argparse, json, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out"
MODELS = {"led_lambda0": "side-neutral-full/lambda0", "led_lambda035": "side-neutral-full/lambda035"}
HF_REPO = "RaunakSeksaria/anlp-neutral-led"
LOCAL = ROOT / "models/neutral_led"
MAX_IN = MAX_NEW = 512


def norm(s):
    return " ".join(s.split())


def items(which, root):
    if which == "test":
        out = []
        for s in map(json.loads, open(root / "stories_10k.jsonl")):
            if s["split"] == "test":
                out += [(f"{s['story_id']}:{side}", s[side]["title"], s[side]["body"]) for side in ("left", "right")]
        return out
    ds = json.load(open(root / "diverse_summ.json"))
    return [(a["aid"], a["title"], a["content"]) for s in ds for a in s["articles"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["test", "diversesumm"], default="test")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--data", default=None, help="dir holding stories_10k.jsonl / diverse_summ.json")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tokens-per-batch", type=int, default=8192)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, LEDForConditionalGeneration
    data = Path(a.data) if a.data else None
    if data is None:
        data = ROOT / ("data" if a.set == "test" else "external/DiverseSumm/data")
    rows = items(a.set, data)[: a.limit]
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    dev = f"cuda:{a.gpu}"

    for name in a.models.split(","):
        sub = MODELS[name]
        src = (LOCAL / sub) if (LOCAL / sub).exists() else None
        tok = (AutoTokenizer.from_pretrained(src) if src else AutoTokenizer.from_pretrained(HF_REPO, subfolder=sub))
        model = (LEDForConditionalGeneration.from_pretrained(src) if src
                 else LEDForConditionalGeneration.from_pretrained(HF_REPO, subfolder=sub))
        model = model.to(dev).half().eval()
        path = out_dir / (f"gen_{name}.jsonl" if a.set == "test" else f"gen_diversesumm_{name}.jsonl")
        done = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()
        todo = [r for r in rows if r[0] not in done]
        texts = [f"{norm(t)} {tok.sep_token} {norm(b)}" for _, t, b in todo]
        lens = [len(tok(x, truncation=True, max_length=MAX_IN)["input_ids"]) for x in texts]
        order = sorted(range(len(todo)), key=lambda i: -lens[i])
        print(f"{name}: {len(done)} done, {len(todo)} to go", flush=True)
        t0, i = time.time(), 0
        with open(path, "a") as fh:
            while i < len(order):
                # output length ~ input length, so budget on input length + MAX_NEW
                bs = max(1, a.tokens_per_batch // (lens[order[i]] + MAX_NEW))
                idx = order[i:i + bs]
                enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True,
                          truncation=True, max_length=MAX_IN).to(dev)
                g = torch.zeros_like(enc["input_ids"]); g[:, 0] = 1
                with torch.no_grad():
                    ids = model.generate(**enc, global_attention_mask=g, num_beams=1, do_sample=False,
                                         max_new_tokens=MAX_NEW)
                for j, o, n in zip(idx, tok.batch_decode(ids, skip_special_tokens=True), (ids != tok.pad_token_id).sum(1)):
                    fh.write(json.dumps({"id": todo[j][0], "output": o.strip(), "input_tokens": lens[j],
                                         "truncated": lens[j] >= MAX_IN, "output_tokens": int(n),
                                         "hit_cap": int(n) >= MAX_NEW}) + "\n")
                fh.flush()
                i += bs
                el = time.time() - t0
                print(f"  {name} {i}/{len(order)}  {el/60:.1f} min, eta {el/i*(len(order)-i)/60:.1f} min", flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
