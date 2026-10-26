
import json, sys, time, torch
from transformers import AutoTokenizer, LEDForConditionalGeneration
inp, out, shard, nshard, gpu, repo, sub, limit = sys.argv[1:9]
shard, nshard, limit = int(shard), int(nshard), int(limit)
rows = [json.loads(l) for l in open(inp)]
if limit: rows = rows[:limit]
rows = rows[shard::nshard]
try:
    tok = AutoTokenizer.from_pretrained(repo, subfolder=sub)
except Exception as e:
    print("tokenizer fallback:", type(e).__name__, flush=True)
    tok = AutoTokenizer.from_pretrained("allenai/led-base-16384")
dev = f"cuda:{gpu}"
model = LEDForConditionalGeneration.from_pretrained(repo, subfolder=sub).to(dev).half().eval()
norm = lambda s: " ".join(str(s).split())
texts = [f"{norm(r['title'])} {tok.sep_token} {norm(r['body'])}" for r in rows]
lens = [len(tok(x, truncation=True, max_length=512)["input_ids"]) for x in texts]
order = sorted(range(len(rows)), key=lambda i: -lens[i])
t0, i = time.time(), 0
with open(out, "w") as fh:
    while i < len(order):
        bs = max(1, 12288 // (lens[order[i]] + 512))
        idx = order[i:i + bs]
        enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True, max_length=512).to(dev)
        g = torch.zeros_like(enc["input_ids"]); g[:, 0] = 1
        with torch.no_grad():
            ids = model.generate(**enc, global_attention_mask=g, num_beams=1, do_sample=False, max_new_tokens=512)
        for j, o, n in zip(idx, tok.batch_decode(ids, skip_special_tokens=True), (ids != tok.pad_token_id).sum(1)):
            fh.write(json.dumps({"id": rows[j]["id"], "output": o.strip(), "input_tokens": lens[j],
                                 "truncated": lens[j] >= 512, "output_tokens": int(n), "hit_cap": int(n) >= 512}) + "\n")
        fh.flush(); i += bs
        print(f"  shard{shard} {i}/{len(order)} {(time.time()-t0)/60:.1f} min", flush=True)
