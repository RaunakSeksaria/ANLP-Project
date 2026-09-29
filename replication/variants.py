"""One parameterised classifier covering every model in the ChunkyBERT paper.

Table 8 baselines (GRU/TCN × chunking × attention) and Table 10 ablations are all
points in the same configuration space, so they share one implementation:

    backbone   bert | embedding      what produces token vectors
    encoder    transformer | gru | tcn | none
    pooling    attention | mean      how tokens collapse to a chunk vector
    chunk_agg  mean | max            how chunk vectors collapse to a document
    chunking   True | False
    simple_head / freeze_backbone

ChunkyBERT itself is backbone=bert, encoder=transformer, pooling=attention,
chunk_agg=mean, chunking=True — identical in behaviour to chunkybert/model.py.

The chunking routine is taken from the authors' released code (chunkybert/model.py).
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

BERT_DIM = 768


class Chomp(nn.Module):
    """Trim right-hand padding so a dilated conv stays causal."""
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[:, :, :-self.n] if self.n > 0 else x


class TCN(nn.Module):
    """Dilated causal convolutions, as described for the TCN baseline (ref 44)."""
    def __init__(self, d, levels=4, k=3, dropout=0.3):
        super().__init__()
        layers = []
        for i in range(levels):
            pad = (k - 1) * 2**i
            layers += [nn.Conv1d(d, d, k, padding=pad, dilation=2**i),
                       Chomp(pad), nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):                       # (B, T, D)
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class VariantClassifier(nn.Module):
    def __init__(self, num_classes=3, backbone="bert", encoder="transformer",
                 pooling="attention", chunk_agg="mean", chunking=True,
                 freeze_backbone=False, simple_head=False,
                 hidden_size=256, nhead=8, dropout=0.3, max_chunk_length=512,
                 emb_dim=256, vocab_size=30522, bert_model_name="bert-base-uncased",
                 tcn_levels=4):
        super().__init__()
        self.cfg = dict(backbone=backbone, encoder=encoder, pooling=pooling,
                        chunk_agg=chunk_agg, chunking=chunking,
                        freeze_backbone=freeze_backbone, simple_head=simple_head)
        self.max_chunk_length = max_chunk_length
        self.chunking = chunking
        self.pooling = pooling
        self.chunk_agg = chunk_agg
        self.backbone_kind = backbone

        if backbone == "bert":
            cfg = BertConfig.from_pretrained(bert_model_name)
            cfg.output_hidden_states = False
            cfg.return_dict = True
            self.bert = BertModel.from_pretrained(bert_model_name, config=cfg)
            d = cfg.hidden_size
            if freeze_backbone:
                for p in self.bert.parameters():
                    p.requires_grad = False
        else:
            d = emb_dim
            self.embed = nn.Embedding(vocab_size, d, padding_idx=0)

        if encoder == "transformer":
            layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead,
                                               dim_feedforward=hidden_size * 4,
                                               dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        elif encoder == "gru":
            self.encoder = nn.GRU(d, d // 2, batch_first=True, bidirectional=True)
        elif encoder == "tcn":
            self.encoder = TCN(d, levels=tcn_levels, dropout=dropout)
        elif encoder == "none":
            self.encoder = None
        else:
            raise ValueError(encoder)
        self.encoder_kind = encoder

        self.attention = nn.Sequential(nn.Linear(d, hidden_size), nn.GELU(),
                                       nn.Linear(hidden_size, 1, bias=False))
        self.classifier = (nn.Linear(d, num_classes) if simple_head
                           else nn.Sequential(nn.Dropout(dropout), nn.Linear(d, num_classes)))
        self.out_features = num_classes

    # --- chunking: from the authors' released model.py ---
    def chunk_text_batched(self, input_ids, attention_mask, tokenizer):
        batch_size, _ = input_ids.shape
        device = input_ids.device
        cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
        content_size = self.max_chunk_length - 2
        ids, masks, owner = [], [], []

        for i in range(batch_size):
            n = attention_mask[i].sum().item()
            seq = input_ids[i, :n]
            content = seq[1:-1] if (n >= 2 and seq[0] == cls_id and seq[-1] == sep_id) else seq
            for j in range((len(content) + content_size - 1) // content_size):
                piece = content[j * content_size:(j + 1) * content_size]
                chunk = torch.cat([torch.tensor([cls_id], device=device), piece,
                                   torch.tensor([sep_id], device=device)])
                pad = self.max_chunk_length - len(chunk)
                if pad > 0:
                    chunk = torch.cat([chunk, torch.full((pad,), pad_id, device=device)])
                masks.append(torch.cat([torch.ones(len(chunk) - pad, device=device),
                                        torch.zeros(pad, device=device)]))
                ids.append(chunk)
                owner.append(i)

        if not ids:
            return None, None, None
        return torch.stack(ids), torch.stack(masks), torch.tensor(owner, device=device)

    def encode(self, ids, mask):
        if self.backbone_kind == "bert":
            h = self.bert(ids, attention_mask=mask).last_hidden_state
        else:
            h = self.embed(ids)
        if self.encoder_kind == "gru":
            h, _ = self.encoder(h)
        elif self.encoder is not None:
            h = self.encoder(h)
        return h

    def pool(self, h, mask):
        m = mask.unsqueeze(-1).to(h.dtype)
        if self.pooling == "attention":
            scores = self.attention(h)
            scores = scores.masked_fill(m == 0, torch.finfo(scores.dtype).min)
            return (torch.softmax(scores, dim=1).to(h.dtype) * h).sum(1)
        return (h * m).sum(1) / m.sum(1).clamp(min=1)          # masked mean

    def forward(self, input_ids, attention_mask, tokenizer):
        B = input_ids.size(0)
        device = input_ids.device

        if not self.chunking:
            # BERT cannot exceed its 512-token limit; GRU/TCN read the whole document.
            if self.backbone_kind == "bert":
                ids, mask = input_ids[:, :self.max_chunk_length], attention_mask[:, :self.max_chunk_length]
            else:
                ids, mask = input_ids, attention_mask
            return self.classifier(self.pool(self.encode(ids, mask), mask))

        ids, mask, owner = self.chunk_text_batched(input_ids, attention_mask, tokenizer)
        if ids is None:
            return torch.zeros(B, self.out_features, device=device)

        pooled = self.pool(self.encode(ids, mask), mask)

        docs = []
        for i in range(B):
            sel = owner == i
            if sel.any():
                v = pooled[sel]
                docs.append(v.max(0).values if self.chunk_agg == "max" else v.mean(0))
            else:
                docs.append(torch.zeros(pooled.size(1), device=device))
        return self.classifier(torch.stack(docs))


# Every model in the paper, as configuration.
TABLE8 = {
    "GRU no chunk, no attn":  dict(backbone="embedding", encoder="gru", pooling="mean",      chunking=False),
    "GRU no chunk, attn":     dict(backbone="embedding", encoder="gru", pooling="attention", chunking=False),
    "GRU with chunk, no attn":dict(backbone="embedding", encoder="gru", pooling="mean",      chunking=True),
    "GRU with chunk, attn":   dict(backbone="embedding", encoder="gru", pooling="attention", chunking=True),
    "TCN no chunk, no attn":  dict(backbone="embedding", encoder="tcn", pooling="mean",      chunking=False),
    "TCN no chunk, attn":     dict(backbone="embedding", encoder="tcn", pooling="attention", chunking=False),
    "TCN with chunk, no attn":dict(backbone="embedding", encoder="tcn", pooling="mean",      chunking=True),
    "TCN with chunk, attn":   dict(backbone="embedding", encoder="tcn", pooling="attention", chunking=True),
}

TABLE10 = {
    "Full Model":      dict(),
    "No Attention":    dict(pooling="mean"),
    "No Classifier":   dict(simple_head=True),
    "Max Pool":        dict(chunk_agg="max"),
    "No Chunking":     dict(chunking=False),
    "Freeze BERT":     dict(freeze_backbone=True),
    "No Transformers": dict(encoder="none"),
    # Hypothesis: the paper's 41% / 5-min figure came from dropping BERT too, not just the
    # 2-layer encoder. That leaves embedding -> attention pooling -> classifier.
    "No Transformers (no BERT)": dict(backbone="embedding", encoder="none"),
}
