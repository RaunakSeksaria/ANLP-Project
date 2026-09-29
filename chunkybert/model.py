"""ChunkedBERTClassifier.

Verbatim from the official ChunkyBERT source (https://github.com/dloiya/chunkybert,
cell 2 of chunkybert-final.ipynb), the code released alongside:

    Loiya, D., Kulal, S. S., Reddy, M. S. M. & Shetty, R. D.
    "ChunkyBERT: a novel technique for multiclass political bias detection in
    news media." Scientific Reports 16, 15323 (2026).
    https://doi.org/10.1038/s41598-026-46646-z

No architectural changes were made.
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class ChunkedBERTClassifier(nn.Module):
    def __init__(self, num_classes, bert_model_name='bert-base-uncased',
                 hidden_size=256, nhead=8, dropout=0.3, max_chunk_length=512):
        super().__init__()
        self.max_chunk_length = max_chunk_length

        bert_config = BertConfig.from_pretrained(bert_model_name)
        bert_config.output_hidden_states = False
        bert_config.return_dict = True

        self.bert = BertModel.from_pretrained(bert_model_name, config=bert_config)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bert_config.hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.attention = nn.Sequential(
            nn.Linear(bert_config.hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1, bias=False)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bert_config.hidden_size, num_classes)
        )

    def chunk_text_batched(self, input_ids, attention_mask, tokenizer):
        """Optimized batched chunking"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        cls_token_id = tokenizer.cls_token_id
        sep_token_id = tokenizer.sep_token_id
        pad_token_id = tokenizer.pad_token_id

        chunk_content_size = self.max_chunk_length - 2

        all_chunk_ids = []
        all_chunk_masks = []
        chunk_to_batch_idx = []

        for i in range(batch_size):
            seq_len_i = attention_mask[i].sum().item()
            seq = input_ids[i, :seq_len_i]

            # Remove CLS and SEP if present
            if seq[0] == cls_token_id and seq[-1] == sep_token_id:
                content = seq[1:-1]
            else:
                content = seq

            # Create chunks
            num_chunks = (len(content) + chunk_content_size - 1) // chunk_content_size

            for j in range(num_chunks):
                start_idx = j * chunk_content_size
                end_idx = min(start_idx + chunk_content_size, len(content))
                chunk_content = content[start_idx:end_idx]

                # Build chunk with CLS and SEP
                chunk = torch.cat([
                    torch.tensor([cls_token_id], device=device),
                    chunk_content,
                    torch.tensor([sep_token_id], device=device)
                ])

                # Pad
                pad_len = self.max_chunk_length - len(chunk)
                if pad_len > 0:
                    chunk = torch.cat([chunk, torch.full((pad_len,), pad_token_id, device=device)])

                mask = torch.cat([
                    torch.ones(len(chunk) - pad_len, device=device),
                    torch.zeros(pad_len, device=device)
                ])

                all_chunk_ids.append(chunk)
                all_chunk_masks.append(mask)
                chunk_to_batch_idx.append(i)

        if not all_chunk_ids:
            return None, None, None

        return (
            torch.stack(all_chunk_ids),
            torch.stack(all_chunk_masks),
            torch.tensor(chunk_to_batch_idx, device=device)
        )

    def forward(self, input_ids, attention_mask, tokenizer):
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Batch all chunks together
        chunk_ids, chunk_masks, chunk_to_batch = self.chunk_text_batched(
            input_ids, attention_mask, tokenizer
        )

        if chunk_ids is None:
            return torch.zeros(batch_size, self.classifier[-1].out_features, device=device)

        # Process all chunks in parallel
        outputs = self.bert(chunk_ids, attention_mask=chunk_masks)
        sequence_output = outputs.last_hidden_state

        # Transformer encoding
        sequence_output = self.transformer(sequence_output)

        # Attention pooling
        attn_weights = torch.softmax(self.attention(sequence_output), dim=1)
        pooled_chunks = torch.sum(attn_weights * sequence_output, dim=1)

        # Aggregate chunks per batch item
        batch_embeddings = []
        for i in range(batch_size):
            mask = chunk_to_batch == i
            if mask.any():
                item_chunks = pooled_chunks[mask]
                item_embedding = item_chunks.mean(dim=0)
                batch_embeddings.append(item_embedding)
            else:
                batch_embeddings.append(torch.zeros(pooled_chunks.size(1), device=device))

        final_embeddings = torch.stack(batch_embeddings)
        logits = self.classifier(final_embeddings)

        return logits
