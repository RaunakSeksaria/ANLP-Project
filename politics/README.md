# POLITICS ideology classifier (Left / Centre / Right)

A news-article ideology classifier: the POLITICS language model (Liu et al., 2022) fine-tuned on the
Article-Bias-Prediction corpus (Baly et al., 2020). Input is an article body; output is
P(Left), P(Centre), P(Right).

## Links

| What | Link |
|---|---|
| POLITICS paper (arXiv) | https://arxiv.org/abs/2205.00619 |
| POLITICS paper (ACL Anthology, Findings of NAACL 2022) | https://aclanthology.org/2022.findings-naacl.101/ |
| POLITICS code (pretraining only; no fine-tuning code released) | https://github.com/launchnlp/POLITICS |
| Pretrained checkpoint (HuggingFace, `launch/POLITICS`) | https://huggingface.co/launch/POLITICS |
| Fine-tuning data: Article-Bias-Prediction corpus + official splits | https://github.com/ramybaly/Article-Bias-Prediction |
| Data paper (Baly et al., EMNLP 2020) | https://arxiv.org/abs/2010.05338 |
| Our fine-tuning script | `politics/politics_baly.py` (this repo) |
| Our fine-tuned weights (`politics_media.pt`, `politics_random.pt`) | https://www.kaggle.com/datasets/harshitlalwani/politics-ideology-classifier |

**License:** POLITICS is CC BY-NC-SA 4.0. The fine-tuned weights are a derivative, so the same terms
apply: non-commercial use, share-alike.

## Which model to use

| Model | Trained on | Use when |
|---|---|---|
| `politics_media.pt` | Media split: train and test outlets disjoint | **Default.** Articles from outlets the model may not have seen |
| `politics_random.pt` | Random split: same outlets in train and test | Articles from outlets in the training corpus |

## Usage

```bash
pip install torch transformers
```

```python
import torch, torch.nn as nn
from transformers import AutoModel, AutoTokenizer

ENCODER = "launch/POLITICS"
LABELS = ["Left", "Centre", "Right"]

class PoliticsClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(ENCODER)
        h = self.enc.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(h, 768), nn.Tanh(),
                                  nn.Dropout(0.1), nn.Linear(768, 3))

    def forward(self, input_ids, attention_mask):
        cls = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.head(cls)

tok = AutoTokenizer.from_pretrained(ENCODER)
model = PoliticsClassifier()
model.load_state_dict(torch.load("politics_media.pt", map_location="cpu", weights_only=True))
model.eval()

texts = ["...article body...", "...another article body..."]
enc = tok(texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
with torch.no_grad():
    probs = torch.softmax(model(enc["input_ids"], enc["attention_mask"]), dim=-1)

for p in probs:
    print({label: round(float(x), 3) for label, x in zip(LABELS, p)})
```

- Pass the **article body only** (no title). That is how it was trained and evaluated.
- Text beyond 512 tokens is truncated.
- Loading prints a warning about unused `lm_head.*` weights and missing `pooler.*` weights. Both are
  expected: the classifier uses neither.

## Retrain it yourself

```bash
git clone https://github.com/ramybaly/Article-Bias-Prediction external/Article-Bias-Prediction
python baly/baly.py prep        # -> data/baly_articles.csv
python politics/politics_baly.py --split media --save-weights --out results     # or --split random
```

A GPU with ≥ 16 GB is needed at batch 16; the runs here used Kaggle's 2× T4, about 11–12 min per epoch.
`politics/kaggle/{media,random}/run.py` are the exact Kaggle scripts.

## How it was fine-tuned

The POLITICS authors released only the pretrained model, not fine-tuning code. The setup below follows
the paper's fine-tuning description (§5.4, Table A7) wherever it gives one.

| Setting | Value | Source |
|---|---|---|
| Base model | `launch/POLITICS` (RoBERTa-base, further pretrained on ~3.6M news articles) | paper |
| Classifier input | `<s>` (first-token / [CLS]) vector of the last layer | paper §5.4 |
| Head | 2-layer feed-forward: Dropout 0.1 → Linear(768, 768) → Tanh → Dropout 0.1 → Linear(768, 3) | Table A7 (2 layers, hidden 768, dropout 0.1); Tanh is our choice |
| Labels | 0 Left, 1 Centre, 2 Right | Baly corpus |
| Training data | Article-Bias-Prediction, official released splits: random 27,978 / 6,996 / 1,300; media 26,590 / 2,356 / 1,300 (train / val / test) | Baly repo |
| Input | Article body, truncated to 512 tokens | ours |
| Optimiser | AdamW, lr 2e-5, weight decay 0.001, betas (0.9, 0.999) | Table A7 (paper lists lr 1e-5 or 2e-5) |
| Schedule | Linear decay with 6% warm-up | Table A7 |
| Batch size | 16 | ours |
| Gradient clipping | 1.0 | ours |
| Precision | Mixed precision (fp16 autocast) | ours |
| Epochs | Max 5, early stopping patience 2 on validation macro-F1; best epoch kept | ours (paper: 10 / patience 4), to match our Baly baselines |
| Seed | 42, single run | ours |
| Loss | Cross-entropy | standard |

Selected checkpoints: the random-split model is from epoch 5 (validation macro-F1 90.85); the
media-split model is from epoch 3 (validation macro-F1 32.81).

## Performance

Article-Bias-Prediction test set (1,300 articles):

| Model | Macro-F1 | Accuracy | MAE |
|---|---|---|---|
| `politics_random.pt` | 89.1 | 89.3 | 0.16 |
| `politics_media.pt` | 53.1 | 55.1 | 0.54 |
| BERT-base, same setup (for comparison) | 82.2 random / 37.1 media | 81.9 / 42.1 | 0.30 / 0.82 |

Team `stories_10k` data (30,000 labelled articles):

| Model | Macro-F1 | Accuracy | MAE |
|---|---|---|---|
| `politics_random.pt` | 61.0 | 63.9 | 0.41 |
| `politics_media.pt` | 63.2 | 64.1 | 0.40 |

## Caveats

- **Labels are outlet-level.** Each article inherits its outlet's AllSides rating, so the model
  partly learns outlet style rather than article-level ideology.
- **Possible pretraining overlap.** POLITICS was pretrained on articles from 11 US outlets: Daily Kos,
  HuffPost, CNN, Washington Post, NYT, USA Today, AP, The Hill, Washington Times, Fox and Breitbart.
  Scores on articles from these outlets, including the media split's "unseen" test outlets, may be
  inflated.
- **Centre is the weakest class.** Recall on `stories_10k` is about 0.2 (random) and about 0.5 (media).
- **Single seed.** A rerun with a different random state moved the random-split model's
  predicted-Centre share on neutral summaries by about 11 points.
