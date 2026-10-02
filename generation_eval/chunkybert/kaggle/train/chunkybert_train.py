# ChunkyBERT — full training run on the Article-Bias-Prediction dataset.
#
# This is the official notebook code (https://github.com/dloiya/chunkybert,
# chunkybert-final.ipynb) concatenated into one script for a Kaggle "script"
# kernel, with the paper's own default hyperparameters (Table 4) left intact.
# The only functional change from the notebook: its documented unterminated
# `print(...)` at the end of main() is closed (see the repo README's "Notes"
# section), and plt.show() is swapped for plt.savefig() so plots persist to
# kernel output instead of vanishing in headless execution.
#
# Reference:
#   Loiya, D., Kulal, S. S., Reddy, M. S. M. & Shetty, R. D.
#   "ChunkyBERT: a novel technique for multiclass political bias detection in
#   news media." Scientific Reports 16, 15323 (2026).
#   https://doi.org/10.1038/s41598-026-46646-z

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
from transformers import BertTokenizer, BertModel, BertConfig, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import sys
import time


# ============================================================================
# MODEL
# ============================================================================
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

            if seq[0] == cls_token_id and seq[-1] == sep_token_id:
                content = seq[1:-1]
            else:
                content = seq

            num_chunks = (len(content) + chunk_content_size - 1) // chunk_content_size

            for j in range(num_chunks):
                start_idx = j * chunk_content_size
                end_idx = min(start_idx + chunk_content_size, len(content))
                chunk_content = content[start_idx:end_idx]

                chunk = torch.cat([
                    torch.tensor([cls_token_id], device=device),
                    chunk_content,
                    torch.tensor([sep_token_id], device=device)
                ])

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

        chunk_ids, chunk_masks, chunk_to_batch = self.chunk_text_batched(
            input_ids, attention_mask, tokenizer
        )

        if chunk_ids is None:
            return torch.zeros(batch_size, self.classifier[-1].out_features, device=device)

        outputs = self.bert(chunk_ids, attention_mask=chunk_masks)
        sequence_output = outputs.last_hidden_state

        sequence_output = self.transformer(sequence_output)

        attn_weights = torch.softmax(self.attention(sequence_output), dim=1)
        pooled_chunks = torch.sum(attn_weights * sequence_output, dim=1)

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


# ============================================================================
# OPTIMIZED DATASET
# ============================================================================
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=5000):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# ============================================================================
# CONFUSION MATRIX PLOTTING
# ============================================================================
def plot_confusion_matrix(y_true, y_pred, epoch, phase, class_names=['Left', 'Centre', 'Right']):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {phase} - Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{phase}_epoch{epoch}.png')
    plt.close()

    return cm


# ============================================================================
# ROC PLOTTING
# ============================================================================
def plot_epoch_multiclass_roc(y_true, y_probs, num_classes, epoch, phase, class_names=None):
    """Plot OvR ROC curves for one epoch"""
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(8, 6))
    plt.plot(fpr["micro"], tpr["micro"], linestyle=":", linewidth=2.5,
             label=f"Micro ROC (AUC={roc_auc['micro']:.3f})")
    plt.plot(fpr["macro"], tpr["macro"], linestyle="--", linewidth=2.5,
             label=f"Macro ROC (AUC={roc_auc['macro']:.3f})")

    for i in range(num_classes):
        name = class_names[i] if class_names else f"Class {i}"
        plt.plot(fpr[i], tpr[i], linewidth=1.5, alpha=0.75,
                 label=f"{name} (AUC={roc_auc[i]:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - {phase} - Epoch {epoch}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'roc_{phase}_epoch{epoch}.png')
    plt.close()

    return roc_auc


# ============================================================================
# TRAINING FUNCTION WITH MULTI-GPU SUPPORT
# ============================================================================
def train(
    model,
    train_dataloader,
    val_dataloader,
    tokenizer,
    device,
    num_classes,
    num_epochs=3,
    learning_rate=2e-5,
    class_names=['Left', 'Centre', 'Right'],
    use_multi_gpu=True
):
    """Optimized training with multi-GPU support"""

    if use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )

    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_val_auc = 0.0
    best_val_acc = 0.0
    best_val_loss = float('inf')

    epoch_metrics = []

    for epoch in range(num_epochs):
        epoch_start = time.time()
        # ============================
        # TRAIN
        # ============================
        model.train()
        train_loss = 0.0
        train_probs, train_targets, train_preds = [], [], []

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} [Train]", file=sys.stdout):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            with autocast():
                logits = model(input_ids, attention_mask, tokenizer)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)

            train_probs.append(probs.detach().cpu())
            train_targets.append(labels.detach().cpu())
            train_preds.append(preds.detach().cpu())

        train_loss /= len(train_dataloader)

        train_probs = torch.cat(train_probs).numpy()
        train_targets = torch.cat(train_targets).numpy()
        train_preds = torch.cat(train_preds).numpy()

        train_accuracy = (train_preds == train_targets).mean()

        train_probs = train_probs / train_probs.sum(axis=1, keepdims=True)

        try:
            unique_train_classes = np.unique(train_targets)
            if len(unique_train_classes) == num_classes:
                train_auc = roc_auc_score(
                    train_targets,
                    train_probs,
                    multi_class="ovr",
                    average="weighted"
                )
            else:
                print(f"Warning: Not all classes in training set. Found: {unique_train_classes}")
                train_auc = 0.0
        except Exception as e:
            print(f"Error calculating training AUROC: {e}")
            train_auc = 0.0

        # ============================
        # VALIDATION
        # ============================
        model.eval()
        val_loss = 0.0
        val_probs, val_targets, val_preds = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1} [Val]", file=sys.stdout):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                with autocast():
                    logits = model(input_ids, attention_mask, tokenizer)
                    loss = criterion(logits, labels)

                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)

                val_probs.append(probs.detach().cpu())
                val_targets.append(labels.detach().cpu())
                val_preds.append(preds.detach().cpu())

        val_loss /= len(val_dataloader)

        val_probs = torch.cat(val_probs).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_preds = torch.cat(val_preds).numpy()

        val_accuracy = (val_preds == val_targets).mean()

        val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)

        try:
            unique_val_classes = np.unique(val_targets)
            if len(unique_val_classes) == num_classes:
                val_auc = roc_auc_score(
                    val_targets,
                    val_probs,
                    multi_class="ovr",
                    average="weighted"
                )
            else:
                print(f"Warning: Not all classes in validation set. Found: {unique_val_classes}")
                val_auc = 0.0
        except Exception as e:
            print(f"Error calculating validation AUROC: {e}")
            val_auc = 0.0

        epoch_time = time.time() - epoch_start

        # ============================
        # PRINT METRICS
        # ============================
        print("\n" + "="*80, flush=True)
        print(f"EPOCH {epoch+1}/{num_epochs} RESULTS  (took {epoch_time:.1f}s)", flush=True)
        print("="*80, flush=True)
        print(f"Train - Loss: {train_loss:.4f} | Acc: {train_accuracy:.4f} | AUC: {train_auc:.4f}", flush=True)
        print(f"Val   - Loss: {val_loss:.4f} | Acc: {val_accuracy:.4f} | AUC: {val_auc:.4f}", flush=True)
        print("="*80 + "\n", flush=True)

        epoch_metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_accuracy,
            'train_auc': train_auc,
            'val_loss': val_loss,
            'val_acc': val_accuracy,
            'val_auc': val_auc,
            'epoch_time_s': epoch_time,
        })
        # Persist after every epoch so progress survives a kernel timeout/crash.
        pd.DataFrame(epoch_metrics).to_csv('Full_Model_epoch_metrics.csv', index=False)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # ============================
        # PLOT VISUALIZATIONS
        # ============================
        print("Training Confusion Matrix:")
        plot_confusion_matrix(
            train_targets, train_preds,
            epoch=epoch+1, phase="Training",
            class_names=class_names
        )

        print("Validation Confusion Matrix:")
        plot_confusion_matrix(
            val_targets, val_preds,
            epoch=epoch+1, phase="Validation",
            class_names=class_names
        )

        if len(unique_train_classes) == num_classes:
            print("Training ROC Curves:")
            plot_epoch_multiclass_roc(
                train_targets, train_probs,
                num_classes=num_classes,
                epoch=epoch + 1,
                phase="Training",
                class_names=class_names
            )

        if len(unique_val_classes) == num_classes:
            print("Validation ROC Curves:")
            plot_epoch_multiclass_roc(
                val_targets, val_probs,
                num_classes=num_classes,
                epoch=epoch + 1,
                phase="Validation",
                class_names=class_names
            )

    torch.save((model.module if isinstance(model, DataParallel) else model).state_dict(),
               'chunkybert_final_epoch.pt')
    print("Saved final-epoch weights to chunkybert_final_epoch.pt", flush=True)

    print(f"\n{'='*80}")
    print("BEST RESULTS")
    print(f"{'='*80}")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Val AUC: {best_val_auc:.4f}")
    print(f"{'='*80}\n")

    metrics_df = pd.DataFrame(epoch_metrics)

    return {
        'best_val_accuracy': best_val_acc,
        'best_val_loss': best_val_loss,
        'best_val_auc': best_val_auc,
        'epoch_metrics': metrics_df
    }


# ============================================================================
# ABLATION STUDY RUNNER
# ============================================================================
def run(
    train_dataloader,
    val_dataloader,
    tokenizer,
    device,
    num_classes,
    num_epochs=3,
    learning_rate=2e-5,
):
    """Run ablation study and return results DataFrame"""

    model_variants = {
        "Full Model": ChunkedBERTClassifier,
    }

    results = []

    for model_name, ModelClass in model_variants.items():
        print("\n" + "=" * 80)
        print(f"Training: {model_name}")
        print("=" * 80)

        model = ModelClass(num_classes=num_classes)

        metrics = train(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            device=device,
            num_classes=num_classes,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            use_multi_gpu=True
        )

        results.append({
            'Model': model_name,
            'Best Val Accuracy': metrics['best_val_accuracy'],
            'Best Val Loss': metrics['best_val_loss'],
            'Best Val AUC': metrics['best_val_auc']
        })

        metrics['epoch_metrics'].to_csv(f'{model_name.replace(" ", "_")}_epoch_metrics.csv', index=False)
        print(f"\nEpoch-wise metrics saved for {model_name}")

        del model
        torch.cuda.empty_cache()

    return pd.DataFrame(results)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    # ========================================================================
    # CONFIGURATION  (paper Table 4 defaults)
    # ========================================================================
    BATCH_SIZE = 16
    NUM_EPOCHS = 5
    LEARNING_RATE = 2e-5
    NUM_CLASSES = 3
    RANDOM_SEED = 42
    USE_MULTI_GPU = True

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    csv_path = None
    for root, dirs, files in os.walk("/kaggle/input"):
        print("WALK:", root, dirs, files)
        for fn in files:
            if fn == "out.csv":
                csv_path = os.path.join(root, fn)
    print("Resolved csv_path:", csv_path)
    df = pd.read_csv(csv_path)
    df = df.reset_index(drop=True)

    texts = df["content"].tolist()
    labels = df["bias"].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"\nDataset sizes:")
    print(f"Training: {len(train_texts)}")
    print(f"Validation: {len(val_texts)}")
    print(f"Class distribution - Train: {np.bincount(train_labels)}")
    print(f"Class distribution - Val: {np.bincount(val_labels)}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    results_df = run(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
        device=device,
        num_classes=NUM_CLASSES,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE
    )

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))

    results_df.to_csv('training_results.csv', index=False)
    print("\nResults saved to: training_results.csv")


if __name__ == "__main__":
    main()
