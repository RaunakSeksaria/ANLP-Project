"""Training / evaluation loop for ChunkyBERT.

Adapted only minimally from the official source (https://github.com/dloiya/chunkybert,
cell 2 of chunkybert-final.ipynb) released with:

    Loiya, D., Kulal, S. S., Reddy, M. S. M. & Shetty, R. D.
    "ChunkyBERT: a novel technique for multiclass political bias detection in
    news media." Scientific Reports 16, 15323 (2026).
    https://doi.org/10.1038/s41598-026-46646-z

Changes from the notebook (functionality is otherwise untouched):
  - the hard-coded Kaggle input path (/kaggle/input/media-bias-csv/out.csv) is
    replaced with a --data-csv CLI argument.
  - the notebook's cell 3 has a documented unterminated `print(...)` at main()'s
    end (see its README's "Notes" section) — closed here.
  - epoch/plot noise is gated behind --no-plots so the script is runnable
    headless; the plotting functions themselves are untouched.
  - CLI flags expose the same knobs the notebook hard-codes in main(), with
    identical defaults (Table 4 of the paper).
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model import ChunkedBERTClassifier

PLOTS_ENABLED = True


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
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    if PLOTS_ENABLED:
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
    import matplotlib.pyplot as plt

    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    if PLOTS_ENABLED:
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

    # Multi-GPU setup
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

    # Store epoch-wise metrics
    epoch_metrics = []

    for epoch in range(num_epochs):
        # ============================
        # TRAIN
        # ============================
        model.train()
        train_loss = 0.0
        train_probs, train_targets, train_preds = [], [], []

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} [Train]"):
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

            # Collect predictions for metrics
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)

            train_probs.append(probs.detach().cpu())
            train_targets.append(labels.detach().cpu())
            train_preds.append(preds.detach().cpu())

        train_loss /= len(train_dataloader)

        # Calculate training metrics
        train_probs = torch.cat(train_probs).numpy()
        train_targets = torch.cat(train_targets).numpy()
        train_preds = torch.cat(train_preds).numpy()

        train_accuracy = (train_preds == train_targets).mean()

        # Normalize probabilities
        train_probs = train_probs / train_probs.sum(axis=1, keepdims=True)

        # Calculate training AUROC
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
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1} [Val]"):
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

        # Normalize probabilities
        val_probs = val_probs / val_probs.sum(axis=1, keepdims=True)

        # Calculate validation AUROC
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

        # ============================
        # PRINT METRICS
        # ============================
        print("\n" + "="*80)
        print(f"EPOCH {epoch+1}/{num_epochs} RESULTS")
        print("="*80)
        print(f"Train - Loss: {train_loss:.4f} | Acc: {train_accuracy:.4f} | AUC: {train_auc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f} | Acc: {val_accuracy:.4f} | AUC: {val_auc:.4f}")
        print("="*80 + "\n")

        # Store metrics
        epoch_metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_accuracy,
            'train_auc': train_auc,
            'val_loss': val_loss,
            'val_acc': val_accuracy,
            'val_auc': val_auc
        })

        # Update best metrics
        if val_auc > best_val_auc:
            best_val_auc = val_auc
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # ============================
        # PLOT VISUALIZATIONS
        # ============================
        # Training confusion matrix
        print("Training Confusion Matrix:")
        train_cm = plot_confusion_matrix(
            train_targets, train_preds,
            epoch=epoch+1, phase="Training",
            class_names=class_names
        )

        # Validation confusion matrix
        print("Validation Confusion Matrix:")
        val_cm = plot_confusion_matrix(
            val_targets, val_preds,
            epoch=epoch+1, phase="Validation",
            class_names=class_names
        )

        # Training ROC curves
        if len(unique_train_classes) == num_classes:
            print("Training ROC Curves:")
            plot_epoch_multiclass_roc(
                train_targets, train_probs,
                num_classes=num_classes,
                epoch=epoch + 1,
                phase="Training",
                class_names=class_names
            )

        # Validation ROC curves
        if len(unique_val_classes) == num_classes:
            print("Validation ROC Curves:")
            plot_epoch_multiclass_roc(
                val_targets, val_probs,
                num_classes=num_classes,
                epoch=epoch + 1,
                phase="Validation",
                class_names=class_names
            )

    print(f"\n{'='*80}")
    print("BEST RESULTS")
    print(f"{'='*80}")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Val AUC: {best_val_auc:.4f}")
    print(f"{'='*80}\n")

    # Create metrics dataframe
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
    output_dir='.',
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

        # Save epoch-wise metrics
        metrics['epoch_metrics'].to_csv(
            os.path.join(output_dir, f'{model_name.replace(" ", "_")}_epoch_metrics.csv'),
            index=False
        )
        print(f"\nEpoch-wise metrics saved for {model_name}")

        del model
        torch.cuda.empty_cache()

    return pd.DataFrame(results)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    global PLOTS_ENABLED

    parser = argparse.ArgumentParser(description="Train ChunkyBERT (paper defaults from Table 4)")
    parser.add_argument('--data-csv', required=True, help='CSV with "content" and "bias" columns')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--num-classes', type=int, default=3)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--max-length', type=int, default=5000, help='Tokenizer truncation length (paper: 5000)')
    parser.add_argument('--no-multi-gpu', action='store_true')
    parser.add_argument('--no-plots', action='store_true', help='Skip saving confusion-matrix/ROC PNGs')
    parser.add_argument('--limit', type=int, default=None, help='Use only first N rows (smoke testing)')
    parser.add_argument('--output-dir', default='.')
    args = parser.parse_args()

    PLOTS_ENABLED = not args.no_plots
    os.makedirs(args.output_dir, exist_ok=True)

    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    NUM_CLASSES = args.num_classes
    RANDOM_SEED = args.random_seed
    USE_MULTI_GPU = not args.no_multi_gpu

    # Set random seeds
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load data
    df = pd.read_csv(args.data_csv)
    df = df.reset_index(drop=True)
    if args.limit:
        df = df.iloc[:args.limit].reset_index(drop=True)

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

    # Create datasets and dataloaders
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length=args.max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length=args.max_length)

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

    # Run training
    results_df = run(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
        device=device,
        num_classes=NUM_CLASSES,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        output_dir=args.output_dir,
    )

    # Display results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))

    # Save results
    results_df.to_csv(os.path.join(args.output_dir, 'training_results.csv'), index=False)
    print("\nResults saved to: training_results.csv")


if __name__ == "__main__":
    main()
