ChunkyBERT (Loiya et al., Sci. Rep. 2026) trained on Article-Bias-Prediction (Baly et al.), 3-class Left/Centre/Right.
Final-epoch (5) weights, paper hyperparameters. Val acc 85.7%, AUC 0.962.
Load: m = ChunkedBERTClassifier(num_classes=3); m.load_state_dict(torch.load("chunkybert_final_epoch.pt")).
Forward needs a BertTokenizer('bert-base-uncased'): m(input_ids, attention_mask, tokenizer). Labels: 0 Left, 1 Centre, 2 Right.
