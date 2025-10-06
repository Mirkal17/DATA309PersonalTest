# Agents/train_sentiment_hf.py
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)
import torch

#replace these if want other mapping
LABEL_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def map_rating_to_sentiment(rating):
    """
    Map customer satisfaction rating (assumed numeric, 0-5) to sentiment labels using these thresholds:
      - Negative: rating < 2.5
      - Neutral : 2.5 <= rating < 3.5
      - Positive: rating >= 3.5

    Non-numeric or missing values default to 'Neutral' (safe fallback).
    """
    try:
        r = float(r)
    except Exception:
        return "Neutral"
    if r < 2.5:
        return "Negative"
    if r < 3.5:  # implies r >= 2.5 and r < 3.5
        return "Neutral"
    return "Positive"

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    report = classification_report(labels, preds, target_names=["Negative","Neutral","Positive"], zero_division=0, output_dict=True)
    # Return simple metrics for Trainer logging
    return {
        "accuracy": report["accuracy"] if "accuracy" in report else np.mean(np.array(list(report.values()))),
        "f1_macro": report.get("macro avg", {}).get("f1-score", 0.0)
    }

def prepare_dataframe(args):
    # load cleaned csv if available, otherwise call project's prepare function
    if os.path.exists(args.cleaned_csv):
        print("Loading cleaned CSV:", args.cleaned_csv)
        df = pd.read_csv(args.cleaned_csv)
    else:
        # try to import your project prepare function
        try:
            from Preprocessing.Prepare import load_and_prepare
        except Exception as e:
            raise RuntimeError("No cleaned CSV and could not import Preprocessing.Prepare.load_and_prepare") from e
        df = load_and_prepare(args.raw_csv)
        df.to_csv(args.cleaned_csv, index=False)
        print("Saved cleaned CSV to:", args.cleaned_csv)

    # map satisfaction to sentiment labels
    if 'Customer Satisfaction Rating' not in df.columns:
        raise KeyError("Dataset requires 'Customer Satisfaction Rating' to create training labels.")
    df['sentiment_label'] = df['Customer Satisfaction Rating'].apply(map_rating_to_sentiment)
    df = df[['Cleaned Description', 'sentiment_label']].rename(columns={'Cleaned Description':'text'})
    df = df[df['text'].notna() & (df['text'].str.strip() != '')].reset_index(drop=True)
    df['label'] = df['sentiment_label'].map(LABEL_MAP)
    return df

def balance_oversample(train_df, label_col="label"):
    # simple oversampling of minority classes to balance training
    counts = train_df[label_col].value_counts().to_dict()
    max_count = max(counts.values())
    frames = []
    for label, cnt in counts.items():
        df_label = train_df[train_df[label_col]==label]
        if cnt < max_count:
            df_label_up = resample(df_label, replace=True, n_samples=max_count, random_state=42)
            frames.append(df_label_up)
        else:
            frames.append(df_label)
    return pd.concat(frames).sample(frac=1, random_state=42).reset_index(drop=True)

def main(args):
    set_seed(42)
    df = prepare_dataframe(args)

    # split
    train_df, test_df = train_test_split(df, stratify=df['label'], test_size=args.test_size, random_state=42)
    train_df, val_df = train_test_split(train_df, stratify=train_df['label'], test_size=args.val_size, random_state=42)

    print("Sizes => train:", len(train_df), "val:", len(val_df), "test:", len(test_df))

    if args.oversample:
        train_df = balance_oversample(train_df, label_col="label")
        print("After oversampling, train size:", len(train_df))

    # Convert to HF Dataset
    train_ds = Dataset.from_pandas(train_df[['text','label']])
    val_ds = Dataset.from_pandas(val_df[['text','label']])
    test_ds = Dataset.from_pandas(test_df[['text','label']])

    # tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3)

    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    columns = ['input_ids','attention_mask','label']
    train_ds.set_format(type='torch', columns=columns)
    val_ds.set_format(type='torch', columns=columns)
    test_ds.set_format(type='torch', columns=columns)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=3,
        fp16=args.fp16 and torch.cuda.is_available(),
        logging_dir=os.path.join(args.output_dir, "logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()

    print("Evaluating on test set...")
    test_metrics = trainer.predict(test_ds)
    print("Test metrics:", test_metrics.metrics)

    # save model/tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved model to:", args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", type=str, default="Data/customer_support_tickets.csv")
    parser.add_argument("--cleaned_csv", type=str, default="Data/customer_support_tickets_cleaned.csv")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="models/sentiment_hf")
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--oversample", action="store_true", help="Simple oversampling to balance classes")
    args = parser.parse_args()
    main(args)
