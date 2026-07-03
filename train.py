"""
Fine-tunes a multilingual transformer (MuRIL) on Tamil sentiment data.
Run this once to produce a saved model in model/saved_model/
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch
from torch.utils.data import Dataset

MODEL_NAME = "google/muril-base-cased"
DATA_PATH = "data/sample_tamil_sentiment.csv"
OUTPUT_DIR = "model/saved_model"

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_len
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def main():
    df = pd.read_csv(DATA_PATH)
    df["label_id"] = df["label"].map(LABEL2ID)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(),
        df["label_id"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df["label_id"],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

    training_args = TrainingArguments(
        output_dir="model/checkpoints",
        num_train_epochs=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="model/logs",
        logging_steps=5,
        save_safetensors=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
