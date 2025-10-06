import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# 3-class sentiment labels (negative, neutral, positive)
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

def load_model(model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
    """Load pre-trained Roberta sentiment analysis model (3 classes: Neg/Neu/Pos)"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return tokenizer, model, device


def predict_texts(texts, tokenizer, model, device, max_length=128, batch_size=16):
    """Predict sentiment for a list of texts"""
    results = []
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.cpu().numpy()
            probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
            preds = np.argmax(logits, axis=1)

        for j, t in enumerate(batch):
            results.append({
                "text": t,
                "label": LABEL_MAP[int(preds[j])],
                "probs": probs[j].tolist()
            })
    return results
