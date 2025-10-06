# Key hyperparams: model='distilbert-base-uncased', max_len=128, lr=2e-5, epochs=3, batch_size=16
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

def preprocess(examples): return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
# prepare HF Dataset objects, map preprocess, set format

training_args = TrainingArguments(
    output_dir='out',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
                  tokenizer=tokenizer)
trainer.train()
# Evaluate with trainer.predict on test set
