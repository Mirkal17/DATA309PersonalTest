import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Clean raw text: lowercase, remove links, punctuation, etc., but keep important info."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)              # Remove emails
    text = re.sub(r'[^\w\s]', '', text)              # Remove punctuation
    return ' '.join(text.split()).strip()

def preprocess_text(text):
    """Tokenize, remove minimal noise, keep important context."""
    if not text:
        return ""

    tokens = word_tokenize(text)

    # Keep numbers & short tokens that might be relevant (e.g., codes, IDs)
    tokens = [t for t in tokens if t.isalnum()]

    # Remove only common stopwords, keep negations like 'not' and 'no'
    filtered_stopwords = {w for w in stop_words if w not in {"not", "no"}}
    tokens = [t for t in tokens if t.lower() not in filtered_stopwords]

    # Lemmatize to base form
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return ' '.join(tokens)
