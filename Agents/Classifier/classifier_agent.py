import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer #Random forest with bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils import resample

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re

# Make sure nltk resources are available
nltk.download('stopwords')

# ----------------------------
# Load data
# ----------------------------
import kagglehub

# Download latest version
import pandas as pd
import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("parthpatil256/it-support-ticket-data")
print("Path to dataset files:", path)

# Find the CSV file inside the downloaded path
for f in os.listdir(path):
    if f.endswith(".csv"):
        csv_file = os.path.join(path, f)
        break

# Load dataset
data = pd.read_csv(csv_file)
print(data.head())


# Top departments by frequency
top_types = (
    data.dropna(subset=["Department", "Body"])
        .groupby("Department")
        .size()
        .sort_values(ascending=False)
        .head(5)
)
print(top_types)

top_departments = [
    "Technical Support",
    "Product Support",
    "Customer Service",
    "IT Support",
    "Billing and Payments"
]

# ----------------------------
# Balance dataset
# ----------------------------
n_per_group = 3017
balanced_subset = (
    data.dropna(subset=["Department", "Body"])
        .query("Department in @top_departments")
        .groupby("Department", group_keys=False)
        .apply(lambda x: resample(x, n_samples=n_per_group, random_state=0))
        .reset_index(drop=True)
)

print(balanced_subset["Department"].value_counts())

# ----------------------------
# Preprocessing & tokenization
# ----------------------------
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)  # remove numbers & symbols
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

balanced_subset["CleanBody"] = balanced_subset["Body"].astype(str).apply(clean_text)

# ----------------------------
# Create DTM (document-term matrix)
# ----------------------------

#Random Forest method with bag of words
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(balanced_subset["CleanBody"])

vectorizer = CountVectorizer(
    max_features=10000,      # limit vocab size (tune as needed)
    ngram_range=(1,2),       # unigrams + bigrams
    min_df=5,                # ignore very rare words
    max_df=0.8              # ignore very common words
)

X = vectorizer.fit_transform(balanced_subset["CleanBody"])
y = balanced_subset["Department"]

# ----------------------------
# Remove near-zero variance features
# ----------------------------
nzv = VarianceThreshold(threshold=0.001)
X_reduced = nzv.fit_transform(X)

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=0, stratify=y
)

# ----------------------------
# Random Forest model
# ----------------------------
rf = RandomForestClassifier(
    n_estimators=1000,
    max_features=int(np.sqrt(X_train.shape[1])),
    random_state=0,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ----------------------------
# Evaluation
# ----------------------------
print("Random Forest")
pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print("Classification Report:\n", classification_report(y_test, pred))
