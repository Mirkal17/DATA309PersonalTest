import os
import shutil
import pandas as pd
import kagglehub

DATA_DIR = os.path.join(os.path.dirname(__file__), "")
DATA_FILE = os.path.join(DATA_DIR, "customer_support_tickets.csv")

def get_dataset():
    """Load dataset from data/ or download from KaggleHub if missing."""
    if not os.path.exists(DATA_FILE):
        print("Dataset not found locally. Downloading from KaggleHub...")
        path = kagglehub.dataset_download("suraj520/customer-support-ticket-dataset")
        kaggle_file = os.path.join(path, "customer_support_tickets.csv")
        os.makedirs(DATA_DIR, exist_ok=True)
        shutil.copy(kaggle_file, DATA_FILE)  # <-- changed from os.rename to shutil.copy
        print(f"Dataset saved to {DATA_FILE}")
    else:
        print(f"Loading dataset from {DATA_FILE}")

    return pd.read_csv(DATA_FILE)
