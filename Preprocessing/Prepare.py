# Preprocessing/Prepare.py
import pandas as pd
from Preprocessing.Clean import clean_text, preprocess_text

def load_and_prepare(path):
    """Load dataset and return DataFrame with both LLM and ML-friendly text."""
    df = pd.read_csv(path)

    # Fill missing values
    df['Resolution'] = df['Resolution'].fillna('Not Resolved')
    df['Customer Satisfaction Rating'] = df['Customer Satisfaction Rating'].fillna(
        df['Customer Satisfaction Rating'].mean()
    )

    # --- Replace {product_purchased} placeholder with actual product name ---
    if 'Product Purchased' in df.columns and 'Ticket Description' in df.columns:
        df['Ticket Description'] = df.apply(
            lambda row: str(row['Ticket Description']).replace("{product_purchased}", str(row['Product Purchased'])),
            axis=1
        )
        # Drop the column if no longer needed
        df = df.drop(columns=['Product Purchased'])

    # --- Light cleaning (for LLMs) ---
    df['Ticket Subject_cleaned'] = df['Ticket Subject'].apply(clean_text)
    df['Ticket Description_cleaned'] = df['Ticket Description'].apply(clean_text)
    df['Resolution_cleaned'] = df['Resolution'].apply(clean_text)

    # Combine into a single LLM-friendly column
    df['Cleaned Description'] = df['Ticket Description_cleaned'].fillna('').str.strip()
    df = df[df['Cleaned Description'] != '']  # remove empties

    # --- Heavy preprocessing (for ML baselines, optional) ---
    df['Ticket Subject_processed'] = df['Ticket Subject_cleaned'].apply(preprocess_text)
    df['Ticket Description_processed'] = df['Ticket Description_cleaned'].apply(preprocess_text)
    df['Resolution_processed'] = df['Resolution_cleaned'].apply(preprocess_text)

    # Combined processed version
    df['Processed Description'] = df['Ticket Description_processed'].fillna('').str.strip()

    return df

