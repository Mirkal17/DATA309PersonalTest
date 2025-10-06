import pandas as pd
import kagglehub

# Download dataset
path = kagglehub.dataset_download("suraj520/customer-support-ticket-dataset")
df = pd.read_csv(path + "/customer_support_tickets.csv")

print("Dataset shape:", df.shape)

# Check for duplicate rows
print("\nNumber of duplicate rows in the dataset:")
print(df.duplicated().sum())

# Check if there are any duplicates
if df.duplicated().sum() > 0:
    print("There ARE duplicate rows!")
    print("First few duplicate rows:")
    print(df[df.duplicated()].head())
else:
    print("There are NO duplicate rows in this dataset.")
