import numpy as np
import matplotlib.pyplot as plt # for visualization
import pandas as pd # for data manipulation and processing
import seaborn as sns # for statistical data visualization
from sklearn.model_selection import train_test_split # for splitting the dataset
import nltk # for natural language processing tasks
from nltk.corpus import stopwords # for removing common words in text
from nltk.tokenize import word_tokenize # for tokenizing text
import re # for regular expressions
from nltk.stem import WordNetLemmatizer # for lemmatizing words
import os
from openai import OpenAI

import kagglehub

# Download NLTK data
print("Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
print("NLTK data downloaded successfully!")

# Download latest version
path = kagglehub.dataset_download("suraj520/customer-support-ticket-dataset")

print("Path to dataset files:", path)
# Load the dataset
df = pd.read_csv(path + "/customer_support_tickets.csv")

#Explore the dataset
print("Dataset shape:", df.shape)   

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Display basic information about the dataset
print("\nDataset Information:")
print(df.info())
# Display basic statistics of the dataset
print("\nDataset Statistics:")
print(df.describe())

# Check for duplicate rows
print("\nNumber of duplicate rows in the dataset:")
duplicate_count = df.duplicated().sum()
print(duplicate_count)
if duplicate_count > 0:
    print(f"Found {duplicate_count} duplicate rows!")
else:
    print("No duplicate rows found - dataset is clean!")

# Check for missing values
print("\nMissing Values in the Dataset:")
print(df.isnull().sum())

# Check what columns are available
print("\nColumn names in the dataset:")
print(df.columns.tolist())

# Visualize the distribution of ticket categories 
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Ticket Type', order=df['Ticket Type'].value_counts().index)
plt.title('Distribution of Ticket Type')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent label cutoff
# Save the plot as an image file
plt.show()  # Display the plot  
plt.savefig('ticket_type_distribution.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'ticket_type_distribution.png' in the current directory")


# Visualize the distribution of ticket priority.
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Ticket Priority', order=df['Ticket Priority'].value_counts().index)
plt.title('Distribution of Ticket Priority')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent label cutoff
plt.show()  # Display the plot
# Save the plot as an image file
plt.savefig('ticket_priority_distribution.png', dpi=300, bbox_inches='tight')   
print("\nPlot saved as 'ticket_priority_distribution.png' in the current directory")

# Visualize the distribution of ticket status.
plt.figure(figsize=(10, 6))             
sns.countplot(data=df, x='Ticket Status', order=df['Ticket Status'].value_counts().index)
plt.title('Distribution of Ticket Status')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent label cutoff
plt.show()  # Display the plot
# Save the plot as an image file
plt.savefig('ticket_status_distribution.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'ticket_status_distribution.png' in the current directory")

# Visualize the distribution of ticket channel.
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Ticket Channel', order=df['Ticket Channel'].value_counts().index)
plt.title('Distribution of Ticket Channel')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent label cutoff
plt.show()  # Display the plot
# Save the plot as an image file
plt.savefig('ticket_channel_distribution.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'ticket_channel_distribution.png' in the current directory")

# Visualize the distribution of customer satisfaction.
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Customer Satisfaction Rating', order=df['Customer Satisfaction Rating'].value_counts().index)
plt.title('Distribution of Customer Satisfaction Rating')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent label cutoff
plt.show()  # Display the plot
# Save the plot as an image file
plt.savefig('customer_satisfaction_distribution.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'customer_satisfaction_distribution.png' in the current directory")  

# Show value counts for Ticket Type
print("\nTicket Type Value Counts:")
print(df["Ticket Type"].value_counts())

# Show value counts for Ticket Priority
print("\nTicket Priority Value Counts:")
print(df["Ticket Priority"].value_counts())


#Cleaning and Pre-processing the dataset
print("\nCleaning and Pre-processing the dataset...")

#Fill missing value in Resolution with Not Resolved
df['Resolution'] = df['Resolution'].fillna('Not Resolved')

#Customer satisfaction rating imputation
df['Customer Satisfaction Rating'] = df['Customer Satisfaction Rating'].fillna(df['Customer Satisfaction Rating'].mean())

#Convert Date of Purchase to datetime format
df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'], errors='coerce')

#replace product_purchased placeholder in Ticket Description

df['Ticket Description'] = df.apply(lambda row: row['Ticket Description'].replace('{product_purchased}', row['Product Purchased']), axis=1)

# Define a function to clean text data
def clean_text(text):
    """Function to clean text data following NLP best practices."""
    import re
    
    if pd.isna(text) or text is None:
        return ""
    
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase for uniformity
    text = text.lower()
    
    # Remove URLs and email addresses
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'\S+@\S+', '', text)

    # Remove newlines and carriage returns
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Remove numbers (if not relevant for analysis)
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation and special characters - keep only words and spaces
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra spaces and tabs
    text = ' '.join(text.split())
    
    # Remove leading/trailing spaces
    text = text.strip()
    
    return text

# Apply text cleaning to relevant text columns
print("\nCleaning text columns...")
text_columns = ['Ticket Subject', 'Ticket Description', 'Resolution']

for col in text_columns:
    if col in df.columns:
        print(f"Cleaning {col}...")
        df[f'{col}_cleaned'] = df[col].apply(clean_text)

stopwords = set(stopwords.words('english'))
stemmer = WordNetLemmatizer()

def preprocess_text(text):
    """Enhanced function to preprocess text data following NLP best practices."""
    if pd.isna(text) or text is None:
        return ""
    
    # Handle short texts - if text is too short after cleaning, return empty
    if len(text.strip()) < 3:
        return ""
    
    # Tokenization: Split into words using NLTK
    tokens = word_tokenize(text)
    
    # Stopword removal & filtering:
    # - Remove common words like "the", "is", "and"
    # - Keep only alphabetic tokens
    # - Filter out very short tokens (less than 2 characters)
    tokens = [word for word in tokens 
              if word.isalpha() 
              and word not in stopwords 
              and len(word) > 1]
    
    # Lemmatization: Reduce words to base form for better context preservation
    # (e.g., "running" -> "run", "better" -> "good")
    tokens = [stemmer.lemmatize(word) for word in tokens]
    
    # Handle edge case: if no meaningful tokens remain after processing
    if not tokens:
        return ""
    
    return ' '.join(tokens) 

# Apply text preprocessing to cleaned text columns
print("\nPreprocessing cleaned text columns...") 
for col in text_columns:
    if f'{col}_cleaned' in df.columns:
        print(f"Preprocessing {col}_cleaned...")
        df[f'{col}_processed'] = df[f'{col}_cleaned'].apply(preprocess_text) 

# Display the first few rows of the updated dataset
print("\nFirst few rows of the updated dataset:")
print(df.head())

# Show detailed comparison of text processing stages
print("\n" + "="*80)
print("TEXT PROCESSING COMPARISON EXAMPLES")
print("="*80)

for i in range(3):
    print(f"\n--- EXAMPLE {i+1} ---")
    print("ORIGINAL:")
    print(f"  Subject: {df['Ticket Subject'].iloc[i]}")
    print(f"  Description: {df['Ticket Description'].iloc[i][:100]}...")
    
    print("CLEANED:")
    print(f"  Subject: {df['Ticket Subject_cleaned'].iloc[i]}")
    print(f"  Description: {df['Ticket Description_cleaned'].iloc[i][:100]}...")
    
    print("PROCESSED (Tokenized, Stopwords Removed, Lemmatized):")
    print(f"  Subject: {df['Ticket Subject_processed'].iloc[i]}")
    print(f"  Description: {df['Ticket Description_processed'].iloc[i][:100]}...")

print("\n" + "="*80)

# Display the updated dataset information
print("\nUpdated Dataset Information:")
print(df.info())

# Create a single cleaned description column that combines processed text
print("\nCombining cleaned description columns...")
df['Cleaned Description'] = df['Ticket Description_processed'].fillna('')

# Drop rows with empty descriptions after processing
df = df.dropna(subset=['Cleaned Description'])
df = df[df['Cleaned Description'].str.strip() != '']  # Remove empty strings

print(f"Dataset shape after removing empty descriptions: {df.shape}")

# Convert to list for batch processing
texts = df['Cleaned Description'].tolist()

#Preparing the data for LLM input
print("\nPreparing data for LLM input...")


# Select relevant columns
llm_input_data = df[['Ticket ID', 'Cleaned Description', 'Ticket Type']].copy()

# Convert to a list of dictionaries for LLM input
llm_input_list = llm_input_data.to_dict(orient='records')

# Display the first few entries of the prepared data
print("Prepared data for LLM (first 5 entries):")
for i in range(min(5, len(llm_input_list))):
    print(llm_input_list[i])

# OpenAI Integration Setup
print("\n" + "="*80)
print("OPENAI INTEGRATION")
print("="*80)

import os
from openai import OpenAI

# Set API key directly in code (for development convenience)
os.environ['OPENAI_API_KEY'] = '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'

# Check if API key is set
def check_openai_setup():
    """Check if OpenAI API key is properly configured"""
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        print(f" OpenAI API key found (ends with: ...{api_key[-4:]})")
        return True
    else:

        return False

# Check setup
if check_openai_setup():
    # Initialize OpenAI client
    client = OpenAI()
    
    # Test function for ticket classification
    def classify_ticket_with_openai(ticket_description):
        """Classify a customer support ticket using OpenAI's API"""
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using cheaper model for testing
                messages=[
                    {"role": "system", "content": "You are a customer support ticket classifier. Classify tickets into these categories: Technical Issue, Billing Problem, Product Question, Complaint, Feature Request. Respond with just the category name."},
                    {"role": "user", "content": f"Classify this ticket: {ticket_description}"}
                ],
                max_tokens=20,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Test with a sample ticket
    print("\n TESTING OPENAI CLASSIFICATION:")
    sample_ticket = df['Ticket Description'].iloc[0]
    print(f"Sample ticket: {sample_ticket[:150]}...")
    
    try:
        classification_result = classify_ticket_with_openai(sample_ticket)
        print(f" OpenAI Classification: {classification_result}")
        
        # Test with multiple tickets
        print(f"\n CLASSIFYING FIRST 3 TICKETS:")
        for i in range(3):
            ticket = df['Ticket Description'].iloc[i]
            actual_type = df['Ticket Type'].iloc[i]
            predicted_type = classify_ticket_with_openai(ticket)
            print(f"Ticket {i+1}:")
            print(f"  Actual: {actual_type}")
            print(f"  Predicted: {predicted_type}")
            print(f"  Description: {ticket[:100]}...")
            print()
            
    except Exception as e:
        print(f" Error during classification: {e}")
        
else:
    print("\n OpenAI integration skipped - API key not configured")
# Display the first few rows of the final dataset

print("\nFinal Dataset with Processed Text:")
print(df.head())




# Show before & after of a cleaned ticket
print("\nBefore & After Cleaning Example:")

# Pick a random row (change to iloc[0] if you want first row)
example_row = df.sample(1).iloc[0]

print("\n--- ORIGINAL ---")
print("Subject:", example_row['Ticket Subject'])
print("Description:", example_row['Ticket Description'])

print("\n--- CLEANED ---")
print("Subject (Processed):", example_row['Ticket Subject_processed'])
print("Description (Processed):", example_row['Ticket Description_processed'])

print("\n--- FINAL LLM INPUT ---")
print(example_row['Cleaned Description'])
