import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import kagglehub
from Agents.Classifier.Classifier_agent import classify_ticket
import pandas as pd

path = kagglehub.dataset_download("suraj520/customer-support-ticket-dataset")
df = pd.read_csv(path + "/customer_support_tickets.csv")

#Use only the cleaned description column 
tickets = df['Ticket Description'].dropna().tolist()

#Limit how many to classify
LIMIT = 50
tickets_to_classify = tickets[:LIMIT]

print(f"Classifying {len(tickets_to_classify)} tickets (limit = {LIMIT})...\n")

for i, ticket in enumerate(tickets_to_classify, start=1):
    print(f"Ticket {i}: {ticket}")
    predicted_category = classify_ticket(ticket)
    print(f"Predicted Category: {predicted_category}\n")
