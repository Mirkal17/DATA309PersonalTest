from Preprocessing.Prepare import load_and_prepare
from Preprocessing.EDA import plot_distribution
from Agents.Classifier.Classifier_agent import classify_ticket

# Step 1: Load and clean dataset
df = load_and_prepare("Data/customer_support_tickets.csv")

df.to_csv("Data/customer_support_tickets_cleaned.csv", index=False)
print("✅ Cleaned dataset saved to Data/customer_support_tickets_cleaned.csv")

# Step 2: Quick EDA
plot_distribution(df, 'Ticket Type', 'Distribution of Ticket Type', 'ticket_type_distribution.png')

# # Step 3: Test classifier on first 3 tickets
# for i in range(3):
#     ticket = df.iloc[i]['Cleaned Description']
#     actual = df.iloc[i]['Ticket Type']
#     predicted = classify_ticket(ticket)
#     print(f"\nTicket {i+1}")
#     print(f"Actual: {actual}")
#     print(f"Predicted: {predicted}")
#     print(f"Text: {ticket[:100]}...")
