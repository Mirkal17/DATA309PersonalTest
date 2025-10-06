# import os
# import pandas as pd
# from Agents.Classifier.classifier_agent import classify_tickets
# from Preprocessing.Prepare import load_and_prepare

# def main():
#     df = pd.read_csv("Data/customer_support_tickets_cleaned.csv")

#     result = []

#     for i in range(50):
#         text = df.iloc[i]["Ticket Description_cleaned"]
#         prediction = classify_tickets(text)

#         result.append({
#             "Ticket ID": df.iloc[i]["Ticket ID"],
#             "Text": text,
#             "Predicted Category": prediction
#         })

#     output_path = os.path.join("Data", "classification_results.csv")
#     result_df = pd.DataFrame(result)
#     result_df.to_csv(output_path, index=False)

#     print(f"Results saved to {output_path}")

# if __name__ == "__main__":
#     main()



# # import subprocess
# # import pandas as pd

# # #Step 1: Run the R script
# # print("Running R classifier...")
# # subprocess.run(["Rscript", "Agents/Classifier/classifier_agent.R"], check=True)

# # #Step 2: Load the output results
# # classifier_results = pd.read_csv("Data/classifier_results_r.csv")

# # print("Classifier results (from R):")
# # print(classifier_results.head())


from Agents.Classifier.classifier_agent import load_and_balance_data, train_classifier, save_model, load_model, classify_tickets

# Step 1: Load balanced dataset
df = load_and_balance_data("Data/customer_support_tickets_cleaned.csv")

# Step 2: Train classifier
clf, vectorizer = train_classifier(df)

# Step 3: Save model
save_model(clf, vectorizer)

# Step 4: Load model + classify whole dataset
saved = load_model()
classify_tickets(df, saved["model"], saved["vectorizer"])
