# run_escalation.py
import os
import pandas as pd
from Agents.Escalation.Escalation_agent import should_escalate

def main():
    classifier_df = pd.read_csv("Data/classification_results.csv")
    sentiment_df = pd.read_csv("Data/sentiment_results.csv")

    #Merge on Ticket ID
    merged_df = pd.merge(classifier_df, sentiment_df, on=["Ticket ID", "Text"], how="inner")

    results = []
    for _, row in merged_df.iterrows():
        decision, reason = should_escalate(row)
        results.append({
            "Ticket ID": row["Ticket ID"],
            "Text": row["Text"],
            "Predicted Category": row["Predicted Category"],
            "Sentiment": row["Sentiment"],
            "Confidence": row["Confidence"],
            "Escalation Decision": decision,
            "Reason": reason
        })

    # Save results
    output_path = os.path.join("Data", "escalation_results.csv")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    print(f"\n✅ Escalation results saved to {output_path}")
    print(results_df.head())

if __name__ == "__main__":
    main()
