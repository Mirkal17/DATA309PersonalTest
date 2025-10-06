import os
import pandas as pd
from Agents.Sentiment.Sentiment_Agent import classify_sentiment
from Preprocessing.Prepare import load_and_prepare 

def main():
    df = pd.read_csv("Data/customer_support_tickets_cleaned.csv")

    result = []

    # Test on first 5 tickets
    for i in range(1000):
        text = df.iloc[i]["Ticket Description_cleaned"]
        res = classify_sentiment(text)
        result.append({
            "Ticket ID": df.iloc[i]["Ticket ID"],
            "Text": text,
            "Sentiment": res['label'],
            "Method": res['method'],
            "Confidence": res['confidence']
        })

        # print(f"\nTicket {i+1}:")
        # print(f"Text: {text[:80]}...")
        # print(f"Label: {result['label']} (via {result['method']}, confidence {result['confidence']:.2f})")

    output_path = os.path.join("Data", "sentiment_results.csv")
    result_df = pd.DataFrame(result)
    result_df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")
        

if __name__ == "__main__":
    main()


