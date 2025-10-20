import pandas as pd
from transformers import pipeline

# ✅ Load the cleaned dataset (with all tweets)
df = pd.read_csv("european/football_tweets_all.csv")

# Clean up empty text
df = df.dropna(subset=["text"])
df = df[df["text"].str.strip() != ""]

# ✅ Load a fast transformer model (runs great on M1/M2)
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Predict sentiment safely
def safe_predict(text):
    try:
        return sentiment_analyzer(str(text)[:512])[0]["label"]
    except Exception:
        return "NEUTRAL"

df["sentiment"] = df["text"].apply(safe_predict)

# ✅ Save output
output_path = "european/football_sentiments_all.csv"
df.to_csv(output_path, index=False)
print(f"✅ Sentiment file created: {output_path}")
print(df["sentiment"].value_counts())
