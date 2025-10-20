import pandas as pd

df = pd.read_csv("european/clubs_tweets.csv", low_memory=False)

# Use the correct datetime column
df['tweet_created_at'] = pd.to_datetime(df['tweet_created_at'], errors='coerce')

# ✅ Instead of filtering by year, keep all valid tweets
df_clean = df[['tweet_created_at', 'tweet_full_text', 'user_screen_name']].rename(
    columns={
        'tweet_created_at': 'date',
        'tweet_full_text': 'text',
        'user_screen_name': 'club_name'
    }
)

# Drop missing or blank text
df_clean = df_clean.dropna(subset=['text'])
df_clean = df_clean[df_clean['text'].str.strip() != ""]

# Sample 1500 random tweets for analysis
df_clean = df_clean.sample(n=1500, random_state=42) if len(df_clean) > 1500 else df_clean

# Save the dataset
output_path = "european/football_tweets_all.csv"
df_clean.to_csv(output_path, index=False)
print(f"✅ Created dataset with {len(df_clean)} tweets: {output_path}")
