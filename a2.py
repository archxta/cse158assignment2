import pandas as pd
import numpy as np


# 1. LOAD RAW DATA (skip corrupted lines)

df = pd.read_csv(
    "/Users/architap/Downloads/redditSubmissions.csv",
    on_bad_lines='skip',
    engine='python'
)

print("Loaded dataset shape:", df.shape)



# 2. DROP COLUMNS WE DO NOT WANT
#    - They are redundant OR they leak post-outcome information

cols_to_drop = [
    '#image_id', 'rawtime', 'localtime',
    'reddit_id', 'username',
    'total_votes', 'number_of_upvotes', 'number_of_downvotes'
]

df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

print("After dropping columns:", df.shape)



# 3. DROP ROWS WITH MISSING TITLES

df = df[df['title'].notnull()].copy()
df['title'] = df['title'].astype(str)

print("After dropping missing titles:", df.shape)



# 4. PROCESS TIMESTAMP → extract hour + day-of-week

df['created'] = pd.to_datetime(df['unixtime'], unit='s', errors='coerce')

df['hour'] = df['created'].dt.hour
df['dayofweek'] = df['created'].dt.dayofweek

print("Extracted time features.")



# 5. FEATURE ENGINEERING ON TITLES

df['title_len'] = df['title'].str.len()
df['word_count'] = df['title'].str.split().str.len()
df['has_question'] = df['title'].str.contains(r'\?', regex=True).astype(int)
df['has_exclamation'] = df['title'].str.contains('!').astype(int)

print("Created text-based features.")



# 6. DEFINE POPULARITY: TOP 10% SCORE PER SUBREDDIT

df['pop_threshold'] = df.groupby('subreddit')['score'].transform(
    lambda x: x.quantile(0.90)
)

df['popular'] = (df['score'] >= df['pop_threshold']).astype(int)

print("Defined popularity labels.")



# 7. FINAL MODELING DATAFRAME

df_model = df[[
    'title',
    'subreddit',
    'hour',
    'dayofweek',
    'title_len',
    'word_count',
    'has_question',
    'has_exclamation',
    'popular'
]]

print("Final cleaned dataset shape:", df_model.shape)



# 8. SAVE CLEANED CSV

output_path = "/Users/architap/Downloads/redditSubmissions_cleaned.csv"
df_model.to_csv(output_path, index=False)

print("Saved cleaned dataset to:", output_path)