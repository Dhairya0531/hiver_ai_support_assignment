import os
import re
import pandas as pd
import numpy as np

RAW_CSV_PATH = os.path.expanduser('~/.cache/kagglehub/datasets/thoughtvector/customer-support-on-twitter/versions/10/twcs/twcs.csv')
OUTPUT_DIR = 'data/processed'
RAW_APPLE_CACHE = os.path.join(OUTPUT_DIR, 'apple_tweets_raw.parquet')

def clean_tweet_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Remove twitter handles like @AppleSupport @115854
    text = re.sub(r'@[\w_]+', '', text)
    # Fix unicode variation selectors like I️ (U+FE0F) common in iOS 11 bug
    text = text.replace('\ufe0f', '').replace('\u200b', '')
    # Replace HTML entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    # Normalize multiple whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_applesupport_pairs(limit_rows: int = None, sample_size: int = 15000):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if os.path.exists(RAW_APPLE_CACHE):
        print(f"Loading cached Apple tweets from {RAW_APPLE_CACHE}...")
        df_all = pd.read_parquet(RAW_APPLE_CACHE)
    else:
        print(f"Reading {RAW_CSV_PATH} to extract AppleSupport tweets...")
        chunks = []
        chunk_size = 250000
        total_read = 0
        
        for chunk in pd.read_csv(RAW_CSV_PATH, chunksize=chunk_size, dtype=str):
            total_read += len(chunk)
            mask = (chunk['author_id'] == 'AppleSupport') | (chunk['text'].str.contains('@AppleSupport', case=False, na=False))
            relevant = chunk[mask].copy()
            if not relevant.empty:
                chunks.append(relevant)
            print(f"Scanned {total_read:,} rows, captured {sum(len(c) for c in chunks):,} candidate tweets...")
            if limit_rows and total_read >= limit_rows:
                break
                
        df_all = pd.concat(chunks, ignore_index=True)
        print(f"Total candidate AppleSupport rows: {len(df_all):,}")
        df_all.to_parquet(RAW_APPLE_CACHE, index=False)
        print(f"Cached raw Apple tweets to {RAW_APPLE_CACHE}")

    # Ensure clean string IDs
    df_all['tweet_id'] = df_all['tweet_id'].astype(str).str.strip()
    df_all['inbound'] = df_all['inbound'].astype(str).str.lower() == 'true'
    
    # 1. Separate inbound customer tweets from AppleSupport agent replies
    inbound_df = df_all[df_all['inbound'] == True].copy()
    outbound_df = df_all[df_all['author_id'] == 'AppleSupport'].copy()
    
    print(f"Inbound customer tweets: {len(inbound_df):,}")
    print(f"Outbound AppleSupport tweets: {len(outbound_df):,}")
    
    # Clean in_response_to_tweet_id in outbound
    outbound_df['in_response_to_clean'] = outbound_df['in_response_to_tweet_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    outbound_valid = outbound_df[~outbound_df['in_response_to_clean'].isin(['nan', 'none', '', '<na>'])].copy()
    
    # Deduplicate to earliest response per customer tweet
    outbound_first_reply = outbound_valid.drop_duplicates(subset=['in_response_to_clean'], keep='first')
    
    # Merge inbound with outbound
    pairs = pd.merge(
        inbound_df,
        outbound_first_reply[['tweet_id', 'text', 'in_response_to_clean', 'created_at']],
        left_on='tweet_id',
        right_on='in_response_to_clean',
        suffixes=('_cust', '_agent')
    )
    print(f"Matched customer-agent conversation pairs: {len(pairs):,}")
    
    # Filter for initial inquiries (where customer did NOT specify in_response_to_tweet_id)
    cust_resp_col = pairs['in_response_to_tweet_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip().str.lower()
    is_initial = cust_resp_col.isna() | cust_resp_col.isin(['nan', 'none', '', '<na>'])
    initial_inquiries = pairs[is_initial].copy()
    print(f"Initial inquiry customer-agent pairs: {len(initial_inquiries):,}")
    
    # Clean text
    initial_inquiries['customer_text'] = initial_inquiries['text_cust'].apply(clean_tweet_text)
    initial_inquiries['agent_text'] = initial_inquiries['text_agent'].apply(clean_tweet_text)
    
    # Quality filter: must have substantive query & response (>= 15 chars)
    valid_pairs = initial_inquiries[
        (initial_inquiries['customer_text'].str.len() >= 15) &
        (initial_inquiries['agent_text'].str.len() >= 15)
    ].copy()
    print(f"High-quality valid initial pairs: {len(valid_pairs):,}")
    
    # Select & rename columns
    df_clean = valid_pairs[[
        'tweet_id_cust', 'customer_text', 'text_cust', 'created_at_cust',
        'tweet_id_agent', 'agent_text', 'text_agent', 'created_at_agent'
    ]].rename(columns={
        'tweet_id_cust': 'customer_tweet_id',
        'text_cust': 'customer_raw_text',
        'created_at_cust': 'customer_created_at',
        'tweet_id_agent': 'agent_tweet_id',
        'text_agent': 'agent_raw_text',
        'created_at_agent': 'agent_created_at'
    }).reset_index(drop=True)
    
    output_parquet = os.path.join(OUTPUT_DIR, 'applesupport_pairs.parquet')
    df_clean.to_parquet(output_parquet, index=False)
    print(f"Saved {len(df_clean):,} pairs to {output_parquet}")
    
    sample_df = df_clean.sample(n=min(sample_size, len(df_clean)), random_state=42).reset_index(drop=True)
    sample_csv = os.path.join(OUTPUT_DIR, 'applesupport_pairs_sample.csv')
    sample_df.to_csv(sample_csv, index=False)
    print(f"Saved {len(sample_df):,} sample pairs to {sample_csv}")
    
    print("\n--- SAMPLE EXTRACTED CONVERSATIONS ---")
    for idx, row in sample_df.head(3).iterrows():
        print(f"\n[Pair #{idx+1}]")
        print(f"Customer: {row['customer_text']}")
        print(f"AppleSupport: {row['agent_text']}")
        
    return df_clean

if __name__ == '__main__':
    extract_applesupport_pairs()
