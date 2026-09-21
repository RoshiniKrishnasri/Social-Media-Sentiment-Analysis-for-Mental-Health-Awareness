"""
Stage 3: Data Cleaning Module
Project: Social Media Sentiment Analysis for Mental Health Awareness

This module loads the raw Twitter CSV, performs cleaning and text normalization,
preserves sentiment-bearing words and emoji signals, deduplicates rows,
and saves the cleaned dataset to data/processed/cleaned_tweets.csv.
"""

import os
import re
import html
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "twitter_training.csv")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_REPORTS_DIR, exist_ok=True)

COLUMN_NAMES = ["TweetID", "Entity", "Sentiment", "TweetText"]

# Common social media emoji mapping to sentiment-bearing descriptive words
EMOJI_SENTIMENT_MAP = {
    "😊": " happy ", "🙂": " happy ", "😀": " happy ", "😃": " happy ", "😄": " happy ",
    "😁": " joyful ", "🥰": " love ", "😍": " love ", "❤️": " love ", "💕": " love ",
    "😢": " sad ", "😭": " crying sad ", "😞": " depressed sad ", "😔": " sad pensive ",
    "🙁": " unhappy ", "☹️": " unhappy ", "💔": " heartbroken ", "🥺": " pleading sad ",
    "😡": " angry ", "😠": " angry mad ", "🤬": " furious angry ", "😤": " frustrated ",
    "😱": " anxious panic ", "😨": " fearful anxious ", "😰": " nervous anxiety ",
    "😴": " tired exhausted ", "🥱": " exhausted ", "🙏": " hopeful grateful ",
    "👍": " good ", "👎": " bad ", "✨": " inspired ", "🔥": " awesome "
}


def clean_tweet_text(text: str) -> str:
    """
    Cleans and normalizes a single tweet string:
    1. Decodes HTML entities (e.g., &amp; -> &)
    2. Maps common emotional emojis to sentiment words
    3. Removes URLs and media links (http, https, pic.twitter.com, dlvr.it)
    4. Removes @mentions
    5. Strips hashtag symbol while keeping the hashtag word intact (#mentalhealth -> mentalhealth)
    6. Removes HTML markup tags
    7. Cleans corrupted encoding artifacts (e.g. ?T, dY)
    8. Normalizes repeated characters (e.g., "sooooo" -> "soo")
    9. Removes non-alphanumeric noise while keeping words, numbers, and basic punctuation
    10. Normalizes multiple spaces to a single space
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Decode HTML entities
    text = html.unescape(text)
    
    # 2. Replace known emotional emojis with sentiment words
    for emoji_char, replacement in EMOJI_SENTIMENT_MAP.items():
        if emoji_char in text:
            text = text.replace(emoji_char, replacement)
            
    # 3. Remove URLs & Twitter picture links (including spaced slashes like 'https: / / ...')
    text = re.sub(r'https?\s*:\s*/\s*/\s*\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+|www\.\S+|pic\.twitter\.com/\S+|dlvr\.it/\S+|j\.mp\s*/\s*\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:https?|ftp)://\S+', '', text)
    
    # 4. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 5. Remove @user mentions
    text = re.sub(r'@\w+', '', text)
    
    # 6. Preserve hashtag content (strip '#' symbol only)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # 7. Remove corrupted encoding replacement characters often found in Twitter dumps
    text = re.sub(r'[\uFFFD]', '', text)
    text = re.sub(r'd[A-Z0-9]{1,3}', '', text)
    
    # 8. Normalize repeated characters (3+ occurrences reduced to 2, e.g. "sooooo" -> "soo")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # 9. Clean special symbols while retaining standard English apostrophes and hyphens
    text = re.sub(r"[^a-zA-Z0-9\s'\-]", ' ', text)
    
    # 10. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def clean_dataset(input_filepath: str = RAW_DATA_PATH) -> tuple[pd.DataFrame, dict]:
    """
    Executes the complete dataset cleaning procedure.
    """
    print("=" * 60)
    print("Executing Stage 3: Data Cleaning...")
    print("=" * 60)

    # 1. Load Raw CSV
    print(f"\n Loading raw dataset from: {input_filepath}")
    df_raw = pd.read_csv(input_filepath, header=None, names=COLUMN_NAMES, encoding="utf-8")
    original_row_count = len(df_raw)
    print(f" Original row count: {original_row_count:,}")

    # Preserve original tweet text in a dedicated column
    df = df_raw.copy()
    df["original_tweet"] = df["TweetText"]

    # 2. Remove rows where tweet text is null or blank
    missing_text_mask = df["TweetText"].isnull() | (df["TweetText"].astype(str).str.strip() == "")
    removed_missing_rows = int(missing_text_mask.sum())
    df = df[~missing_text_mask].copy()
    print(f" Removed missing/empty text rows: {removed_missing_rows:,}")

    # 3. Remove duplicate rows (based on TweetID and TweetText to ensure integrity)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["TweetText"]).copy()
    removed_duplicate_rows = before_dedup - len(df)
    print(f" Removed duplicate tweet rows: {removed_duplicate_rows:,}")

    # 4. Clean text using the robust cleaning pipeline
    print(" Cleaning and normalizing tweet text...")
    df["cleaned_text"] = df["original_tweet"].apply(clean_tweet_text)

    # 5. Remove rows where cleaned text resulted in an empty string
    empty_cleaned_mask = df["cleaned_text"].str.strip() == ""
    removed_empty_cleaned = int(empty_cleaned_mask.sum())
    df = df[~empty_cleaned_mask].copy()
    print(f" Removed empty tweets after cleaning: {removed_empty_cleaned:,}")

    final_row_count = len(df)
    print(f" Final cleaned row count: {final_row_count:,}")

    # Reorder columns logically
    df = df[["TweetID", "Entity", "Sentiment", "original_tweet", "cleaned_text"]]

    # 6. Save Cleaned Dataset
    output_csv_path = os.path.join(PROCESSED_DATA_DIR, "cleaned_tweets.csv")
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f" Saved cleaned dataset to: {output_csv_path}")

    # 7. Generate Cleaning Summary Report
    summary = {
        "original_row_count": original_row_count,
        "removed_missing_rows": removed_missing_rows,
        "removed_duplicate_rows": removed_duplicate_rows,
        "removed_empty_cleaned": removed_empty_cleaned,
        "total_removed": original_row_count - final_row_count,
        "final_row_count": final_row_count,
        "retention_rate_pct": (final_row_count / original_row_count) * 100,
        "sentiment_counts": df["Sentiment"].value_counts().to_dict(),
        "entities_count": df["Entity"].nunique()
    }

    report_path = os.path.join(OUTPUT_REPORTS_DIR, "cleaning_summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("STAGE 3: DATA CLEANING SUMMARY\n")
        f.write("Project: Social Media Sentiment Analysis for Mental Health Awareness\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Original Row Count           : {summary['original_row_count']:,}\n")
        f.write(f"Removed Missing Text Rows    : {summary['removed_missing_rows']:,}\n")
        f.write(f"Removed Duplicate Text Rows  : {summary['removed_duplicate_rows']:,}\n")
        f.write(f"Removed Post-Clean Empty Rows: {summary['removed_empty_cleaned']:,}\n")
        f.write(f"Total Rows Removed           : {summary['total_removed']:,}\n")
        f.write(f"Final Cleaned Row Count      : {summary['final_row_count']:,}\n")
        f.write(f"Data Retention Rate          : {summary['retention_rate_pct']:.2f}%\n\n")
        f.write("Cleaned Sentiment Distribution:\n")
        for sent, cnt in summary["sentiment_counts"].items():
            f.write(f"  - {sent:<12}: {cnt:>6,} ({cnt/final_row_count*100:.2f}%)\n")
        f.write("\nSample Cleaned Transformations:\n")
        f.write("-" * 40 + "\n")
        samples = df.sample(n=min(5, len(df)), random_state=42)
        for _, row in samples.iterrows():
            f.write(f"[RAW]:     {row['original_tweet']}\n")
            f.write(f"[CLEANED]: {row['cleaned_text']}\n")
            f.write("-" * 40 + "\n")

    print(f" Saved cleaning summary report to: {report_path}")
    print(" Stage 3: Data Cleaning completed successfully!")
    return df, summary


if __name__ == "__main__":
    clean_dataset()
