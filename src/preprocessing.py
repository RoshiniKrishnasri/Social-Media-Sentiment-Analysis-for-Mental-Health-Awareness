"""
Stage 5: Text Preprocessing Module (NLP Pipeline)
Project: Social Media Sentiment Analysis for Mental Health Awareness

This module loads data/processed/cleaned_tweets.csv and applies an NLP pipeline:
1. Lowercasing
2. Tokenization (NLTK word_tokenize)
3. Punctuation removal
4. Stopwords removal while preserving sentiment-critical negations & intensifiers
5. Lemmatization (NLTK WordNetLemmatizer)
6. Saving processed dataset to data/processed/processed_tweets.csv
"""

import os
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_tweets.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_tweets.csv")
OUTPUT_REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")

os.makedirs(OUTPUT_REPORTS_DIR, exist_ok=True)

# Ensure NLTK resources
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("wordnet")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("averaged_perceptron_tagger_eng")

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Build custom stopwords set:
# We preserve critical negations, intensifiers, and emotional markers
PRESERVE_WORDS = {
    "not", "no", "never", "nor", "neither", "cannot", "none",
    "very", "really", "so", "too", "much", "more", "most",
    "but", "however", "although", "except",
    "sad", "happy", "good", "bad", "great", "terrible", "depressed",
    "anxious", "stressed", "pain", "hurt", "hate", "love"
}

STANDARD_STOPWORDS = set(stopwords.words("english"))
CUSTOM_STOPWORDS = STANDARD_STOPWORDS - PRESERVE_WORDS


def get_wordnet_pos(treebank_tag: str):
    """
    Map POS tag to first character lemmatize() accepts.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess_text(text: str) -> str:
    """
    Full NLP Preprocessing pipeline:
    - Lowercase
    - Tokenize
    - Remove punctuation & non-alpha
    - Remove non-sentiment stopwords
    - Lemmatize tokens
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Tokenize
    tokens = word_tokenize(text)

    # 3. Filter punctuation, numbers, and non-sentiment stopwords
    filtered_tokens = []
    for token in tokens:
        # Strip internal punctuation e.g. 'm -> m, don't -> don't
        clean_token = token.strip(string.punctuation)
        if not clean_token or clean_token.isdigit() or len(clean_token) <= 1 and clean_token not in {"a", "i"}:
            continue
        if clean_token in CUSTOM_STOPWORDS:
            continue
        filtered_tokens.append(clean_token)

    if not filtered_tokens:
        return ""

    # 4. Lemmatization
    lemmatized = [lemmatizer.lemmatize(tok) for tok in filtered_tokens]

    return " ".join(lemmatized)


def run_preprocessing(input_path: str = CLEANED_DATA_PATH) -> pd.DataFrame:
    """
    Executes text preprocessing on the cleaned dataset.
    """
    print("=" * 60)
    print("Executing Stage 5: Text Preprocessing (NLP Pipeline)...")
    print("=" * 60)

    print(f"\n Loading cleaned dataset from: {input_path}")
    df = pd.read_csv(input_path, encoding="utf-8")
    initial_count = len(df)
    print(f" Loaded {initial_count:,} cleaned rows.")

    print("\n Applying tokenization, stopword removal & lemmatization...")
    # Fill any null values in cleaned_text
    df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)
    
    # Process text
    df["processed_text"] = df["cleaned_text"].apply(preprocess_text)

    # Safely handle any empty processed text by falling back to cleaned_text lowercased
    empty_mask = df["processed_text"].str.strip() == ""
    empty_count = int(empty_mask.sum())
    if empty_count > 0:
        print(f" Note: {empty_count:,} rows had empty processed tokens. Using cleaned_text fallback.")
        df.loc[empty_mask, "processed_text"] = df.loc[empty_mask, "cleaned_text"].str.lower()

    # Save processed dataset
    print(f"\n Saving processed dataset to: {PROCESSED_DATA_PATH}")
    df.to_csv(PROCESSED_DATA_PATH, index=False, encoding="utf-8")
    print(f" Successfully saved {len(df):,} processed tweets.")

    # Save summary report
    report_path = os.path.join(OUTPUT_REPORTS_DIR, "preprocessing_summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("STAGE 5: TEXT PREPROCESSING (NLP) SUMMARY\n")
        f.write("Project: Social Media Sentiment Analysis for Mental Health Awareness\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Input Dataset Path         : {input_path}\n")
        f.write(f"Total Rows Processed       : {initial_count:,}\n")
        f.write(f"Rows with Empty Fallback   : {empty_count:,}\n")
        f.write(f"Output Dataset Path        : {PROCESSED_DATA_PATH}\n\n")
        f.write(f"Stopwords Strategy:\n")
        f.write(f"  - Total Standard Stopwords : {len(STANDARD_STOPWORDS)}\n")
        f.write(f"  - Preserved Sentiment Words: {len(PRESERVE_WORDS)} ({', '.join(sorted(PRESERVE_WORDS))})\n")
        f.write(f"  - Active Custom Stopwords  : {len(CUSTOM_STOPWORDS)}\n\n")
        f.write("Sample Transformations:\n")
        f.write("-" * 40 + "\n")
        samples = df.sample(n=min(5, len(df)), random_state=42)
        for _, row in samples.iterrows():
            f.write(f"[ORIGINAL]:  {row['original_tweet']}\n")
            f.write(f"[CLEANED]:   {row['cleaned_text']}\n")
            f.write(f"[PROCESSED]: {row['processed_text']}\n")
            f.write("-" * 40 + "\n")

    print(f" Saved preprocessing summary report to: {report_path}")
    print(" Stage 5: Text Preprocessing completed successfully!")
    return df


if __name__ == "__main__":
    run_preprocessing()
