"""
Stage 2: Data Understanding Module
Project: Social Media Sentiment Analysis for Mental Health Awareness

This script loads the raw Twitter CSV files, inspects the schema, checks for missing
values and duplicates, summarizes sentiment and entity distributions, and writes
a comprehensive data understanding report and visual charts.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
OUTPUT_FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")

os.makedirs(OUTPUT_REPORTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_FIGURES_DIR, exist_ok=True)

COLUMN_NAMES = ["TweetID", "Entity", "Sentiment", "TweetText"]


def load_dataset(filename: str) -> pd.DataFrame:
    """
    Loads raw CSV without header and assigns standardized column names.
    """
    filepath = os.path.join(RAW_DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Load dataset with predefined column names
    df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES, encoding="utf-8")
    return df


def inspect_dataset(df: pd.DataFrame, dataset_name: str) -> dict:
    """
    Performs comprehensive data inspection on the given dataframe.
    """
    total_rows, total_cols = df.shape
    missing_vals = df.isnull().sum()
    duplicate_count = df.duplicated().sum()
    duplicate_texts = df["TweetText"].dropna().duplicated().sum()
    sentiments = df["Sentiment"].value_counts()
    sentiment_pct = df["Sentiment"].value_counts(normalize=True) * 100
    unique_entities = df["Entity"].nunique()
    
    # Text length statistics (for non-null tweets)
    tweet_lengths = df["TweetText"].dropna().astype(str).apply(len)
    word_counts = df["TweetText"].dropna().astype(str).apply(lambda x: len(x.split()))
    
    stats = {
        "name": dataset_name,
        "shape": (total_rows, total_cols),
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": missing_vals.to_dict(),
        "total_duplicates": int(duplicate_count),
        "duplicate_tweets": int(duplicate_texts),
        "sentiments": sentiments.to_dict(),
        "sentiment_percentages": sentiment_pct.to_dict(),
        "unique_entities_count": unique_entities,
        "sample_entities": list(df["Entity"].unique()[:10]),
        "text_length_mean": float(tweet_lengths.mean()),
        "text_length_max": int(tweet_lengths.max()),
        "word_count_mean": float(word_counts.mean()),
        "word_count_max": int(word_counts.max()),
        "tweet_column": "TweetText",
        "sentiment_column": "Sentiment"
    }
    return stats


def generate_visualizations(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """
    Generates beginner-friendly, high-contrast exploratory plots for data understanding.
    """
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. Training vs Validation Sentiment Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.countplot(
        data=df_train,
        x="Sentiment",
        order=["Positive", "Negative", "Neutral", "Irrelevant"],
        ax=axes[0],
        palette="Blues_d"
    )
    axes[0].set_title("Training Set: Sentiment Distribution (N=74,682)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Sentiment Label", fontsize=11)
    axes[0].set_ylabel("Tweet Count", fontsize=11)
    for p in axes[0].patches:
        axes[0].annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 3),
                         textcoords='offset points')
        
    sns.countplot(
        data=df_val,
        x="Sentiment",
        order=["Positive", "Negative", "Neutral", "Irrelevant"],
        ax=axes[1],
        palette="Greens_d"
    )
    axes[1].set_title("Validation Set: Sentiment Distribution (N=1,000)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Sentiment Label", fontsize=11)
    axes[1].set_ylabel("Tweet Count", fontsize=11)
    for p in axes[1].patches:
        axes[1].annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 3),
                         textcoords='offset points')

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_FIGURES_DIR, "data_understanding_sentiment_distribution.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f" Saved sentiment distribution plot: {plot_path}")

    # 2. Top Entities in Training Set
    plt.figure(figsize=(12, 6))
    entity_counts = df_train["Entity"].value_counts().head(15)
    sns.barplot(x=entity_counts.values, y=entity_counts.index, palette="viridis")
    plt.title("Top 15 Most Frequent Entities in Dataset", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Tweets", fontsize=11)
    plt.ylabel("Entity", fontsize=11)
    plt.tight_layout()
    entity_plot_path = os.path.join(OUTPUT_FIGURES_DIR, "data_understanding_top_entities.png")
    plt.savefig(entity_plot_path, dpi=300)
    plt.close()
    print(f" Saved entity distribution plot: {entity_plot_path}")


def save_report(train_stats: dict, val_stats: dict, df_train: pd.DataFrame):
    """
    Writes a formatted, beginner-friendly report to outputs/reports/data_understanding_report.txt.
    """
    report_path = os.path.join(OUTPUT_REPORTS_DIR, "data_understanding_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("STAGE 2: DATA UNDERSTANDING REPORT\n")
        f.write("Project: Social Media Sentiment Analysis for Mental Health Awareness\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. DATASET OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training Dataset Shape: {train_stats['shape'][0]:,} rows x {train_stats['shape'][1]} columns\n")
        f.write(f"Validation Dataset Shape: {val_stats['shape'][0]:,} rows x {val_stats['shape'][1]} columns\n")
        f.write(f"Columns: {train_stats['columns']}\n")
        f.write(f"Tweet Text Column: '{train_stats['tweet_column']}'\n")
        f.write(f"Sentiment Label Column: '{train_stats['sentiment_column']}'\n\n")

        f.write("2. COLUMN DATA TYPES\n")
        f.write("-" * 40 + "\n")
        for col, dtype in train_stats["dtypes"].items():
            f.write(f"  - {col}: {dtype}\n")
        f.write("\n")

        f.write("3. MISSING VALUES & DUPLICATES\n")
        f.write("-" * 40 + "\n")
        f.write("Training Set:\n")
        for col, count in train_stats["missing_values"].items():
            f.write(f"  - Missing in {col}: {count:,}\n")
        f.write(f"  - Exact Duplicate Rows: {train_stats['total_duplicates']:,}\n")
        f.write(f"  - Duplicate Tweet Texts: {train_stats['duplicate_tweets']:,}\n\n")

        f.write("Validation Set:\n")
        for col, count in val_stats["missing_values"].items():
            f.write(f"  - Missing in {col}: {count:,}\n")
        f.write(f"  - Exact Duplicate Rows: {val_stats['total_duplicates']:,}\n\n")

        f.write("4. SENTIMENT LABELS & DISTRIBUTION (TRAINING SET)\n")
        f.write("-" * 40 + "\n")
        for sentiment, count in train_stats["sentiments"].items():
            pct = train_stats["sentiment_percentages"][sentiment]
            f.write(f"  - {sentiment:<12}: {count:>6,} tweets ({pct:.2f}%)\n")
        f.write("\n")

        f.write("5. ENTITIES / TOPICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Unique Entities: {train_stats['unique_entities_count']}\n")
        f.write(f"Sample Entities: {', '.join(train_stats['sample_entities'])}\n\n")

        f.write("6. TEXT CHARACTERISTICS (TRAINING SET)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  - Mean Character Length: {train_stats['text_length_mean']:.1f} chars\n")
        f.write(f"  - Max Character Length: {train_stats['text_length_max']} chars\n")
        f.write(f"  - Mean Word Count: {train_stats['word_count_mean']:.1f} words\n")
        f.write(f"  - Max Word Count: {train_stats['word_count_max']} words\n\n")

        f.write("7. SAMPLE TWEETS PER SENTIMENT\n")
        f.write("-" * 40 + "\n")
        for sentiment in ["Positive", "Negative", "Neutral", "Irrelevant"]:
            sample = df_train[df_train["Sentiment"] == sentiment]["TweetText"].dropna().iloc[0]
            f.write(f"[{sentiment.upper()}]\n\"{sample}\"\n\n")

        f.write("=" * 80 + "\n")
        f.write("Key Takeaway for Downstream Pipeline:\n")
        f.write("1. 686 tweets in the training set have missing text and must be dropped during data cleaning.\n")
        f.write("2. 2,700 exact duplicate rows should be deduplicated to prevent data leakage and bias.\n")
        f.write("3. Four distinct sentiment classes exist: Positive, Negative, Neutral, and Irrelevant.\n")
        f.write("4. The original dataset text is unnormalized, containing URLs, @mentions, and repeated characters.\n")
        f.write("=" * 80 + "\n")

    print(f" Saved Data Understanding Report: {report_path}")


def run_data_understanding():
    """
    Executes the entire data understanding stage.
    """
    print("=" * 60)
    print("Executing Stage 2: Data Understanding...")
    print("=" * 60)

    # 1. Load data
    print("\n Loading raw datasets...")
    df_train = load_dataset("twitter_training.csv")
    df_val = load_dataset("twitter_validation.csv")
    print(f" Loaded training set: {df_train.shape[0]:,} rows, {df_train.shape[1]} columns")
    print(f" Loaded validation set: {df_val.shape[0]:,} rows, {df_val.shape[1]} columns")

    # 2. Inspect data
    print("\n Inspecting data attributes...")
    train_stats = inspect_dataset(df_train, "Training Set")
    val_stats = inspect_dataset(df_val, "Validation Set")

    print(f" Column names: {train_stats['columns']}")
    print(f" Tweet text column: '{train_stats['tweet_column']}'")
    print(f" Sentiment column: '{train_stats['sentiment_column']}'")
    print(f" Missing TweetText in training set: {train_stats['missing_values']['TweetText']:,}")
    print(f" Exact duplicate rows: {train_stats['total_duplicates']:,}")

    print("\n Sentiment distribution (Training):")
    for sentiment, count in train_stats["sentiments"].items():
        pct = train_stats["sentiment_percentages"][sentiment]
        print(f"  • {sentiment:<12}: {count:>6,} ({pct:.2f}%)")

    # 3. Generate Visualizations
    print("\n Generating exploratory figures...")
    generate_visualizations(df_train, df_val)

    # 4. Save Report
    print("\n Writing data understanding report...")
    save_report(train_stats, val_stats, df_train)

    print("\n Stage 2: Data Understanding completed successfully!")
    return df_train, df_val


if __name__ == "__main__":
    run_data_understanding()
