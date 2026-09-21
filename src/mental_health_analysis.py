"""
Stage 7: Mental-Health Theme Analysis Module
Project: Social Media Sentiment Analysis for Mental Health Awareness

This module identifies mental health themes and emotional well-being topics
using curated keyword lexicons, analyzes sentiment distribution per theme,
cross-references dataset labels and VADER/TextBlob predictions, and produces
high-value visualizations and reports.
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DATA_PATH = os.path.join(BASE_DIR, "outputs", "results", "sentiment_results.csv")
OUTPUT_RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
OUTPUT_FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
OUTPUT_REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")

os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_REPORTS_DIR, exist_ok=True)

# Curated Mental Health & Well-being Themes Lexicon
THEME_LEXICON = {
    "Depression": [
        "depression", "depressed", "depressing", "hopeless", "worthless",
        "miserable", "despair", "unhappy", "suicidal", "grief", "gloomy"
    ],
    "Anxiety": [
        "anxiety", "anxious", "panic", "panicking", "worry", "worried",
        "worrying", "nervous", "phobia", "dread", "apprehensive"
    ],
    "Stress & Burnout": [
        "stress", "stressed", "stressful", "overwhelmed", "burnout",
        "exhausted", "exhaustion", "overworked", "pressure", "breakdown"
    ],
    "Loneliness & Isolation": [
        "lonely", "loneliness", "isolated", "isolation", "alienated",
        "abandoned", "alone"
    ],
    "Sleep Disorders": [
        "insomnia", "sleepless", "nightmare", "restless", "sleep deprivation",
        "cant sleep", "cannot sleep"
    ],
    "Medication & Treatment": [
        "medication", "medicine", "pill", "pills", "prescription",
        "antidepressant", "ssri", "dosage", "pharma"
    ],
    "Therapy & Support": [
        "therapy", "therapist", "counseling", "counselor", "psychologist",
        "psychiatrist", "support group", "mental help"
    ],
    "Self-Care & Mindfulness": [
        "selfcare", "self-care", "wellness", "mindfulness", "meditation",
        "breathing exercise", "recharge", "self love"
    ],
    "Mental Health Awareness": [
        "mental health", "mental illness", "stigma", "break the stigma",
        "mental wellbeing", "psychological", "awareness"
    ],
    "Emotional Wellbeing": [
        "happy", "happiness", "joy", "grateful", "gratitude",
        "contentment", "peaceful", "peace of mind", "thriving"
    ]
}


def tag_themes(text: str) -> list[str]:
    """Identifies all matching mental health themes in text."""
    if not isinstance(text, str):
        return []
    
    text_lower = text.lower()
    matched = []
    for theme, keywords in THEME_LEXICON.items():
        pattern = r'\b(?:' + '|'.join(map(re.escape, keywords)) + r')\b'
        if re.search(pattern, text_lower):
            matched.append(theme)
    return matched


def plot_theme_distribution(theme_counts: pd.Series):
    """Plots frequency of identified mental-health themes."""
    plt.figure(figsize=(12, 6))
    palette = sns.color_palette("mako", len(theme_counts))
    ax = sns.barplot(x=theme_counts.values, y=theme_counts.index, palette=palette)
    
    plt.title("Mental-Health & Emotional Wellbeing Theme Mentions in Social Posts", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Number of Matching Tweets", fontsize=12)
    plt.ylabel("Theme Category", fontsize=12)
    
    for i, v in enumerate(theme_counts.values):
        ax.text(v + 5, i, f"{v:,}", va="center", fontsize=10, fontweight="semibold")

    plt.tight_layout()
    path = os.path.join(OUTPUT_FIGURES_DIR, "mental_health_theme_distribution.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f" Saved: {path}")


def plot_theme_sentiment_heatmap(theme_sentiment_df: pd.DataFrame):
    """Plots a heatmap of sentiment percentages per mental health theme."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(theme_sentiment_df, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Percentage of Tweets (%)'})
    plt.title("Sentiment Profile Across Mental-Health Themes (%)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Ground Truth Sentiment", fontsize=11)
    plt.ylabel("Mental Health Theme", fontsize=11)
    plt.tight_layout()
    
    path = os.path.join(OUTPUT_FIGURES_DIR, "mental_health_theme_sentiment_heatmap.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f" Saved: {path}")


def run_mental_health_analysis(input_path: str = INPUT_DATA_PATH):
    """Executes theme identification and generates insights."""
    print("=" * 60)
    print("Executing Stage 7: Mental-Health Theme Analysis...")
    print("=" * 60)

    print(f"\n Loading sentiment results from: {input_path}")
    df = pd.read_csv(input_path, encoding="utf-8")
    print(f" Loaded {len(df):,} annotated tweets.")

    print("\n Scanning texts for mental-health and emotional wellbeing themes...")
    # Scan cleaned and processed text for themes
    df["themes_list"] = df["cleaned_text"].fillna("").apply(tag_themes)
    df["theme_count"] = df["themes_list"].apply(len)
    df["is_mental_health_related"] = df["theme_count"] > 0
    df["primary_theme"] = df["themes_list"].apply(lambda x: x[0] if len(x) > 0 else "General Social")
    df["themes_detected"] = df["themes_list"].apply(lambda x: "; ".join(x) if len(x) > 0 else "None")

    # Filter mental health tweets
    mh_tweets = df[df["is_mental_health_related"]].copy()
    total_mh_tweets = len(mh_tweets)
    print(f" Total mental-health related tweets identified: {total_mh_tweets:,} ({total_mh_tweets/len(df)*100:.2f}%)")

    # Count individual theme occurrences (a tweet can belong to multiple themes)
    theme_occurrence = {}
    for theme in THEME_LEXICON.keys():
        theme_occurrence[theme] = int(df["themes_list"].apply(lambda x: theme in x).sum())

    theme_series = pd.Series(theme_occurrence).sort_values(ascending=False)

    # Calculate sentiment distribution per theme (for non-empty themes)
    theme_sentiment_rows = []
    for theme in theme_series.index:
        sub = df[df["themes_list"].apply(lambda x: theme in x)]
        if len(sub) > 0:
            counts = sub["Sentiment"].value_counts(normalize=True) * 100
            theme_sentiment_rows.append({
                "Theme": theme,
                "Negative": round(counts.get("Negative", 0.0), 1),
                "Positive": round(counts.get("Positive", 0.0), 1),
                "Neutral": round(counts.get("Neutral", 0.0), 1),
                "Irrelevant": round(counts.get("Irrelevant", 0.0), 1),
                "Total_Tweets": len(sub)
            })

    theme_sentiment_df = pd.DataFrame(theme_sentiment_rows).set_index("Theme")

    # Save output CSVs
    results_path = os.path.join(OUTPUT_RESULTS_DIR, "mental_health_themes.csv")
    df_out = df.drop(columns=["themes_list"])
    df_out.to_csv(results_path, index=False, encoding="utf-8")
    print(f" Saved full dataset with theme annotations: {results_path}")

    # Generate figures
    print("\n Generating mental health theme visualizations...")
    plot_theme_distribution(theme_series)
    
    # Plot heatmap using Negative, Positive, Neutral columns
    heatmap_cols = [c for c in ["Negative", "Positive", "Neutral", "Irrelevant"] if c in theme_sentiment_df.columns]
    plot_theme_sentiment_heatmap(theme_sentiment_df[heatmap_cols])

    # Save report
    report_path = os.path.join(OUTPUT_REPORTS_DIR, "mental_health_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("STAGE 7: MENTAL-HEALTH THEME ANALYSIS REPORT\n")
        f.write("Project: Social Media Sentiment Analysis for Mental Health Awareness\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. SUMMARY OF THEMATIC IDENTIFICATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Evaluated Tweets              : {len(df):,}\n")
        f.write(f"Mental Health / Well-being Tweets   : {total_mh_tweets:,} ({total_mh_tweets/len(df)*100:.2f}%)\n\n")

        f.write("2. THEME FREQUENCIES\n")
        f.write("-" * 40 + "\n")
        for theme, count in theme_series.items():
            f.write(f"  - {theme:<25}: {count:>5,} tweets\n")
        f.write("\n")

        f.write("3. SENTIMENT DISTRIBUTION PER THEME (%)\n")
        f.write("-" * 40 + "\n")
        f.write(theme_sentiment_df.to_string() + "\n\n")

        f.write("4. SAMPLE TWEETS BY MENTAL HEALTH THEME\n")
        f.write("-" * 40 + "\n")
        for theme in ["Depression", "Anxiety", "Stress & Burnout", "Therapy & Support"]:
            sample = df[df["themes_detected"].str.contains(theme, na=False)]["original_tweet"].iloc[:1]
            if len(sample) > 0:
                f.write(f"[{theme.upper()}]\n\"{sample.values[0]}\"\n\n")

        f.write("=" * 80 + "\n")
        f.write("ACADEMIC NOTE:\n")
        f.write("Keyword-based heuristic thematic analysis identifies linguistic mentions of mental health concepts\n")
        f.write("and does not indicate clinical diagnostic validation.\n")
        f.write("=" * 80 + "\n")

    print(f" Saved mental-health analysis report to: {report_path}")
    print(" Stage 7: Mental-Health Theme Analysis completed successfully!")
    return df, theme_sentiment_df


if __name__ == "__main__":
    run_mental_health_analysis()
