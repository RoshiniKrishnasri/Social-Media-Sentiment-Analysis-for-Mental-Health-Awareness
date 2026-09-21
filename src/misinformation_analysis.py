"""
Stage 9: Misinformation & Questionable Health Claims Analysis
Project: Social Media Sentiment Analysis for Mental Health Awareness

This module identifies social media posts that contain potentially questionable,
exaggerated, or medically unsubstantiated claims regarding mental health conditions
and pharmacological treatments.

ACADEMIC DISCLAIMER:
"The system flags potentially questionable content for further human review;
it does not determine medical truth automatically."
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DATA_PATH = os.path.join(BASE_DIR, "outputs", "results", "topic_model_results.csv")
OUTPUT_RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
OUTPUT_FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
OUTPUT_REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")

os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_REPORTS_DIR, exist_ok=True)

# Questionable / Suspicious Health Claim Patterns
QUESTIONABLE_PATTERNS = {
    "Toxicity & Contamination Alarm": [
        r"\b(?:toxic|poison|poisonous|asbestos|carcinogen|harmful\s+chemical)\b",
        r"\b(?:cancer\s+causing|causes?\s+cancer)\b",
        r"\bcontaminated\s+with\b"
    ],
    "Dangerous Drug & Substance Claim": [
        r"\b(?:dangerous|party|deadly)\s+drugs?\b",
        r"\bwar\s+on\s+drugs\b",
        r"\bdrug\s+dealer\b"
    ],
    "Unsubstantiated Cure & Miracle Claim": [
        r"\bcures?\s+(?:depression|anxiety|mental\s+illness|cancer|illness|disease|ptsd)\b",
        r"\bcure\s+for\s+(?:depression|anxiety|mental\s+illness|cancer)\b",
        r"\bguaranteed\s+(?:cure|fix)\b",
        r"\bmiracle\s+(?:cure|drug|remedy|treatment|pill)\b",
        r"\b100%\s+effective\s+cure\b"
    ],
    "Medication Cessation / Anti-Treatment Advice": [
        r"\bstop\s+taking\s+(?:antidepressants|meds|medication|pills|ssris?)\b",
        r"\bthrow\s+away\s+your\s+(?:pills|meds|antidepressants)\b",
        r"\bantidepressants?\s+(?:are|is)\s+(?:poison|toxic|fake|a\s+scam)\b",
        r"\bmedication\s+is\s+poison\b"
    ],
    "Mental Health Denialism": [
        r"\b(?:depression|anxiety|mental\s+illness)\s+is(?:n't|\s+not)\s+real\b",
        r"\bmental\s+illness\s+(?:is\s+a\s+myth|is\s+fake|is\s+made\s+up)\b",
        r"\ball\s+in\s+your\s+head\s+just\s+get\s+over\s+it\b"
    ]
}


def scan_for_questionable_claims(text: str) -> tuple[bool, str, str]:
    """
    Scans a tweet for potential misinformation or unsupported claims.
    Returns: (is_flagged, category, matched_text)
    """
    if not isinstance(text, str) or not text.strip():
        return False, "None", "None"

    text_lower = text.lower()
    for category, patterns in QUESTIONABLE_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return True, category, match.group(0)

    return False, "None", "None"


def plot_flag_categories(cat_counts: pd.Series):
    """Plots frequency of flagged questionable content by category."""
    plt.figure(figsize=(10, 5))
    palette = sns.color_palette("rocket", len(cat_counts))
    ax = sns.barplot(x=cat_counts.values, y=cat_counts.index, palette=palette)
    
    plt.title("Questionable Content Flags by Category (Human Review Required)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Number of Flagged Tweets", fontsize=11)
    plt.ylabel("Flag Reason Category", fontsize=11)
    
    for i, v in enumerate(cat_counts.values):
        ax.text(v + 0.5, i, f"{v:,}", va="center", fontsize=10, fontweight="semibold")

    plt.tight_layout()
    path = os.path.join(OUTPUT_FIGURES_DIR, "misinformation_flags_by_category.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f" Saved: {path}")


def run_misinformation_analysis(input_path: str = INPUT_DATA_PATH):
    """Executes questionable claim scanning across all tweets."""
    print("=" * 60)
    print("Executing Stage 9: Misinformation & Questionable Claim Analysis...")
    print("=" * 60)

    print(f"\n Loading dataset from: {input_path}")
    df = pd.read_csv(input_path, encoding="utf-8")
    print(f" Loaded {len(df):,} tweets.")

    print("\n Scanning for unsupported medical claims and high-risk language...")
    results = [scan_for_questionable_claims(t) for t in df["cleaned_text"].fillna("").astype(str)]
    
    is_flagged, categories, matched_phrases = zip(*results)
    
    df["is_questionable_claim"] = is_flagged
    df["flag_category"] = categories
    df["flag_matched_pattern"] = matched_phrases
    df["human_review_required"] = is_flagged

    flagged_df = df[df["is_questionable_claim"]].copy()
    total_flagged = len(flagged_df)
    print(f" Total flagged tweets requiring human review: {total_flagged:,}")

    # Category counts
    cat_counts = flagged_df["flag_category"].value_counts()

    # Plot if flags exist
    if len(cat_counts) > 0:
        plot_flag_categories(cat_counts)
    else:
        print(" No questionable claims matched the strict clinical criteria.")

    # Save flagged dataset
    flagged_csv_path = os.path.join(OUTPUT_RESULTS_DIR, "flagged_content.csv")
    flagged_df.to_csv(flagged_csv_path, index=False, encoding="utf-8")
    print(f" Saved flagged tweets to: {flagged_csv_path}")

    # Save complete enriched final dataset
    final_dataset_path = os.path.join(OUTPUT_RESULTS_DIR, "final_analysis_dataset.csv")
    df.to_csv(final_dataset_path, index=False, encoding="utf-8")
    print(f" Saved consolidated final dataset to: {final_dataset_path}")

    # Write Report
    report_path = os.path.join(OUTPUT_REPORTS_DIR, "misinformation_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("STAGE 9: MISINFORMATION & QUESTIONABLE CLAIM ANALYSIS REPORT\n")
        f.write("Project: Social Media Sentiment Analysis for Mental Health Awareness\n")
        f.write("=" * 80 + "\n\n")

        f.write("ACADEMIC STATEMENT & SAFEGUARD POLICY:\n")
        f.write('"The system flags potentially questionable content for further human review;\n')
        f.write('it does not determine medical truth automatically."\n\n')

        f.write("1. SUMMARY OF SCAN RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Evaluated Tweets          : {len(df):,}\n")
        f.write(f"Flagged for Human Review        : {total_flagged:,}\n")
        f.write(f"Flag Rate                       : {total_flagged/len(df)*100:.4f}%\n\n")

        f.write("2. BREAKDOWN BY QUESTIONABLE CATEGORY\n")
        f.write("-" * 40 + "\n")
        if total_flagged > 0:
            for cat, cnt in cat_counts.items():
                f.write(f"  - {cat:<42}: {cnt:>4,} tweets\n")
        else:
            f.write("  None detected.\n")
        f.write("\n")

        f.write("3. SAMPLE FLAGGED TWEETS & TRIGGER PHRASES\n")
        f.write("-" * 40 + "\n")
        if total_flagged > 0:
            sample_sub = flagged_df.head(5)
            for _, row in sample_sub.iterrows():
                f.write(f"Category: {row['flag_category']}\n")
                f.write(f"Matched Trigger: '{row['flag_matched_pattern']}'\n")
                f.write(f"Tweet: \"{row['original_tweet']}\"\n")
                f.write("-" * 40 + "\n")
        else:
            f.write("  No sample tweets.\n")

        f.write("=" * 80 + "\n")

    print(f" Saved misinformation report to: {report_path}")
    print(" Stage 9: Misinformation Analysis completed successfully!")
    return df, flagged_df


if __name__ == "__main__":
    run_misinformation_analysis()
