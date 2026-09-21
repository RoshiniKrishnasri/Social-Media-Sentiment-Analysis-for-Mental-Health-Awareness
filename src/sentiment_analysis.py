"""
Stage 6: Sentiment Analysis & NLP Model Evaluation
Project: Social Media Sentiment Analysis for Mental Health Awareness

This module performs:
1. TextBlob Sentiment Analysis (Polarity & Subjectivity)
2. NLTK VADER Sentiment Analysis (Compound, Pos, Neg, Neu)
3. Three-class Sentiment Categorization (Positive, Negative, Neutral)
4. Comparative Model Evaluation against Reference Ground-Truth Dataset Labels
5. Confusion Matrices, Classification Metrics (Accuracy, Precision, Recall, F1)
6. Export of sentiment_results.csv, model_comparison.csv, and evaluation charts.

*Ethical Disclaimer: Rule-based and lexical NLP sentiment models provide linguistic
sentiment estimation and do NOT constitute clinical diagnosis of mental health conditions.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_tweets.csv")
OUTPUT_RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
OUTPUT_FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
OUTPUT_REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")

os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_REPORTS_DIR, exist_ok=True)


def compute_vader_sentiment(texts: list[str]) -> tuple[list[float], list[float], list[float], list[float], list[str]]:
    """
    Computes VADER sentiment scores for a list of texts.
    Returns: compound, pos, neg, neu, sentiment_label
    """
    sia = SentimentIntensityAnalyzer()
    compounds = []
    pos_scores = []
    neg_scores = []
    neu_scores = []
    labels = []

    for text in texts:
        if not isinstance(text, str) or not text.strip():
            compounds.append(0.0)
            pos_scores.append(0.0)
            neg_scores.append(0.0)
            neu_scores.append(1.0)
            labels.append("Neutral")
            continue
            
        scores = sia.polarity_scores(text)
        comp = scores["compound"]
        compounds.append(round(comp, 4))
        pos_scores.append(round(scores["pos"], 4))
        neg_scores.append(round(scores["neg"], 4))
        neu_scores.append(round(scores["neu"], 4))

        if comp >= 0.05:
            labels.append("Positive")
        elif comp <= -0.05:
            labels.append("Negative")
        else:
            labels.append("Neutral")

    return compounds, pos_scores, neg_scores, neu_scores, labels


def compute_textblob_sentiment(texts: list[str]) -> tuple[list[float], list[float], list[str]]:
    """
    Computes TextBlob polarity, subjectivity, and predicted label.
    """
    polarities = []
    subjectivities = []
    labels = []

    for text in texts:
        if not isinstance(text, str) or not text.strip():
            polarities.append(0.0)
            subjectivities.append(0.0)
            labels.append("Neutral")
            continue

        blob = TextBlob(text)
        pol = round(blob.sentiment.polarity, 4)
        subj = round(blob.sentiment.subjectivity, 4)
        polarities.append(pol)
        subjectivities.append(subj)

        if pol > 0.05:
            labels.append("Positive")
        elif pol < -0.05:
            labels.append("Negative")
        else:
            labels.append("Neutral")

    return polarities, subjectivities, labels


def plot_confusion_matrices(y_true, y_pred_tb, y_pred_vader, classes=["Positive", "Negative", "Neutral"]):
    """Plots normalized confusion matrices for TextBlob and VADER."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    cm_tb = confusion_matrix(y_true, y_pred_tb, labels=classes, normalize="true")
    cm_vader = confusion_matrix(y_true, y_pred_vader, labels=classes, normalize="true")

    sns.heatmap(cm_tb, annot=True, fmt=".2%", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title("TextBlob vs Dataset Ground Truth (Normalized)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted Sentiment")
    axes[0].set_ylabel("True Dataset Sentiment")

    sns.heatmap(cm_vader, annot=True, fmt=".2%", cmap="Greens", xticklabels=classes, yticklabels=classes, ax=axes[1])
    axes[1].set_title("VADER vs Dataset Ground Truth (Normalized)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted Sentiment")
    axes[1].set_ylabel("True Dataset Sentiment")

    plt.tight_layout()
    path = os.path.join(OUTPUT_FIGURES_DIR, "sentiment_confusion_matrices.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f" Saved: {path}")


def plot_model_metrics(comp_df: pd.DataFrame):
    """Plots comparative accuracy, precision, recall, and F1 bar charts."""
    plt.figure(figsize=(10, 5))
    plot_df = comp_df.melt(id_vars=["Metric"], value_vars=["TextBlob", "VADER"], var_name="Model", value_name="Score")
    
    palette = {"TextBlob": "#3498db", "VADER": "#2ecc71"}
    ax = sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", palette=palette)
    plt.title("NLP Sentiment Model Performance Comparison (Compatible Subset)", fontsize=13, fontweight="bold")
    plt.xlabel("Evaluation Metric", fontsize=11)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=11)
    plt.ylim(0, 1.15)
    
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", fontsize=10, fontweight="semibold", xytext=(0, 3),
                        textcoords="offset points")

    plt.tight_layout()
    path = os.path.join(OUTPUT_FIGURES_DIR, "sentiment_model_comparison_metrics.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f" Saved: {path}")


def run_sentiment_analysis(input_path: str = PROCESSED_DATA_PATH):
    """Executes sentiment analysis and model benchmarking."""
    print("=" * 60)
    print("Executing Stage 6: Sentiment Analysis & Evaluation...")
    print("=" * 60)

    print(f"\n Loading preprocessed dataset from: {input_path}")
    df = pd.read_csv(input_path, encoding="utf-8")
    print(f" Loaded {len(df):,} tweets.")

    # Apply VADER
    print("\n Running VADER Sentiment Intensity Analyzer...")
    vader_comp, vader_pos, vader_neg, vader_neu, vader_labels = compute_vader_sentiment(df["cleaned_text"].tolist())
    df["vader_compound"] = vader_comp
    df["vader_pos"] = vader_pos
    df["vader_neg"] = vader_neg
    df["vader_neu"] = vader_neu
    df["vader_sentiment"] = vader_labels

    # Apply TextBlob
    print(" Running TextBlob Polarity & Subjectivity Analyzer...")
    tb_pol, tb_subj, tb_labels = compute_textblob_sentiment(df["cleaned_text"].tolist())
    df["tb_polarity"] = tb_pol
    df["tb_subjectivity"] = tb_subj
    df["tb_sentiment"] = tb_labels

    # Save full sentiment results
    results_path = os.path.join(OUTPUT_RESULTS_DIR, "sentiment_results.csv")
    print(f"\n Saving annotated sentiment results to: {results_path}")
    df.to_csv(results_path, index=False, encoding="utf-8")

    # Evaluate on compatible 3-class subset (Positive, Negative, Neutral)
    print("\n Evaluating model performance on 3-class compatible subset...")
    compat_mask = df["Sentiment"].isin(["Positive", "Negative", "Neutral"])
    compat_df = df[compat_mask].copy()
    y_true = compat_df["Sentiment"]
    y_pred_tb = compat_df["tb_sentiment"]
    y_pred_vader = compat_df["vader_sentiment"]

    classes = ["Positive", "Negative", "Neutral"]

    # TextBlob metrics
    acc_tb = accuracy_score(y_true, y_pred_tb)
    prec_tb, rec_tb, f1_tb, _ = precision_recall_fscore_support(y_true, y_pred_tb, average="macro")

    # VADER metrics
    acc_vader = accuracy_score(y_true, y_pred_vader)
    prec_vader, rec_vader, f1_vader, _ = precision_recall_fscore_support(y_true, y_pred_vader, average="macro")

    comparison_data = {
        "Metric": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score"],
        "TextBlob": [round(acc_tb, 4), round(prec_tb, 4), round(rec_tb, 4), round(f1_tb, 4)],
        "VADER": [round(acc_vader, 4), round(prec_vader, 4), round(rec_vader, 4), round(f1_vader, 4)]
    }
    comp_df = pd.DataFrame(comparison_data)
    comp_csv_path = os.path.join(OUTPUT_RESULTS_DIR, "model_comparison.csv")
    comp_df.to_csv(comp_csv_path, index=False)
    print(f" Saved model comparison metrics table to: {comp_csv_path}")

    # Generate Confusion Matrices & Metrics Plot
    print("\n Generating confusion matrices and comparison charts...")
    plot_confusion_matrices(y_true, y_pred_tb, y_pred_vader, classes)
    plot_model_metrics(comp_df)

    # Save detailed report
    report_path = os.path.join(OUTPUT_REPORTS_DIR, "sentiment_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("STAGE 6: SENTIMENT ANALYSIS & MODEL EVALUATION REPORT\n")
        f.write("Project: Social Media Sentiment Analysis for Mental Health Awareness\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. EVALUATION SCOPE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Processed Tweets        : {len(df):,}\n")
        f.write(f"Compatible Evaluation Tweets  : {len(compat_df):,} (excluding 'Irrelevant')\n\n")

        f.write("2. MODEL PERFORMANCE BENCHMARK\n")
        f.write("-" * 40 + "\n")
        f.write(comp_df.to_string(index=False) + "\n\n")

        f.write("3. TEXTBLOB DETAILED CLASSIFICATION REPORT\n")
        f.write("-" * 40 + "\n")
        f.write(classification_report(y_true, y_pred_tb, target_names=classes) + "\n")

        f.write("4. VADER DETAILED CLASSIFICATION REPORT\n")
        f.write("-" * 40 + "\n")
        f.write(classification_report(y_true, y_pred_vader, target_names=classes) + "\n")

        f.write("5. SENTIMENT DISTRIBUTION COMPARISON ACROSS MODELS\n")
        f.write("-" * 40 + "\n")
        dist = pd.DataFrame({
            "Dataset Ground Truth": df["Sentiment"].value_counts(),
            "TextBlob Predicted": df["tb_sentiment"].value_counts(),
            "VADER Predicted": df["vader_sentiment"].value_counts()
        }).fillna(0).astype(int)
        f.write(dist.to_string() + "\n\n")

        f.write("=" * 80 + "\n")
        f.write("ETHICAL NOTE:\n")
        f.write("These NLP models analyze lexical valence in tweets. They do NOT evaluate clinical psychiatric\n")
        f.write("state or offer medical diagnosis.\n")
        f.write("=" * 80 + "\n")

    print(f" Saved sentiment analysis report to: {report_path}")
    print(" Stage 6: Sentiment Analysis completed successfully!")
    return df, comp_df


if __name__ == "__main__":
    run_sentiment_analysis()
