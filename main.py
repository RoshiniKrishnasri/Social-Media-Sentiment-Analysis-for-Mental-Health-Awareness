"""
Master Pipeline Orchestrator
Project: Social Media Sentiment Analysis for Mental Health Awareness

This script executes the complete end-to-end pipeline:
1. Data Understanding & Inspection
2. Data Cleaning & Deduplication
3. NLP Text Preprocessing (Lemmatization & Stopwords)
4. Exploratory Data Analysis (EDA) & Word Clouds
5. Sentiment Analysis (TextBlob vs VADER) & Model Benchmarking
6. Mental-Health Theme Identification & Thematic Cross-Tabulation
7. Topic Modeling (Latent Dirichlet Allocation)
8. Misinformation & Questionable Health Claims Flagging
9. Dynamic Insights Generation
"""

import os
import sys
import time

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.data_understanding import run_data_understanding
from src.data_cleaning import clean_dataset
from src.preprocessing import run_preprocessing
from src.eda import run_eda
from src.sentiment_analysis import run_sentiment_analysis
from src.mental_health_analysis import run_mental_health_analysis
from src.topic_analysis import run_topic_modeling
from src.misinformation_analysis import run_misinformation_analysis
from src.insights_generator import generate_insights


def main():
    start_time = time.time()
    print("=" * 80)
    print(" SOCIAL MEDIA SENTIMENT ANALYSIS FOR MENTAL HEALTH AWARENESS")
    print(" End-to-End Pipeline Execution")
    print("=" * 80)

    # Stage 2: Data Understanding
    print("\n>>> STAGE 2: DATA UNDERSTANDING")
    run_data_understanding()

    # Stage 3: Data Cleaning
    print("\n>>> STAGE 3: DATA CLEANING & DEDUPLICATION")
    clean_dataset()

    # Stage 5: NLP Preprocessing
    print("\n>>> STAGE 5: NLP TEXT PREPROCESSING")
    run_preprocessing()

    # Stage 4: EDA
    print("\n>>> STAGE 4: EXPLORATORY DATA ANALYSIS (EDA)")
    run_eda()

    # Stage 6: Sentiment Analysis & Evaluation
    print("\n>>> STAGE 6: SENTIMENT ANALYSIS (TEXTBLOB & VADER)")
    run_sentiment_analysis()

    # Stage 7: Mental Health Theme Analysis
    print("\n>>> STAGE 7: MENTAL-HEALTH THEME ANALYSIS")
    run_mental_health_analysis()

    # Stage 8: Topic Modeling (LDA)
    print("\n>>> STAGE 8: TOPIC MODELING (LDA)")
    run_topic_modeling()

    # Stage 9: Misinformation & Questionable Content Flagging
    print("\n>>> STAGE 9: MISINFORMATION & QUESTIONABLE CLAIM ANALYSIS")
    run_misinformation_analysis()

    # Stage 12: Automated Insights
    print("\n>>> STAGE 12: AUTOMATED DATA-DRIVEN INSIGHTS GENERATION")
    findings = generate_insights()

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f" PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.1f} SECONDS")
    print("=" * 80)
    print("\nAll artifacts generated:")
    print("  • Cleaned dataset:        data/processed/cleaned_tweets.csv")
    print("  • Processed NLP dataset:  data/processed/processed_tweets.csv")
    print("  • Final Analysis dataset: outputs/results/final_analysis_dataset.csv")
    print("  • Model comparison table: outputs/results/model_comparison.csv")
    print("  • Visualizations saved:   outputs/figures/ (12 high-resolution charts)")
    print("  • Analytical reports:     outputs/reports/ (6 stage summary reports)")
    print("\nTo launch the interactive dashboard, run:")
    print("    streamlit run app/dashboard.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
