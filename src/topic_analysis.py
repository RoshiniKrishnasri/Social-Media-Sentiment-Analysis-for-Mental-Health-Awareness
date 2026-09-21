"""
Stage 8: Topic Modeling Module (Latent Dirichlet Allocation)
Project: Social Media Sentiment Analysis for Mental Health Awareness

This module applies Latent Dirichlet Allocation (LDA) via Scikit-Learn:
1. Vectorizes preprocessed tweet tokens with CountVectorizer (Bag-of-Words).
2. Fits an LDA model to discover hidden thematic structures.
3. Fits a specialized sub-model on mental-health-related tweets to extract fine-grained psychological themes.
4. Assigns dominant topic IDs and topic labels to tweets.
5. Produces topic-word distribution bar charts and exports topic keywords and distributions.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DATA_PATH = os.path.join(BASE_DIR, "outputs", "results", "mental_health_themes.csv")
OUTPUT_RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
OUTPUT_FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
OUTPUT_REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")

os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_REPORTS_DIR, exist_ok=True)

NUM_TOPICS = 6


def get_top_topic_words(model: LatentDirichletAllocation, feature_names: list[str], n_top_words: int = 10) -> dict:
    """Extracts top words and their weight for each discovered topic."""
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        top_weights = [topic[i] for i in top_indices]
        topic_dict[f"Topic_{topic_idx + 1}"] = list(zip(top_words, top_weights))
    return topic_dict


def plot_topic_words(topic_dict: dict, title_suffix: str = "", filename: str = "topic_modeling_word_distributions.png"):
    """Plots top 8 words for each topic in a clean multi-panel subplot."""
    num_topics = len(topic_dict)
    cols = 3
    rows = (num_topics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), sharex=False)
    axes = axes.flatten()

    for idx, (topic_name, word_weight_pairs) in enumerate(topic_dict.items()):
        words, weights = zip(*word_weight_pairs[:8])
        ax = axes[idx]
        palette = sns.color_palette("Blues_r", len(words))
        sns.barplot(x=list(weights), y=list(words), palette=palette, ax=ax)
        ax.set_title(f"{topic_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("LDA Component Weight", fontsize=10)
        ax.tick_params(labelsize=10)

    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"LDA Discovered Topics - Top Keyword Distribution {title_suffix}", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_FIGURES_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {path}")


def run_topic_modeling(input_path: str = INPUT_DATA_PATH):
    """Executes topic modeling on the corpus and mental health subset."""
    print("=" * 60)
    print("Executing Stage 8: Topic Modeling (LDA)...")
    print("=" * 60)

    print(f"\n Loading dataset from: {input_path}")
    df = pd.read_csv(input_path, encoding="utf-8")
    print(f" Loaded {len(df):,} tweets.")

    # 1. Corpus-Wide Topic Modeling
    print("\n Vectorizing preprocessed texts for Global LDA...")
    valid_texts = df["processed_text"].fillna("").astype(str)
    
    vectorizer = CountVectorizer(max_df=0.85, min_df=10, max_features=2500, stop_words="english")
    dtm = vectorizer.fit_transform(valid_texts)
    vocab = vectorizer.get_feature_names_out()

    print(f" Document-Term Matrix: {dtm.shape[0]:,} docs x {dtm.shape[1]} features.")
    print(" Training Latent Dirichlet Allocation model (6 topics)...")
    lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, random_state=42, max_iter=15, learning_method="online")
    lda_topic_matrix = lda_model.fit_transform(dtm)

    # Assign dominant topic
    df["dominant_topic_id"] = lda_topic_matrix.argmax(axis=1) + 1
    df["topic_confidence"] = np.round(lda_topic_matrix.max(axis=1), 4)

    # Extract keywords
    top_words = get_top_topic_words(lda_model, vocab, n_top_words=10)

    # Label topics semantically based on top keywords
    topic_labels_map = {
        "Topic_1": "Digital Culture & Media",
        "Topic_2": "Gaming Performance & Play",
        "Topic_3": "Customer Experience & Support",
        "Topic_4": "Community & Social Interaction",
        "Topic_5": "Frustration & Bug Complaints",
        "Topic_6": "Product Updates & Announcements"
    }
    df["dominant_topic_label"] = df["dominant_topic_id"].apply(lambda x: topic_labels_map.get(f"Topic_{x}", f"Topic {x}"))

    # Plot global topic words
    plot_topic_words(top_words, "(Overall Corpus)", "topic_modeling_word_distributions.png")

    # 2. Mental Health Subset Topic Modeling
    print("\n Training specialized LDA on Mental-Health Tweets (N=2,144)...")
    mh_mask = df["is_mental_health_related"]
    mh_texts = df.loc[mh_mask, "processed_text"].fillna("").astype(str)
    
    mh_vectorizer = CountVectorizer(max_df=0.85, min_df=3, max_features=1200, stop_words="english")
    mh_dtm = mh_vectorizer.fit_transform(mh_texts)
    mh_vocab = mh_vectorizer.get_feature_names_out()

    mh_lda = LatentDirichletAllocation(n_components=4, random_state=42, max_iter=15, learning_method="online")
    mh_lda.fit(mh_dtm)
    mh_top_words = get_top_topic_words(mh_lda, mh_vocab, n_top_words=10)

    plot_topic_words(mh_top_words, "(Mental Health Focus)", "topic_modeling_mental_health_subtopics.png")

    # Save output CSV
    results_path = os.path.join(OUTPUT_RESULTS_DIR, "topic_model_results.csv")
    df.to_csv(results_path, index=False, encoding="utf-8")
    print(f"\n Saved tweets with dominant topics to: {results_path}")

    # Save keywords CSV
    topic_keywords_rows = []
    for t_name, pairs in top_words.items():
        words_only = [w for w, _ in pairs]
        topic_keywords_rows.append({
            "Topic": t_name,
            "Label": topic_labels_map.get(t_name, t_name),
            "Top_Keywords": ", ".join(words_only)
        })
    keywords_df = pd.DataFrame(topic_keywords_rows)
    keywords_path = os.path.join(OUTPUT_RESULTS_DIR, "lda_topic_words.csv")
    keywords_df.to_csv(keywords_path, index=False)

    # Plot topic distribution
    plt.figure(figsize=(10, 5))
    counts = df["dominant_topic_label"].value_counts()
    sns.barplot(x=counts.values, y=counts.index, palette="mako")
    plt.title("Discovered Topic Distribution Across Dataset", fontsize=13, fontweight="bold")
    plt.xlabel("Tweet Count", fontsize=11)
    plt.ylabel("Thematic Topic", fontsize=11)
    plt.tight_layout()
    dist_path = os.path.join(OUTPUT_FIGURES_DIR, "topic_modeling_distribution.png")
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f" Saved: {dist_path}")

    # Write report
    report_path = os.path.join(OUTPUT_REPORTS_DIR, "topic_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("STAGE 8: TOPIC MODELING (LDA) REPORT\n")
        f.write("Project: Social Media Sentiment Analysis for Mental Health Awareness\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. MODEL CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write("Algorithm: Latent Dirichlet Allocation (Scikit-Learn)\n")
        f.write(f"Global Topics Discovered: {NUM_TOPICS}\n")
        f.write("Mental-Health Focused Sub-Topics: 4\n")
        f.write("Feature Representation: Bag-of-Words (CountVectorizer)\n\n")

        f.write("2. GLOBAL TOPICS & TOP KEYWORDS\n")
        f.write("-" * 40 + "\n")
        for _, row in keywords_df.iterrows():
            f.write(f"[{row['Topic']} - {row['Label']}]\n")
            f.write(f"  Keywords: {row['Top_Keywords']}\n\n")

        f.write("3. MENTAL-HEALTH SUBSET TOPICS\n")
        f.write("-" * 40 + "\n")
        for t_name, pairs in mh_top_words.items():
            words_only = [w for w, _ in pairs]
            f.write(f"[{t_name}]: {', '.join(words_only)}\n")
        f.write("\n")

        f.write("4. DOMINANT TOPIC COUNTS ACROSS DATASET\n")
        f.write("-" * 40 + "\n")
        for label, cnt in counts.items():
            f.write(f"  - {label:<35}: {cnt:>6,} ({cnt/len(df)*100:.2f}%)\n")
        f.write("\n")

        f.write("=" * 80 + "\n")

    print(f" Saved topic analysis report to: {report_path}")
    print(" Stage 8: Topic Modeling completed successfully!")
    return df, keywords_df


if __name__ == "__main__":
    run_topic_modeling()
