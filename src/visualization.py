"""
Stage 10: Centralized Visualization Module
Project: Social Media Sentiment Analysis for Mental Health Awareness

This module contains centralized charting utilities using Matplotlib, Seaborn,
and Plotly to produce figures for reports and interactive dashboard displays.
"""

import os

# NumPy 2.0 compatibility shims for legacy dependencies (xarray, dask)
import numpy as np
if not hasattr(np, "round_"):
    np.round_ = np.round
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_
if not hasattr(np, "string_"):
    np.string_ = np.bytes_

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(OUTPUT_FIGURES_DIR, exist_ok=True)

COLOR_PALETTE = {
    "Positive": "#2ecc71",
    "Negative": "#e74c3c",
    "Neutral": "#3498db",
    "Irrelevant": "#95a5a6",
    "Primary": "#4f46e5",
    "Secondary": "#06b6d4",
    "Background": "#0f172a"
}


def create_plotly_sentiment_donut(df: pd.DataFrame) -> go.Figure:
    """Creates a high-contrast interactive donut chart of sentiment distribution."""
    counts = df["Sentiment"].value_counts()
    colors = [COLOR_PALETTE.get(s, "#888888") for s in counts.index]

    fig = go.Figure(data=[go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color='#ffffff', width=2)),
        textinfo='label+percent',
        hoverinfo='label+value+percent'
    )])
    fig.update_layout(
        title="<b>Sentiment Distribution Across Tweets</b>",
        font=dict(family="Inter, sans-serif", size=13),
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    return fig


def create_plotly_theme_bar(theme_series: pd.Series) -> go.Figure:
    """Creates an interactive horizontal bar chart for mental health themes."""
    fig = px.bar(
        x=theme_series.values,
        y=theme_series.index,
        orientation='h',
        labels={'x': 'Number of Tweets', 'y': 'Mental Health Theme'},
        title="<b>Frequency of Mental Health & Wellbeing Themes</b>",
        color=theme_series.values,
        color_continuous_scale="Viridis"
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(t=40, b=20, l=20, r=20),
        coloraxis_showscale=False
    )
    return fig


def create_plotly_model_comparison(comp_df: pd.DataFrame) -> go.Figure:
    """Creates a grouped bar chart comparing TextBlob and VADER performance."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=comp_df["Metric"],
        y=comp_df["TextBlob"],
        name="TextBlob",
        marker_color="#3498db"
    ))
    fig.add_trace(go.Bar(
        x=comp_df["Metric"],
        y=comp_df["VADER"],
        name="VADER",
        marker_color="#2ecc71"
    ))
    fig.update_layout(
        barmode='group',
        title="<b>NLP Sentiment Model Evaluation Benchmark</b>",
        yaxis=dict(title="Score (0.0 to 1.0)", range=[0, 1.1]),
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def generate_wordcloud_image(text: str, colormap: str = "Blues") -> plt.Figure:
    """Generates a Matplotlib figure containing a Word Cloud."""
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap=colormap,
        max_words=100
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig
