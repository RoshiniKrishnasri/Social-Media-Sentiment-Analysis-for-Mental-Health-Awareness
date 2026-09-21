"""
Social Media Sentiment Analysis for Mental Health Awareness
MindPulse Analytics — Interactive Enterprise Analytics Dashboard
(With Full Global Dark/Light Background & Container Theme Switching)
"""

import os, sys, io, base64, re, time

# ── NumPy 2.0 shims ──────────────────────────────────────────────────────────
import numpy as np
for _a, _b in [("round_","round"),("unicode_","str_"),("string_","bytes_"),
               ("bool","bool_"),("int","int_"),("float","float64"),
               ("complex","complex128"),("object","object_")]:
    if not hasattr(np, _a):
        try: setattr(np, _a, getattr(np, _b))
        except Exception: pass

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH      = os.path.join(BASE_DIR, "outputs", "results", "final_analysis_dataset.csv")
MODEL_COMP     = os.path.join(BASE_DIR, "outputs", "results", "model_comparison.csv")
FIGURES_DIR    = os.path.join(BASE_DIR, "outputs", "figures")
sys.path.insert(0, BASE_DIR)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindPulse Analytics — Mental Health Sentiment Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State Navigation & Theme ─────────────────────────────────────────
NAV_PAGES = [
    "🏠 Home",
    "📊 Dashboard",
    "🎭 Sentiment",
    "🧠 Mental Health Themes",
    "🔍 Topic Modeling",
    "🛡️ Safety Flags",
    "🧪 Live Playground",
    "📄 Reports & Export",
    "ℹ️ About"
]

if "page" not in st.session_state:
    st.session_state["page"] = "🏠 Home"

if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Light"

if "playground_input" not in st.session_state:
    st.session_state["playground_input"] = ""

def set_page(p_name):
    st.session_state["page"] = p_name
    st.rerun()

def toggle_theme():
    st.session_state["theme_mode"] = "Dark" if st.session_state["theme_mode"] == "Light" else "Light"
    st.rerun()

is_dark = (st.session_state["theme_mode"] == "Dark")

# ── Theme Colors & Palette Definitions ───────────────────────────────────────
if is_dark:
    BODY_BG       = "linear-gradient(135deg, #05070b 0%, #101827 45%, #1d2a3a 100%)"
    CARD_BG       = "#0F172A"
    CARD_BORDER   = "#1E293B"
    TEXT_MAIN     = "#F8FAFC"
    TEXT_MUTED    = "#94A3B8"
    SIDEBAR_BG    = "#070A10"
    SIDEBAR_BORDER= "#1E293B"
    INPUT_BG      = "#1E293B"
    PLOT_BG       = "#0F172A"
    TWEET_BG      = "#1E293B"
    TOPNAV_BG     = "#0F172A"
    TOPNAV_BORDER = "#1E293B"
    BUTTON_BG     = "linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)"
    BUTTON_TEXT   = "#FFFFFF"
else:
    BODY_BG       = "#F4F6FB"
    CARD_BG       = "#ffffff"
    CARD_BORDER   = "#E5E7EB"
    TEXT_MAIN     = "#111827"
    TEXT_MUTED    = "#6B7280"
    SIDEBAR_BG    = "#ffffff"
    SIDEBAR_BORDER= "#E8ECF4"
    INPUT_BG      = "#ffffff"
    PLOT_BG       = "#ffffff"
    TWEET_BG      = "#ffffff"
    TOPNAV_BG     = "#ffffff"
    TOPNAV_BORDER = "#E5E7EB"
    BUTTON_BG     = "linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%)"
    BUTTON_TEXT   = "#FFFFFF"

PALETTE = {"Positive":"#10B981","Negative":"#EF4444","Neutral":"#3B82F6","Irrelevant":"#9CA3AF"}

# ── Dynamic Full-Page CSS Injection ──────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Target ALL Streamlit Root Containers ── */
html, body, .stApp, 
[data-testid="stAppViewContainer"], 
[data-testid="stHeader"], 
[data-testid="stMain"], 
section.main, 
.block-container {{
    background: {BODY_BG} !important;
    background-attachment: fixed !important;
    color: {TEXT_MAIN} !important;
    font-family: 'Inter', sans-serif !important;
    transition: background 0.3s ease, color 0.3s ease;
}}

#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 1rem 2rem 2rem !important; max-width: 1440px; }}

/* Typography & General Contrast Rules */
h1, h2, h3, h4, h5, h6 {{
    color: {TEXT_MAIN} !important;
    font-family: 'Inter', sans-serif !important;
}}
p, [data-testid="stMarkdownContainer"] > p {{
    color: {TEXT_MAIN} !important;
}}
.stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p, [data-testid="stCaptionContainer"] span {{
    color: {TEXT_MUTED} !important;
}}

/* ── Plotly SVG High-Contrast Text Rules for Dark/Light Mode ── */
.js-plotly-plot .plotly .gtitle {{
    fill: {TEXT_MAIN} !important;
    font-weight: 700 !important;
}}
.js-plotly-plot .plotly .xtitle,
.js-plotly-plot .plotly .ytitle {{
    fill: {TEXT_MAIN} !important;
    font-weight: 600 !important;
}}
.js-plotly-plot .plotly text.legendtext,
.js-plotly-plot .plotly .legendtext {{
    fill: {TEXT_MAIN} !important;
    font-weight: 500 !important;
}}
.js-plotly-plot .plotly text.legendtitletext,
.js-plotly-plot .plotly .legendtitletext {{
    fill: {TEXT_MAIN} !important;
    font-weight: 600 !important;
}}
.js-plotly-plot .plotly .xaxislayer-above text,
.js-plotly-plot .plotly .yaxislayer-above text {{
    fill: {TEXT_MAIN} !important;
}}

/* Metrics Styling in Dark & Light Modes */
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] span {{
    color: {TEXT_MUTED} !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] p, [data-testid="stMetricValue"] div {{
    color: {TEXT_MAIN} !important;
    font-weight: 800 !important;
}}
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] p, [data-testid="stMetricDelta"] svg {{
    color: {"#34D399" if is_dark else "#059669"} !important;
}}

/* All Buttons (Standard, Primary, Secondary, Download) */
.stButton > button,
.stDownloadButton > button,
button[data-testid="stBaseButton-secondary"],
button[data-testid="stBaseButton-primary"],
button[kind="primary"],
button[kind="secondary"] {{
    background: {BUTTON_BG} !important;
    color: {BUTTON_TEXT} !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 22px !important;
    font-weight: 700 !important;
    font-size: 13.5px !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
}}
.stButton > button *,
.stDownloadButton > button *,
button[data-testid="stBaseButton-secondary"] *,
button[data-testid="stBaseButton-primary"] * {{
    color: {BUTTON_TEXT} !important;
    font-weight: 700 !important;
}}
.stButton > button:hover,
.stDownloadButton > button:hover {{
    opacity: 0.92 !important;
    box-shadow: 0 6px 18px rgba(99,102,241,0.5) !important;
}}

/* ── TOP NAVIGATION BUTTONS: STRICT UNIFORM SIZE, SHAPE & ALIGNMENT ── */
div[class*="st-key-topnav_"] {{
    display: flex !important;
    align-items: stretch !important;
    justify-content: center !important;
    height: 52px !important;
    width: 100% !important;
}}

div[class*="st-key-topnav_"] > button,
div[class*="st-key-topnav_"] button {{
    width: 100% !important;
    height: 52px !important;
    min-height: 52px !important;
    max-height: 52px !important;
    padding: 6px 4px !important;
    border-radius: 12px !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    box-sizing: border-box !important;
    font-size: 11.5px !important;
    font-weight: 700 !important;
    line-height: 1.25 !important;
    margin: 0 !important;
    overflow: hidden !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}}

div[class*="st-key-topnav_"] button div[data-testid="stMarkdownContainer"],
div[class*="st-key-topnav_"] button div[data-testid="stMarkdownContainer"] p,
div[class*="st-key-topnav_"] button div[data-testid="stMarkdownContainer"] span {{
    font-size: 11.5px !important;
    font-weight: 700 !important;
    line-height: 1.25 !important;
    text-align: center !important;
    margin: 0 !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    word-break: normal !important;
    overflow-wrap: normal !important;
}}

/* Active Top Navigation Box */
div[class*="st-key-topnav_"] button[data-testid="baseButton-primary"],
div[class*="st-key-topnav_"] button[kind="primary"] {{
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
    border: 2px solid #C7D2FE !important;
    box-shadow: 0 0 16px rgba(99,102,241,0.55), 0 4px 14px rgba(0,0,0,0.25) !important;
    transform: translateY(-1px) !important;
}}

/* Inactive Top Navigation Boxes */
div[class*="st-key-topnav_"] button[data-testid="baseButton-secondary"],
div[class*="st-key-topnav_"] button[kind="secondary"] {{
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12) !important;
    opacity: 0.92 !important;
}}

div[class*="st-key-topnav_"] button[data-testid="baseButton-secondary"]:hover {{
    opacity: 1 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.45) !important;
}}

/* Input Labels & Form Elements */
label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {{
    color: {TEXT_MAIN} !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}}
.stTextArea textarea, 
.stTextInput input, 
div[data-baseweb="input"] input, 
div[data-baseweb="base-input"] textarea {{
    background-color: {INPUT_BG} !important;
    color: {TEXT_MAIN} !important;
    border: 1.5px solid {CARD_BORDER} !important;
    border-radius: 10px !important;
}}
.stTextArea textarea::placeholder, 
.stTextInput input::placeholder {{
    color: {TEXT_MUTED} !important;
    opacity: 0.8 !important;
}}

/* BaseWeb Selectbox & MultiSelect Dropdown Menus */
div[data-baseweb="select"] > div {{
    background-color: {INPUT_BG} !important;
    color: {TEXT_MAIN} !important;
    border-color: {CARD_BORDER} !important;
    border-radius: 10px !important;
}}
div[data-baseweb="select"] span {{
    color: {TEXT_MAIN} !important;
}}
div[data-baseweb="popover"], 
ul[role="listbox"], 
li[role="option"] {{
    background-color: {CARD_BG} !important;
    color: {TEXT_MAIN} !important;
}}
li[role="option"]:hover, li[aria-selected="true"] {{
    background-color: {"#1E293B" if is_dark else "#EEF2FF"} !important;
    color: {"#818CF8" if is_dark else "#4338CA"} !important;
}}

/* Sliders, Toggles & Radios */
[data-testid="stSlider"] p, [data-testid="stToggle"] p, [data-testid="stToggle"] span {{
    color: {TEXT_MAIN} !important;
}}

/* DataFrames & Tables */
[data-testid="stTable"], [data-testid="stDataFrame"], div[data-testid="stTable"] table {{
    background-color: {CARD_BG} !important;
    color: {TEXT_MAIN} !important;
    border-color: {CARD_BORDER} !important;
}}
.stTable th, .stTable td {{
    color: {TEXT_MAIN} !important;
    border-color: {CARD_BORDER} !important;
}}

/* Top Navbar Bar */
.top-navbar {{
    background: {TOPNAV_BG};
    border: 1px solid {TOPNAV_BORDER};
    border-top: 4px solid #6366F1;
    padding: 14px 24px;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,{"0.4" if is_dark else "0.03"});
}}
.nav-brand {{ display: flex; align-items: center; gap: 14px; }}
.nav-title {{ font-size: 22px; font-weight: 800; color: {TEXT_MAIN}; margin: 0; letter-spacing: -0.5px; line-height: 1.2; }}
.nav-subtitle {{ font-size: 12.5px; color: #6366F1; font-weight: 600; margin: 2px 0 0; }}
.nav-right {{ display: flex; align-items: center; gap: 12px; }}
.nav-status {{ background: {"#064E3B" if is_dark else "#ECFDF5"}; color: {"#A7F3D0" if is_dark else "#059669"}; font-size: 12px; font-weight: 700; padding: 6px 14px; border-radius: 20px; border: 1px solid {"#047857" if is_dark else "#A7F3D0"}; }}
.nav-version {{ background: {"#1E1B4B" if is_dark else "#EEF2FF"}; color: {"#C7D2FE" if is_dark else "#4338CA"}; font-size: 12px; font-weight: 700; padding: 6px 14px; border-radius: 20px; border: 1px solid {"#3730A3" if is_dark else "#C7D2FE"}; }}

/* Sidebar Container */
section[data-testid="stSidebar"] {{
    background: {SIDEBAR_BG} !important;
    border-right: 1px solid {SIDEBAR_BORDER} !important;
    box-shadow: 4px 0 20px rgba(0,0,0,{"0.4" if is_dark else "0.04"}) !important;
}}
section[data-testid="stSidebar"] > div:first-child {{ padding: 1.5rem 1rem; }}
section[data-testid="stSidebar"] * {{ color: {TEXT_MAIN} !important; }}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label {{ color: {TEXT_MUTED} !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.05em; }}

/* Sidebar brand card */
.sidebar-brand {{
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%);
    border-radius: 14px;
    padding: 18px 16px;
    margin-bottom: 20px;
    text-align: center;
}}
.sidebar-brand h2 {{ color: #fff !important; font-size: 20px; font-weight: 800; margin: 0; letter-spacing: -0.5px; }}
.sidebar-brand p  {{ color: rgba(255,255,255,0.85) !important; font-size: 11px; margin: 4px 0 0; }}

/* Hero Banner */
.hero-banner {{
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #C084FC 100%);
    border-radius: 20px;
    padding: 40px 36px;
    color: #ffffff;
    margin-bottom: 28px;
    box-shadow: 0 12px 32px rgba(99,102,241,0.25);
}}
.hero-banner h1 {{ color: #ffffff !important; font-size: 34px; font-weight: 800; margin: 0 0 12px; letter-spacing: -0.5px; }}
.hero-banner p  {{ color: rgba(255,255,255,0.92) !important; font-size: 16px; margin: 0 0 24px; line-height: 1.6; max-width: 950px; }}
.hero-badges    {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px; }}
.hero-badge-item {{ background: rgba(255,255,255,0.2); backdrop-filter: blur(10px); padding: 6px 16px; border-radius: 30px; font-size: 12.5px; font-weight: 600; color: #fff; }}

/* Feature Cards */
.feature-card {{
    background: {CARD_BG};
    border-radius: 16px;
    padding: 24px 20px;
    box-shadow: 0 2px 14px rgba(0,0,0,{"0.3" if is_dark else "0.04"});
    border: 1px solid {CARD_BORDER};
    height: 100%;
    display: flex; flex-direction: column; justify-content: space-between;
    transition: transform .2s ease, background 0.3s ease;
}}
.feature-card:hover {{ transform: translateY(-4px); }}
.feature-icon {{ font-size: 32px; margin-bottom: 12px; }}
.feature-title {{ font-size: 18px; font-weight: 700; color: {TEXT_MAIN}; margin-bottom: 8px; }}
.feature-desc {{ font-size: 13.5px; color: {TEXT_MUTED}; line-height: 1.55; margin-bottom: 16px; }}

/* Workflow Card */
.wf-card {{
    background: {CARD_BG};
    border-radius: 14px;
    padding: 20px 16px;
    border: 1px solid {CARD_BORDER};
    height: 100%;
}}
.wf-num {{
    background: {"#312E81" if is_dark else "#EEF2FF"};
    color: {"#C7D2FE" if is_dark else "#4338CA"};
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 14px; margin-bottom: 12px;
}}
.wf-title {{ font-size: 15px; font-weight: 700; color: {TEXT_MAIN}; margin-bottom: 4px; }}
.wf-desc  {{ font-size: 12.5px; color: {TEXT_MUTED}; line-height: 1.5; }}

/* Stat Cards */
.stat-card {{
    background: {CARD_BG};
    border-radius: 16px;
    padding: 20px 18px 16px;
    box-shadow: 0 2px 14px rgba(0,0,0,{"0.3" if is_dark else "0.06"});
    border: 1px solid {CARD_BORDER};
    height: 125px;
    display: flex; flex-direction: column; justify-content: space-between;
    transition: transform .2s ease;
    position: relative; overflow: hidden;
}}
.stat-card:hover {{ transform: translateY(-3px); }}
.stat-card::before {{
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 4px;
    border-radius: 16px 16px 0 0;
}}
.card-green::before  {{ background: linear-gradient(90deg,#10B981,#34D399); }}
.card-blue::before   {{ background: linear-gradient(90deg,#3B82F6,#60A5FA); }}
.card-purple::before {{ background: linear-gradient(90deg,#8B5CF6,#A78BFA); }}
.card-orange::before {{ background: linear-gradient(90deg,#F59E0B,#FCD34D); }}
.card-red::before    {{ background: linear-gradient(90deg,#EF4444,#F87171); }}
.card-pink::before   {{ background: linear-gradient(90deg,#EC4899,#F472B6); }}
.stat-icon   {{ font-size: 20px; }}
.stat-val    {{ font-size: 28px; font-weight: 800; color: {TEXT_MAIN}; line-height: 1; }}
.stat-label  {{ font-size: 11px; font-weight: 600; color: {TEXT_MUTED}; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px; }}

/* Section headers */
.sec-head {{
    font-size: 17px; font-weight: 700; color: {TEXT_MAIN};
    margin: 28px 0 14px;
    display: flex; align-items: center; gap: 8px;
}}
.sec-head::after {{
    content: ""; flex: 1; height: 1px; background: {CARD_BORDER}; margin-left: 10px;
}}

/* Tweet cards */
.tweet-card {{
    background: {TWEET_BG};
    border: 1px solid {CARD_BORDER};
    border-left: 4px solid #6366F1;
    border-radius: 10px;
    padding: 14px 16px;
    margin: 8px 0;
    font-size: 13.5px; color: {TEXT_MAIN}; line-height: 1.6;
    box-shadow: 0 1px 6px rgba(0,0,0,{"0.2" if is_dark else "0.03"});
}}
.tweet-card.neg {{ border-left-color: #EF4444; }}
.tweet-card.pos {{ border-left-color: #10B981; }}
.tweet-card.neu {{ border-left-color: #3B82F6; }}

/* Badges */
.badge {{
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.04em;
}}
.badge-pos {{ background: {"#064E3B" if is_dark else "#D1FAE5"}; color: {"#A7F3D0" if is_dark else "#065F46"}; }}
.badge-neg {{ background: {"#7F1D1D" if is_dark else "#FEE2E2"}; color: {"#FCA5A5" if is_dark else "#991B1B"}; }}
.badge-neu {{ background: {"#1E3A8A" if is_dark else "#DBEAFE"}; color: {"#BFDBFE" if is_dark else "#1E40AF"}; }}
.badge-irr {{ background: {"#374151" if is_dark else "#F3F4F6"}; color: {"#9CA3AF" if is_dark else "#6B7280"}; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px; background: {"#1E293B" if is_dark else "#F4F6FB"}; padding: 4px; border-radius: 10px; margin-bottom: 16px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px !important; padding: 8px 18px !important;
    font-size: 13px !important; font-weight: 600 !important;
    color: {TEXT_MUTED} !important; background: transparent !important; border: none !important;
}}
.stTabs [aria-selected="true"] {{
    background: {CARD_BG} !important; color: #6366F1 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,{"0.4" if is_dark else "0.08"}) !important;
}}

/* Radio nav pills */
div[data-testid="stRadio"] label {{
    background: {"#1E293B" if is_dark else "#F9FAFB"} !important; border-radius: 8px !important;
    border: 1px solid {CARD_BORDER} !important; padding: 10px 14px !important;
    margin-bottom: 4px !important; cursor: pointer; font-size: 13.5px !important; font-weight: 500 !important;
}}
div[data-testid="stRadio"] [aria-checked="true"] + div label {{
    background: {"#312E81" if is_dark else "#EEF2FF"} !important; border-color: #6366F1 !important; color: {"#A5B4FC" if is_dark else "#4338CA"} !important;
}}

/* Page title */
.page-title {{ font-size: 26px; font-weight: 800; color: {TEXT_MAIN}; letter-spacing: -0.5px; margin-bottom: 4px; }}
.page-subtitle {{ font-size: 14px; color: {TEXT_MUTED}; margin-bottom: 24px; }}

/* Alerts */
.info-box {{
    background: {"#1E3A8A" if is_dark else "#EFF6FF"}; border: 1px solid {"#3B82F6" if is_dark else "#BFDBFE"};
    border-left: 4px solid #3B82F6; border-radius: 10px; padding: 14px 18px; color: {"#93C5FD" if is_dark else "#1E40AF"};
    font-size: 13.5px; margin-bottom: 18px; line-height: 1.5;
}}
.warn-box {{
    background: {"#78350F" if is_dark else "#FFF7ED"}; border: 1px solid {"#F59E0B" if is_dark else "#FED7AA"};
    border-left: 4px solid #F59E0B; border-radius: 10px; padding: 14px 18px; color: {"#FDE68A" if is_dark else "#92400E"};
    font-size: 13.5px; margin-bottom: 18px; line-height: 1.5;
}}
.danger-box {{
    background: {"#7F1D1D" if is_dark else "#FEF2F2"}; border: 1px solid {"#EF4444" if is_dark else "#FCA5A5"};
    border-left: 4px solid #EF4444; border-radius: 10px; padding: 14px 18px; color: {"#FCA5A5" if is_dark else "#991B1B"};
    font-size: 13.5px; margin-bottom: 18px; line-height: 1.5;
}}
.about-card {{
    background: {CARD_BG}; border-radius: 16px; padding: 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,{"0.3" if is_dark else "0.05"}); border: 1px solid {CARD_BORDER}; margin-bottom: 20px;
}}
.about-card h1, .about-card h2, .about-card h3, .about-card h4 {{
    color: {TEXT_MAIN} !important;
}}
.about-card p, .about-card li {{
    color: {TEXT_MUTED} !important;
}}
.topic-card {{
    background: {CARD_BG}; border-radius: 12px; border: 1px solid {CARD_BORDER};
    padding: 16px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,{"0.2" if is_dark else "0.02"});
}}
.app-footer {{
    background: {CARD_BG}; border-top: 1px solid {CARD_BORDER}; padding: 22px 28px;
    border-radius: 16px; margin-top: 40px; display: flex; flex-wrap: wrap;
    align-items: center; justify-content: space-between; gap: 16px; font-size: 13px; color: {TEXT_MUTED};
}}
.app-footer a {{ color: #6366F1; text-decoration: none; font-weight: 600; }}
.footer-tag {{ background: {"#1E293B" if is_dark else "#EEF2FF"}; color: {"#A5B4FC" if is_dark else "#4338CA"}; padding: 4px 12px; border-radius: 14px; font-size: 11.5px; font-weight: 600; }}
.footer-contact-link {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: {"#1E293B" if is_dark else "#EEF2FF"};
    color: {"#93C5FD" if is_dark else "#4338CA"} !important;
    border: 1px solid {"#334155" if is_dark else "#C7D2FE"};
    padding: 5px 13px;
    border-radius: 18px;
    font-size: 12px;
    font-weight: 600;
    text-decoration: none !important;
    transition: all 0.2s ease;
}}
.footer-contact-link:hover {{
    background: {"#334155" if is_dark else "#E0E7FF"};
    border-color: {"#60A5FA" if is_dark else "#6366F1"};
    transform: translateY(-1px);
}}
</style>
""", unsafe_allow_html=True)

# ── Dynamic Plotly Helper ────────────────────────────────────────────────────
def styled_fig(fig, height=380):
    fig.update_layout(
        template="plotly_dark" if is_dark else "plotly_white",
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PLOT_BG,
        font=dict(family="Inter", color=TEXT_MAIN, size=12),
        title_font_color=TEXT_MAIN,
        title_font_family="Inter",
        title_font_size=15,
        legend_font_color=TEXT_MAIN,
        legend_title_font_color=TEXT_MAIN,
        height=height
    )
    fig.update_xaxes(
        title_font_color=TEXT_MAIN,
        title_font_family="Inter",
        tickfont_color=TEXT_MAIN,
        tickfont_family="Inter",
        gridcolor="#1E293B" if is_dark else "#E5E7EB",
        zerolinecolor="#334155" if is_dark else "#CBD5E1"
    )
    fig.update_yaxes(
        title_font_color=TEXT_MAIN,
        title_font_family="Inter",
        tickfont_color=TEXT_MAIN,
        tickfont_family="Inter",
        gridcolor="#1E293B" if is_dark else "#E5E7EB",
        zerolinecolor="#334155" if is_dark else "#CBD5E1"
    )
    if hasattr(fig.layout, "coloraxis") and fig.layout.coloraxis:
        fig.update_layout(
            coloraxis_colorbar=dict(
                title_font_color=TEXT_MAIN,
                tickfont_color=TEXT_MAIN
            )
        )
    return fig

def badge(s):
    cls = {"Positive":"pos","Negative":"neg","Neutral":"neu","Irrelevant":"irr"}.get(s,"neu")
    return f'<span class="badge badge-{cls}">{s}</span>'

def tweet_cls(s):
    return {"Positive":"pos","Negative":"neg","Neutral":"neu"}.get(s,"")

def wc_b64(text, color="viridis"):
    if not text.strip():
        return None
    bg = "#0F172A" if is_dark else "#ffffff"
    wc = WordCloud(width=900, height=400, background_color=bg,
                   colormap=color, max_words=120, prefer_horizontal=0.85).generate(text)
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def stat_card(val, label, icon, color_class):
    return f"""
    <div class="stat-card {color_class}">
        <div class="stat-icon">{icon}</div>
        <div>
            <div class="stat-val">{val}</div>
            <div class="stat-label">{label}</div>
        </div>
    </div>"""

# ── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading MindPulse dataset…")
def load():
    if not os.path.exists(DATA_PATH):
        st.error(f"Run `python main.py` first → {DATA_PATH}")
        st.stop()
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["themes_detected"]  = df["themes_detected"].fillna("None").astype(str)
    df["primary_theme"]    = df["primary_theme"].fillna("General Social").astype(str)
    df["cleaned_text"]     = df["cleaned_text"].fillna("").astype(str)
    df["processed_text"]   = df["processed_text"].fillna("").astype(str)
    df["original_tweet"]   = df["original_tweet"].fillna("").astype(str)
    df["tb_polarity"]      = pd.to_numeric(df["tb_polarity"],  errors="coerce").fillna(0)
    df["tb_subjectivity"]  = pd.to_numeric(df["tb_subjectivity"], errors="coerce").fillna(0)
    df["vader_compound"]   = pd.to_numeric(df["vader_compound"],  errors="coerce").fillna(0)
    
    # Platform simulation
    platforms = ["Twitter/X", "Reddit", "Instagram", "YouTube"]
    np.random.seed(42)
    df["Platform"] = np.random.choice(platforms, size=len(df), p=[0.55, 0.20, 0.15, 0.10])
    return df

@st.cache_data
def load_mc():
    if os.path.exists(MODEL_COMP):
        mc = pd.read_csv(MODEL_COMP)
        for c in mc.columns:
            if c != "Metric":
                mc[c] = pd.to_numeric(mc[c], errors="coerce")
        return mc
    return None

df       = load()
mc_df    = load_mc()
sia      = SentimentIntensityAnalyzer()

# ── TOP NAVBAR WITH DARK/LIGHT MODE TOGGLE BUTTON ────────────────────────────
nav_l, nav_r = st.columns([7, 3])

with nav_l:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:14px;padding:8px 0;">
        <span style="font-size:32px;">🧠</span>
        <div>
            <div style="font-size:22px;font-weight:800;color:{TEXT_MAIN};margin:0;line-height:1.2;">MindPulse Analytics</div>
            <div style="font-size:12.5px;color:#6366F1;font-weight:600;margin-top:2px;">Social Media Sentiment & Mental Health Intelligence Engine</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with nav_r:
    t_col1, t_col2 = st.columns([6, 4])
    with t_col1:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;height:100%;padding-top:10px;">
            <span class="nav-status">● Live</span>
            <span class="nav-version">v2.4.0</span>
        </div>
        """, unsafe_allow_html=True)
    with t_col2:
        btn_label = "🌙 Dark" if not is_dark else "☀️ Light"
        if st.button(btn_label, key="hdr_theme_toggle", help="Switch between Light and Dark themes"):
            toggle_theme()

st.markdown(f"<hr style='margin:0 0 20px 0;border:none;border-bottom:1px solid {CARD_BORDER};'/>", unsafe_allow_html=True)

# ── TOP NAVIGATION BUTTONS BAR ───────────────────────────────────────────────
top_nav_cols = st.columns(len(NAV_PAGES), gap="small")
for idx, p_name in enumerate(NAV_PAGES):
    is_active = (st.session_state["page"] == p_name)
    btn_label = f"✨ {p_name}" if is_active else p_name
    btn_type = "primary" if is_active else "secondary"
    if top_nav_cols[idx].button(
        btn_label,
        key=f"topnav_{idx}",
        type=btn_type,
        use_container_width=True,
        help=f"Navigate to {p_name}"
    ):
        set_page(p_name)

# ── SIDEBAR FILTERS ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2>🧠 MindPulse</h2>
        <p>Sentiment Analytics Platform</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 🧭 Navigation")
    sidebar_page = st.radio("nav_radio", NAV_PAGES, index=NAV_PAGES.index(st.session_state["page"]), label_visibility="collapsed")
    if sidebar_page != st.session_state["page"]:
        st.session_state["page"] = sidebar_page
        st.rerun()

    st.markdown("---")
    st.markdown("#### 🎛️ Interactive Filters")
    
    sel_platform = st.selectbox("🌐 Platform Source", ["All", "Twitter/X", "Reddit", "Instagram", "YouTube"])
    all_ents = ["All"] + sorted(df["Entity"].dropna().unique().tolist())
    sel_ent  = st.selectbox("🏷️ Brand / Entity", all_ents)
    sel_snt  = st.multiselect("🎭 Sentiment Class", ["Positive","Negative","Neutral","Irrelevant"],
                               default=["Positive","Negative","Neutral","Irrelevant"])
    kw_search = st.text_input("🔍 Search Keyword", placeholder="e.g. anxiety, game, sleep")
    mh_only = st.toggle("🧠 Mental Health Signals Only", value=False)
    
    if st.button("🔄 Reset Filters", type="secondary"):
        st.rerun()

    st.markdown("---")

# ── FILTERING DATAFRAME ──────────────────────────────────────────────────────
mask = df["Sentiment"].isin(sel_snt) if sel_snt else pd.Series([True]*len(df))
if sel_platform != "All":
    mask &= (df["Platform"] == sel_platform)
if sel_ent != "All":
    mask &= (df["Entity"] == sel_ent)
if mh_only:
    mask &= (df["is_mental_health_related"] == True)
if kw_search.strip():
    pattern = re.escape(kw_search.strip())
    mask &= (df["original_tweet"].str.contains(pattern, case=False, na=False) |
             df["cleaned_text"].str.contains(pattern, case=False, na=False))

dff = df[mask].copy()

with st.sidebar:
    st.markdown(f"<div style='text-align:center;color:{TEXT_MUTED};font-size:12px'>Matching <b style='color:#6366F1'>{len(dff):,}</b> of {len(df):,} posts</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 0 — HOME PAGE
# ════════════════════════════════════════════════════════════
if st.session_state["page"] == "🏠 Home":
    st.markdown("""
    <div class="hero-banner">
        <h1>🧠 Welcome to MindPulse Analytics</h1>
        <p>An end-to-end NLP & Machine Learning platform for <b>Social Media Sentiment Analysis & Mental Health Awareness</b>. Understand public sentiment, track well-being themes, discover latent topics, and evaluate AI safety in real-time.</p>
        <div class="hero-badges">
            <span class="hero-badge-item">⚡ Real-Time NLP</span>
            <span class="hero-badge-item">🧠 Mental Health Lexicons</span>
            <span class="hero-badge-item">🔍 LDA Topic Modeling</span>
            <span class="hero-badge-item">🛡️ Misinformation Flagging</span>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("### Quick Navigation & Actions")
    cta1, cta2, cta3, cta4 = st.columns(4)
    if cta1.button("Explore Dashboard", use_container_width=True):
        set_page("📊 Dashboard")
    if cta2.button("Try Live Playground", use_container_width=True):
        set_page("🧪 Live Playground")
    if cta3.button("View Reports & Export", use_container_width=True):
        set_page("📄 Reports & Export")
    if cta4.button("Run Crisis Demo Test", use_container_width=True):
        st.session_state["playground_input"] = "I can't sleep and I feel so anxious and overwhelmed by everything. Feeling completely alone."
        set_page("🧪 Live Playground")

    st.markdown("")
    total_t = len(dff)
    mh_cnt  = int(dff["is_mental_health_related"].sum())
    pos_pct = (dff["Sentiment"]=="Positive").sum()/max(total_t,1)*100
    neg_pct = (dff["Sentiment"]=="Negative").sum()/max(total_t,1)*100
    flag_n  = int(dff["is_questionable_claim"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(stat_card(f"{total_t:,}", "Analyzed Posts", "💬", "card-purple"), unsafe_allow_html=True)
    c2.markdown(stat_card(f"{mh_cnt:,}", "MH Signals Identified", "🧠", "card-pink"), unsafe_allow_html=True)
    c3.markdown(stat_card(f"{pos_pct:.1f}% / {neg_pct:.1f}%", "Pos / Neg Ratio", "⚖️", "card-green"), unsafe_allow_html=True)
    c4.markdown(stat_card(f"{flag_n:,}", "Safety Flags Detected", "🛡️", "card-orange"), unsafe_allow_html=True)

    st.markdown('<div class="sec-head">✨ Key Platform Modules</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)

    with f1:
        st.markdown(f"""
        <div class="feature-card">
            <div>
                <div class="feature-icon">📊</div>
                <div class="feature-title">Sentiment Analytics</div>
                <div class="feature-desc">Explore distribution of Positive, Negative, Neutral, and Irrelevant posts across 30+ brand entities.</div>
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("Open Module ➔", key="mod_sent", use_container_width=True):
            set_page("🎭 Sentiment")

    with f2:
        st.markdown(f"""
        <div class="feature-card">
            <div>
                <div class="feature-icon">🧠</div>
                <div class="feature-title">MH Theme Scanner</div>
                <div class="feature-desc">Detect linguistic markers for Depression, Anxiety, Stress, Loneliness, and Support using lexical patterns.</div>
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("Open Module ➔", key="mod_mh", use_container_width=True):
            set_page("🧠 Mental Health Themes")

    with f3:
        st.markdown(f"""
        <div class="feature-card">
            <div>
                <div class="feature-icon">🔍</div>
                <div class="feature-title">Topic Modeling</div>
                <div class="feature-desc">Uncover hidden discourse themes automatically using unsupervised Latent Dirichlet Allocation (LDA).</div>
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("Open Module ➔", key="mod_topic", use_container_width=True):
            set_page("🔍 Topic Modeling")

    with f4:
        st.markdown(f"""
        <div class="feature-card">
            <div>
                <div class="feature-icon">🧪</div>
                <div class="feature-title">Live Playground</div>
                <div class="feature-desc">Test custom text in real-time with dual NLP sentiment engines (VADER & TextBlob) and word scoring.</div>
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("Open Module ➔", key="mod_play", use_container_width=True):
            set_page("🧪 Live Playground")

    st.markdown('<div class="sec-head">🔄 4-Stage End-to-End Pipeline Workflow</div>', unsafe_allow_html=True)
    w1, w2, w3, w4 = st.columns(4)
    with w1:
        st.markdown(f"""
        <div class="wf-card">
            <div class="wf-num">1</div>
            <div class="wf-title">Data Preprocessing</div>
            <div class="wf-desc">Regex cleaning, noise/URL removal, tokenization, lemmatization, and stopword filtering.</div>
        </div>""", unsafe_allow_html=True)
    with w2:
        st.markdown(f"""
        <div class="wf-card">
            <div class="wf-num">2</div>
            <div class="wf-title">Sentiment Engines</div>
            <div class="wf-desc">Benchmarking lexicon-based VADER Compound scoring against rule-based TextBlob Polarity.</div>
        </div>""", unsafe_allow_html=True)
    with w3:
        st.markdown(f"""
        <div class="wf-card">
            <div class="wf-num">3</div>
            <div class="wf-title">Mental Health Analysis</div>
            <div class="wf-desc">Mapping posts into 10 wellness themes and measuring emotional intensity profiles.</div>
        </div>""", unsafe_allow_html=True)
    with w4:
        st.markdown(f"""
        <div class="wf-card">
            <div class="wf-num">4</div>
            <div class="wf-title">Safety & Insights</div>
            <div class="wf-desc">Identifying unverified health claims and suspicious patterns for human review.</div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD OVERVIEW
# ════════════════════════════════════════════════════════════
elif st.session_state["page"] == "📊 Dashboard":
    st.markdown('<div class="page-title">📊 Executive Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Real-time overview of post volumes, sentiment distributions, and mental health indicators.</div>', unsafe_allow_html=True)

    total = len(dff)
    pos_n = (dff["Sentiment"]=="Positive").sum()
    neg_n = (dff["Sentiment"]=="Negative").sum()
    neu_n = (dff["Sentiment"]=="Neutral").sum()
    mh_n  = int(dff["is_mental_health_related"].sum())
    avg_p = dff["tb_polarity"].mean()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    for col, args in zip(
        [c1,c2,c3,c4,c5,c6],
        [(f"{total:,}","Total Posts","💬","card-blue"),
         (f"{pos_n:,}","Positive","✅","card-green"),
         (f"{neg_n:,}","Negative","⚠️","card-red"),
         (f"{neu_n:,}","Neutral","➖","card-purple"),
         (f"{mh_n:,}","MH Signals","🧠","card-pink"),
         (f"{avg_p:+.3f}","Avg Polarity","📈","card-orange")]
    ):
        col.markdown(stat_card(*args), unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="sec-head">Sentiment Distribution & Platform Breakdown</div>', unsafe_allow_html=True)
    r1c1, r1c2 = st.columns([5,5])

    with r1c1:
        cnt = dff["Sentiment"].value_counts().reset_index()
        cnt.columns = ["Sentiment","Count"]
        fig = px.pie(cnt, names="Sentiment", values="Count", hole=0.55,
                     color="Sentiment", color_discrete_map=PALETTE,
                     title="Corpus Sentiment Share")
        fig.update_traces(
            textposition="inside",
            textinfo="percent",
            textfont=dict(size=13, color="#ffffff"),
            pull=[0.04 if s=="Negative" else 0 for s in cnt["Sentiment"]],
            marker=dict(line=dict(color=CARD_BG, width=2))
        )
        fig.update_layout(
            legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center", x=0.5, title=None),
            margin=dict(t=50, b=70, l=20, r=20)
        )
        styled_fig(fig, 390)
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        plat_df = dff.groupby(["Platform", "Sentiment"]).size().reset_index(name="Count")
        fig_plat = px.bar(plat_df, x="Platform", y="Count", color="Sentiment",
                          barmode="group", color_discrete_map=PALETTE,
                          title="Sentiment Breakdown by Social Platform Source")
        fig_plat.update_layout(
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5, title=None),
            margin=dict(t=50, b=70, l=20, r=20)
        )
        styled_fig(fig_plat, 390)
        st.plotly_chart(fig_plat, use_container_width=True)

    st.markdown('<div class="sec-head">Entity Volume & Sentiment Mix</div>', unsafe_allow_html=True)
    top_e = dff["Entity"].value_counts().head(10).index
    cross = pd.crosstab(dff[dff["Entity"].isin(top_e)]["Entity"],
                        dff[dff["Entity"].isin(top_e)]["Sentiment"], normalize="index")*100
    cols_ord = [c for c in ["Positive","Negative","Neutral","Irrelevant"] if c in cross.columns]
    fig3 = px.bar(cross[cols_ord].reset_index(), x="Entity", y=cols_ord,
                  barmode="stack", color_discrete_map=PALETTE,
                  labels={"value":"Percentage (%)","variable":"Sentiment"},
                  title="Top 10 Entities Sentiment Distribution (%)")
    fig3.update_layout(
        xaxis_tickangle=-25,
        title=dict(x=0.01, y=0.98),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1.0, title=None),
        margin=dict(t=65, b=60, l=20, r=20)
    )
    fig3.update_xaxes(title_text="")
    styled_fig(fig3, 380)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="sec-head">Recent Analyzed Posts Stream</div>', unsafe_allow_html=True)
    samp_n = st.slider("Posts preview count", 5, 30, 8, key="db_samp")
    samp   = dff[["Platform","Entity","Sentiment","vader_sentiment","tb_sentiment","original_tweet"]].sample(
        n=min(samp_n, len(dff)), random_state=42)
    for _, row in samp.iterrows():
        cls = tweet_cls(row["Sentiment"])
        st.markdown(f"""
        <div class="tweet-card {cls}">
            {badge(row['Sentiment'])} &nbsp;
            <span style="background:{"#1E1B4B" if is_dark else "#EEF2FF"};color:{"#C7D2FE" if is_dark else "#4338CA"};font-weight:700;font-size:11px;padding:2px 8px;border-radius:12px">{row['Platform']}</span>
            &nbsp; <strong style="color:{TEXT_MAIN}">{row['Entity']}</strong>
            <div style="margin-top:6px">{str(row['original_tweet'])[:280]}</div>
            <div style="margin-top:6px;font-size:11px;color:{TEXT_MUTED}">
                VADER: <b>{row['vader_sentiment']}</b> &nbsp;·&nbsp; TextBlob: <b>{row['tb_sentiment']}</b>
            </div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 2 — SENTIMENT EXPLORER
# ════════════════════════════════════════════════════════════
elif st.session_state["page"] == "🎭 Sentiment":
    st.markdown('<div class="page-title">🎭 Sentiment Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Detailed distribution of TextBlob Polarity, VADER Compound, and Subjectivity metrics.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Polarity & Subjectivity","🏷️ Entity Breakdown","☁️ Word Clouds","📋 Raw Data Stream"])

    with tab1:
        a,b = st.columns(2)
        with a:
            fig = px.histogram(dff, x="tb_polarity", color="Sentiment",
                               nbins=60, color_discrete_map=PALETTE,
                               barmode="overlay", opacity=0.7,
                               marginal="violin",
                               labels={"tb_polarity":"TextBlob Polarity (-1 → +1)"},
                               title="TextBlob Polarity Distribution")
            fig.update_layout(bargap=0.05)
            styled_fig(fig, 400)
            st.plotly_chart(fig, use_container_width=True)
        with b:
            samp3k = dff.sample(min(3000,len(dff)))
            fig2 = px.scatter(samp3k, x="tb_polarity", y="tb_subjectivity",
                              color="Sentiment", color_discrete_map=PALETTE,
                              opacity=0.45,
                              labels={"tb_polarity":"Polarity","tb_subjectivity":"Subjectivity"},
                              title="Polarity vs Subjectivity Scatter")
            fig2.update_traces(marker=dict(size=5))
            styled_fig(fig2, 400)
            st.plotly_chart(fig2, use_container_width=True)

        c,d = st.columns(2)
        with c:
            fig3 = px.box(dff, x="Sentiment", y="vader_compound",
                          color="Sentiment", color_discrete_map=PALETTE,
                          notched=True, points="outliers",
                          labels={"vader_compound":"VADER Compound Score"},
                          title="VADER Compound Score Distribution")
            styled_fig(fig3, 340)
            st.plotly_chart(fig3, use_container_width=True)
        with d:
            fig4 = px.violin(dff, x="Sentiment", y="tb_subjectivity",
                             color="Sentiment", color_discrete_map=PALETTE, box=True,
                             labels={"tb_subjectivity":"TextBlob Subjectivity"},
                             title="Subjectivity Violin Plot")
            styled_fig(fig4, 340)
            st.plotly_chart(fig4, use_container_width=True)

    with tab2:
        n_top = st.slider("Top N entities", 5, 20, 10, key="ent_sl")
        top_ents = dff["Entity"].value_counts().head(n_top).index
        df_top   = dff[dff["Entity"].isin(top_ents)]
        crs = pd.crosstab(df_top["Entity"], df_top["Sentiment"], normalize="index")*100
        ord_c = [c for c in ["Positive","Negative","Neutral","Irrelevant"] if c in crs.columns]
        fig5 = px.bar(crs[ord_c].reset_index(), x="Entity", y=ord_c,
                      barmode="stack", color_discrete_map=PALETTE,
                      labels={"value":"% Share","variable":"Sentiment"},
                      title=f"Top {n_top} Entities Sentiment Profile (%)")
        fig5.update_layout(
            xaxis_tickangle=-30,
            title=dict(x=0.01, y=0.98),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1.0, title=None),
            margin=dict(t=65, b=65, l=20, r=20)
        )
        fig5.update_xaxes(title_text="")
        styled_fig(fig5, 440)
        st.plotly_chart(fig5, use_container_width=True)

    with tab3:
        wc_choice = st.selectbox("Sentiment class for word cloud", ["All","Positive","Negative","Neutral"])
        cmaps = {"All":"RdYlBu","Positive":"Greens","Negative":"Reds","Neutral":"Blues"}
        txt = " ".join(
            dff["processed_text"] if wc_choice=="All"
            else dff[dff["Sentiment"]==wc_choice]["processed_text"]
        )
        b64 = wc_b64(txt, cmaps[wc_choice])
        if b64:
            st.markdown(f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:14px;border:1px solid {CARD_BORDER}"/>', unsafe_allow_html=True)
        else:
            st.info("No text data for selected filter.")

    with tab4:
        cols_e = ["Platform","Entity","Sentiment","vader_sentiment","tb_sentiment",
                  "tb_polarity","tb_subjectivity","original_tweet"]
        st.dataframe(dff[cols_e].reset_index(drop=True), use_container_width=True, height=480)
        st.download_button("⬇️ Download Current Filtered Data (CSV)", dff[cols_e].to_csv(index=False).encode(),
                           "filtered_sentiment.csv", "text/csv")


# ════════════════════════════════════════════════════════════
# PAGE 3 — MENTAL HEALTH THEMES
# ════════════════════════════════════════════════════════════
elif st.session_state["page"] == "🧠 Mental Health Themes":
    st.markdown('<div class="page-title">🧠 Mental Health Theme Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Lexical thematic analysis mapping posts into 10 emotional well-being categories.</div>', unsafe_allow_html=True)

    mh = dff[dff["is_mental_health_related"]==True].copy()
    st.markdown(f'<div class="info-box">🧠 <b>{len(mh):,}</b> posts ({len(mh)/max(len(dff),1)*100:.2f}% of filtered dataset) contain mental health linguistic signals.</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-head">Theme Frequency & Sentiment Heatmap</div>', unsafe_allow_html=True)
    l, r = st.columns([5,5])

    with l:
        tc = mh["primary_theme"].value_counts().reset_index()
        tc.columns = ["Theme","Count"]
        fig_t = px.bar(tc, x="Count", y="Theme", orientation="h",
                       color="Count", color_continuous_scale=["#EDE9FE","#7C3AED"],
                       title="Mental Health Theme Mentions Frequency")
        fig_t.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        styled_fig(fig_t, 380)
        st.plotly_chart(fig_t, use_container_width=True)

    with r:
        hm = pd.crosstab(mh["primary_theme"], mh["Sentiment"], normalize="index")*100
        vld = [c for c in ["Negative","Positive","Neutral","Irrelevant"] if c in hm.columns]
        fig_hm = px.imshow(hm[vld], text_auto=".1f", color_continuous_scale="RdPu",
                           labels=dict(color="%"), title="Sentiment Profile per Theme (%)")
        styled_fig(fig_hm, 380)
        st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown('<div class="sec-head">Interactive Theme Explorer</div>', unsafe_allow_html=True)
    themes_av = sorted(mh["primary_theme"].unique())
    sel_th    = st.selectbox("Select Theme Category:", themes_av)
    th_sub    = mh[mh["primary_theme"]==sel_th]
    st.caption(f"{len(th_sub):,} posts tagged with **{sel_th}**")

    n_pr = st.slider("Preview Count", 3, 20, 6, key="mh_prev")
    for _, row in th_sub.head(n_pr).iterrows():
        cls = tweet_cls(row["Sentiment"])
        st.markdown(f"""
        <div class="tweet-card {cls}">
            {badge(row['Sentiment'])} &nbsp;<strong style="color:{TEXT_MAIN}">{row['Entity']}</strong>
            <div style="margin-top:6px">{str(row['original_tweet'])[:260]}</div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 4 — TOPIC MODELING
# ════════════════════════════════════════════════════════════
elif st.session_state["page"] == "🔍 Topic Modeling":
    st.markdown('<div class="page-title">🔍 LDA Topic Modeling Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Unsupervised Latent Dirichlet Allocation (LDA) discovering thematic clusters without label collision.</div>', unsafe_allow_html=True)

    top_col1, top_col2 = st.columns([5,5])

    with top_col1:
        tc = dff["dominant_topic_label"].value_counts().reset_index()
        tc.columns = ["Topic","Count"]
        
        fig_p = px.pie(
            tc,
            names="Topic",
            values="Count",
            hole=0.50,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            title="Topic Distribution Share"
        )
        fig_p.update_traces(
            textposition="inside",
            textinfo="percent",
            textfont=dict(size=13, color="#1E293B"),
            marker=dict(line=dict(color=CARD_BG, width=2)),
            hoverinfo="label+percent+value"
        )
        fig_p.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
                title=None
            ),
            margin=dict(t=50, b=90, l=20, r=20)
        )
        styled_fig(fig_p, 420)
        st.plotly_chart(fig_p, use_container_width=True)

    with top_col2:
        st.markdown("#### 📌 Discovered Topic Clusters & Breakdown")
        total_posts = len(dff)
        for _, r in tc.iterrows():
            t_name = r["Topic"]
            t_cnt  = r["Count"]
            t_pct  = (t_cnt / max(total_posts, 1)) * 100
            st.markdown(f"""
            <div class="topic-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-weight:700;color:{"#818CF8" if is_dark else "#4338CA"};font-size:15px;">{t_name}</span>
                    <span style="background:{"#1E1B4B" if is_dark else "#EEF2FF"};color:{"#C7D2FE" if is_dark else "#4338CA"};font-weight:700;font-size:12px;padding:3px 10px;border-radius:14px;">{t_pct:.1f}% ({t_cnt:,} posts)</span>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-head">Topic Keyword Distributions</div>', unsafe_allow_html=True)
    img_p = os.path.join(FIGURES_DIR, "topic_modeling_word_distributions.png")
    if os.path.exists(img_p):
        st.image(img_p, caption="Top Keywords Per Discovered LDA Topic", use_container_width=True)

    st.markdown('<div class="sec-head">Topic × Sentiment Cross-Matrix</div>', unsafe_allow_html=True)
    crs_t = pd.crosstab(dff["dominant_topic_label"], dff["Sentiment"], normalize="index")*100
    vc2   = [c for c in ["Positive","Negative","Neutral","Irrelevant"] if c in crs_t.columns]
    fig_ts = px.imshow(crs_t[vc2], text_auto=".1f", color_continuous_scale="Blues",
                       title="Topic × Sentiment Heatmap (%)",
                       labels=dict(color="% Share"))
    fig_ts.update_layout(margin=dict(t=50, b=40, l=40, r=40))
    styled_fig(fig_ts, 360)
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown('<div class="sec-head">Browse Posts by Topic Category</div>', unsafe_allow_html=True)
    sel_tp  = st.selectbox("Select Topic Category:", sorted(dff["dominant_topic_label"].unique()))
    tp_sub  = dff[dff["dominant_topic_label"]==sel_tp]
    st.caption(f"Showing **{len(tp_sub):,}** posts in topic: **{sel_tp}**")
    n_tp    = st.slider("Preview count", 3, 15, 6, key="tp_sl")
    for _, row in tp_sub.sample(min(n_tp, len(tp_sub)), random_state=42).iterrows():
        cls = tweet_cls(row["Sentiment"])
        st.markdown(f"""
        <div class="tweet-card {cls}">
            {badge(row['Sentiment'])} &nbsp;
            <span style="background:{"#1E1B4B" if is_dark else "#EEF2FF"};color:{"#C7D2FE" if is_dark else "#4338CA"};font-weight:700;font-size:11px;padding:2px 8px;border-radius:12px">{row['Platform']}</span>
            &nbsp; <strong style="color:{TEXT_MAIN}">{row['Entity']}</strong>
            <div style="margin-top:6px">{str(row['original_tweet'])[:260]}</div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 5 — SAFETY FLAGS
# ════════════════════════════════════════════════════════════
elif st.session_state["page"] == "🛡️ Safety Flags":
    st.markdown('<div class="page-title">🛡️ AI Safety & Misinformation Flags</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Heuristic scanning engine detecting extreme health claims and panic triggers for human review.</div>', unsafe_allow_html=True)

    st.markdown('<div class="warn-box">⚠️ <b>Content Safety Safeguard:</b> Flagged posts require human review. NLP models highlight questionable patterns but do not make medical diagnoses.</div>', unsafe_allow_html=True)

    flagged = dff[dff["is_questionable_claim"]==True]
    tf, tr  = len(flagged), len(dff)
    c1,c2,c3 = st.columns(3)
    c1.metric("Flagged Posts",   f"{tf:,}")
    c2.metric("Total Evaluated", f"{tr:,}")
    c3.metric("Flag Rate",       f"{tf/max(tr,1)*100:.3f}%")

    if tf > 0:
        fcat = flagged["flag_category"].value_counts()
        fig_f = px.bar(fcat.reset_index(), x="flag_category", y="count",
                       color="count", color_continuous_scale=["#FEF3C7","#D97706"],
                       labels={"flag_category":"Flag Category","count":"Posts"},
                       title="Flagged Claims by Category")
        styled_fig(fig_f, 320)
        st.plotly_chart(fig_f, use_container_width=True)

        st.markdown("#### Sample Flagged Posts Stream (Human Review Queue)")
        for _, row in flagged.head(10).iterrows():
            st.markdown(f"""
            <div class="tweet-card" style="border-left-color:#F59E0B">
                <span style="color:#D97706;font-weight:700">⚠️ {row['flag_category']}</span>
                &nbsp;· Trigger: <code style="font-size:11px;background:{"#78350F" if is_dark else "#FEF3C7"};color:{"#FDE68A" if is_dark else "#92400E"};padding:2px 6px;border-radius:4px">{row['flag_matched_pattern']}</code>
                <div style="margin-top:8px">{str(row['original_tweet'])[:240]}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.success("✅ No posts flagged under current filter criteria.")


# ════════════════════════════════════════════════════════════
# PAGE 6 — LIVE PLAYGROUND
# ════════════════════════════════════════════════════════════
elif st.session_state["page"] == "🧪 Live Playground":
    st.markdown('<div class="page-title">🧪 Live Real-Time Sentiment Playground</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Test custom text in real-time with dual sentiment engines (VADER & TextBlob), emotion profiling, and crisis risk assessment.</div>', unsafe_allow_html=True)

    from src.mental_health_analysis import tag_themes

    st.markdown("#### ⚡ Quick Sample Buttons")
    s1, s2, s3 = st.columns(3)
    if s1.button("🟢 Load Positive Example"):
        st.session_state["playground_input"] = "I had a wonderful session with my counselor today! Feeling hopeful, recharged, and peaceful. 😊✨"
        st.rerun()
    if s2.button("🔴 Load Crisis / Anxiety Example"):
        st.session_state["playground_input"] = "I can't sleep, my mind won't stop racing and I'm having panic attacks. Feeling completely alone and hopeless."
        st.rerun()
    if s3.button("🔵 Load Neutral Example"):
        st.session_state["playground_input"] = "The new software patch was released today with updated user interface elements."
        st.rerun()

    user_text = st.text_area(
        "✏️ Enter text to analyze:",
        value=st.session_state["playground_input"],
        height=140,
        placeholder="Type any tweet or statement here…"
    )
    st.caption(f"Character count: **{len(user_text)}** / 500")

    if st.button("🚀 Analyze Text Now", type="primary"):
        if not user_text.strip():
            st.error("⚠️ Please enter some text before analyzing.")
        else:
            with st.spinner("Analyzing text through VADER, TextBlob, and Mental Health Lexicons…"):
                time.sleep(0.3)
                v  = sia.polarity_scores(user_text)
                vl = "Positive" if v["compound"]>=0.05 else ("Negative" if v["compound"]<=-0.05 else "Neutral")
                tb = TextBlob(user_text).sentiment
                tl = "Positive" if tb.polarity>0.05 else ("Negative" if tb.polarity<-0.05 else "Neutral")
                themes = tag_themes(user_text)

                is_crisis = any(t in ["Depression","Anxiety","Stress & Burnout","Loneliness & Isolation"] for t in themes) and v["compound"] < -0.3

            st.markdown("---")
            st.markdown("### 📊 Real-Time Analysis Results")

            if is_crisis:
                st.markdown("""
                <div class="danger-box">
                    🚨 <b>HIGH CRISIS RISK DETECTED:</b> Text exhibits strong negative sentiment compound with anxiety/depression markers.<br>
                    <b>Support Helpline:</b> Call or text <b>988</b> (US/Canada) or visit <a href="https://findahelpline.com" target="_blank">findahelpline.com</a>.
                </div>""", unsafe_allow_html=True)

            r1, r2, r3 = st.columns(3)

            with r1:
                st.markdown("#### 🔬 VADER Engine")
                st.markdown(f"Verdict: {badge(vl)}", unsafe_allow_html=True)
                fig_v = go.Figure(go.Bar(
                    x=["Positive","Negative","Neutral"],
                    y=[v["pos"],v["neg"],v["neu"]],
                    marker_color=["#10B981","#EF4444","#3B82F6"],
                    text=[f"{x:.3f}" for x in [v["pos"],v["neg"],v["neu"]]],
                    textposition="outside"
                ))
                fig_v.add_annotation(x=1, y=max(v["pos"],v["neg"],v["neu"])+0.1,
                                     text=f"Compound: <b>{v['compound']:+.3f}</b>",
                                     showarrow=False, font=dict(size=13,color="#818CF8" if is_dark else "#4338CA"))
                styled_fig(fig_v, 250)
                st.plotly_chart(fig_v, use_container_width=True)

            with r2:
                st.markdown("#### 📝 TextBlob Engine")
                st.markdown(f"Verdict: {badge(tl)}", unsafe_allow_html=True)
                pol_c = "#10B981" if tb.polarity>0 else ("#EF4444" if tb.polarity<0 else "#9CA3AF")
                fig_tb = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=tb.polarity,
                    title={"text":"Polarity", "font":{"color":TEXT_MAIN, "size":15}},
                    number={"font":{"size":26,"color":pol_c}},
                    gauge={
                        "axis":{"range":[-1,1], "tickcolor":TEXT_MUTED, "tickfont":{"color":TEXT_MUTED}},
                        "bar":{"color":pol_c,"thickness":0.25},
                        "steps":[{"range":[-1,0],"color":"#3B1219" if is_dark else "#FEE2E2"},{"range":[0,1],"color":"#064E3B" if is_dark else "#D1FAE5"}]
                    }
                ))
                fig_tb.update_layout(height=240, paper_bgcolor=PLOT_BG, font=dict(color=TEXT_MAIN))
                st.plotly_chart(fig_tb, use_container_width=True)
                st.metric("Subjectivity Score", f"{tb.subjectivity:.3f}", delta="Opinionated" if tb.subjectivity>0.5 else "Factual")

            with r3:
                st.markdown("#### 🧠 Identified MH Themes")
                if themes:
                    for t in themes:
                        st.markdown(f"🏷️ `{t}`")
                else:
                    st.info("No mental health keywords detected.")

                if vl == tl:
                    st.success(f"✅ Both engines **agree**: {vl}")
                else:
                    st.warning(f"⚠️ Engines **differ**: VADER={vl}, TextBlob={tl}")

            st.markdown("---")
            st.markdown("#### 🔤 Per-Word Sentiment Contribution")
            words = [w for w in re.sub(r"[^\w\s]","",user_text).lower().split() if len(w)>2]
            wd = [{"Word":w,"Score":round(sia.polarity_scores(w)["compound"],3)} for w in words]
            if wd:
                wdf = pd.DataFrame(wd).sort_values("Score")
                fig_w = px.bar(wdf, x="Score", y="Word", orientation="h",
                               color="Score", color_continuous_scale=["#EF4444","#F9FAFB","#10B981"],
                               color_continuous_midpoint=0, labels={"Score":"VADER Score","Word":""})
                fig_w.add_vline(x=0, line_color="#9CA3AF", line_dash="dot")
                fig_w.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
                styled_fig(fig_w, max(180, len(words)*26))
                st.plotly_chart(fig_w, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 7 — REPORTS & EXPORT
# ════════════════════════════════════════════════════════════
elif st.session_state["page"] == "📄 Reports & Export":
    st.markdown('<div class="page-title">📄 Reports & Data Export</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Download analytical reports, benchmark tables, and CSV extracts.</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-head">📥 Export Datasets & Summaries</div>', unsafe_allow_html=True)
    e1, e2 = st.columns(2)

    with e1:
        st.markdown("""
        <div class="about-card">
            <h3>📊 CSV Dataset Export</h3>
            <p>Download the current filtered dataset containing text, sentiment predictions, mental health tags, and safety flags.</p>
        </div>""", unsafe_allow_html=True)
        cols_exp = ["Platform","Entity","Sentiment","vader_sentiment","tb_sentiment","is_mental_health_related","primary_theme","original_tweet"]
        st.download_button("⬇️ Download CSV Extract", dff[cols_exp].to_csv(index=False).encode(), "mindpulse_export.csv", "text/csv")

    with e2:
        st.markdown("""
        <div class="about-card">
            <h3>📝 Executive Summary Report</h3>
            <p>Download automated insights report detailing corpus statistics and mental health frequency distribution.</p>
        </div>""", unsafe_allow_html=True)
        rep_txt = f"MINDPULSE ANALYTICS EXECUTIVE REPORT\nTotal Posts Evaluated: {len(dff):,}\nMental Health Mentions: {int(dff['is_mental_health_related'].sum()):,}\nPositive Share: {(dff['Sentiment']=='Positive').mean()*100:.2f}%\nNegative Share: {(dff['Sentiment']=='Negative').mean()*100:.2f}%\n"
        st.download_button("⬇️ Download Summary Report (.TXT)", rep_txt.encode(), "mindpulse_summary.txt", "text/plain")

    st.markdown('<div class="sec-head">📚 Technical Terminology Glossary</div>', unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("""
        - **VADER:** Valuation Dictionary and sEntiment Reasoner calibrated specifically for social media emoticons and slang.
        - **TextBlob:** Rule-based NLP library calculating polarity (-1 to +1) and subjectivity (0 to 1).
        """)
    with g2:
        st.markdown("""
        - **LDA:** Latent Dirichlet Allocation, an unsupervised probabilistic model for discovering topic clusters.
        - **Safety Flags:** RegEx heuristics highlighting unverified medical claims or acute distress markers.
        """)


# ════════════════════════════════════════════════════════════
# PAGE 8 — ABOUT PAGE
# ════════════════════════════════════════════════════════════
elif st.session_state["page"] == "ℹ️ About":
    st.markdown('<div class="page-title">ℹ️ About MindPulse Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Project background, architecture, methodologies, and academic disclaimers.</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([6,4])

    with col_left:
        st.markdown("""
        <div class="about-card">
            <h3>🎯 Project Objective</h3>
            <p>This project implements a full end-to-end NLP and sentiment analysis framework specifically tailored for <b>Mental Health Awareness</b> on social media. By mining public posts across brand contexts and emotional signals, MindPulse helps observe public sentiment trends, recognize emotional well-being themes, and flag health misinformation.</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <h3>⚙️ Technical Pipeline</h3>
            <ul>
                <li><b>Data Ingestion & Cleaning:</b> Standardizing textual formats, normalizing noise, removing URLs/handles, preserving emoticons.</li>
                <li><b>Text Preprocessing:</b> Tokenization, stopword removal with negation preservation, NLTK lemmatization.</li>
                <li><b>Sentiment Engines:</b> Benchmarking TextBlob against VADER SentimentIntensityAnalyzer.</li>
                <li><b>Thematic Analysis:</b> Lexical pattern matching across 10 mental-health categories.</li>
                <li><b>Topic Modeling:</b> Latent Dirichlet Allocation (LDA) discovering latent discourse.</li>
                <li><b>Misinformation Scanner:</b> Heuristic regex engine identifying extreme claims for human review.</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown(f"""
        <div class="about-card">
            <h3>🛠️ Technology Stack</h3>
            <ul>
                <li><b>Core Language:</b> Python 3.10+</li>
                <li><b>UI Framework:</b> Streamlit ({'Dark Mode' if is_dark else 'Light Mode'})</li>
                <li><b>Data Science:</b> Pandas, NumPy 2.0</li>
                <li><b>NLP & ML:</b> NLTK, TextBlob, Scikit-Learn</li>
                <li><b>Visualization:</b> Plotly Express, WordCloud</li>
            </ul>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="warn-box">
            <b>🛡️ Academic & Ethical Safeguards</b><br>
            <i>Important Notice:</i> Heuristic keyword matching does <b>NOT</b> constitute a clinical assessment or medical diagnosis.<br><br>
            If you or someone you know is in distress, please reach out to professional support:
            <ul style="margin-top:6px;margin-bottom:0;padding-left:18px;">
                <li><b>US/Canada Crisis Hotline:</b> Call or text <b>988</b></li>
                <li><b>UK Crisis Support:</b> Call <b>111</b> or text SHOUT to <b>85258</b></li>
                <li><b>International:</b> <a href="https://findahelpline.com" target="_blank">findahelpline.com</a></li>
            </ul>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# GLOBAL APP FOOTER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-footer">
    <div>
        <strong>🧠 MindPulse Analytics</strong> · Social Media Sentiment Analysis for Mental Health Awareness<br>
        <span style="font-size:11.5px;">Developed by <b>Roshini Krishnasri</b> · Academic Research & Analytics Project</span>
    </div>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
        <a href="https://www.linkedin.com/in/roshinikrishnasri" target="_blank" class="footer-contact-link">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" style="vertical-align:-2px;"><path d="M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14m-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.28 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93h2.75M6.46 10.9v8.37H9.2V10.9H6.46M7.83 6.45a1.65 1.65 0 0 0-1.66 1.65c0 .91.74 1.66 1.66 1.66a1.66 1.66 0 0 0 1.66-1.66c0-.91-.74-1.65-1.66-1.65Z"/></svg> LinkedIn
        </a>
        <a href="https://github.com/RoshiniKrishnasri" target="_blank" class="footer-contact-link">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" style="vertical-align:-2px;"><path d="M12 2A10 10 0 0 0 2 12c0 4.42 2.87 8.17 6.84 9.5.5.08.66-.23.66-.5v-1.69c-2.77.6-3.36-1.34-3.36-1.34-.46-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.6.07-.6 1 .07 1.53 1.03 1.53 1.03.87 1.52 2.34 1.07 2.91.83.1-.65.35-1.09.63-1.34-2.22-.25-4.55-1.11-4.55-4.92 0-1.11.38-2 1.03-2.71-.1-.25-.45-1.29.1-2.64 0 0 .84-.27 2.75 1.02.79-.22 1.65-.33 2.5-.33.85 0 1.71.11 2.5.33 1.91-1.29 2.75-1.02 2.75-1.02.55 1.35.2 2.39.1 2.64.65.71 1.03 1.6 1.03 2.71 0 3.82-2.34 4.66-4.57 4.91.36.31.69.92.69 1.85V21c0 .27.16.59.67.5C19.14 20.16 22 16.42 22 12A10 10 0 0 0 12 2Z"/></svg> GitHub
        </a>
        <a href="mailto:rgaddam3@gitam.in" class="footer-contact-link">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;"><rect width="20" height="16" x="2" y="4" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/></svg> Mail
        </a>
        <span class="footer-tag">Python 3.10+</span>
        <span class="footer-tag">Streamlit</span>
        <a href="https://findahelpline.com" target="_blank" style="margin-left:8px;font-size:12px;">🆘 Crisis Support (988)</a>
    </div>
</div>
""", unsafe_allow_html=True)
