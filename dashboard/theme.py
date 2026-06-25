"""Dark theme + Plotly helpers.

Design philosophy: use NATIVE Streamlit components everywhere, restyle via CSS.
We do not emit custom HTML cards - those are fragile under Streamlit's markdown
parser. Every visual primitive in the dashboard is `st.metric`, `st.container`,
`st.columns`, `st.dataframe`, etc., dressed up with `[data-testid]` selectors.
"""
from __future__ import annotations

import streamlit as st

# ---------- Brand palette ----------
ACCENT = "#a78bfa"          # violet 400 - primary accent
ACCENT_DEEP = "#8b5cf6"     # violet 500
ACCENT_2 = "#22d3ee"        # cyan 400 - secondary
ACCENT_3 = "#f472b6"        # pink 400 - tertiary

BG = "#0b0b12"
SURFACE = "#13131c"
SURFACE_2 = "#1a1a26"
BORDER = "rgba(255,255,255,0.08)"
BORDER_STRONG = "rgba(255,255,255,0.14)"

TEXT = "#fafafa"
MUTED = "#a1a1aa"
DIM = "#71717a"

RISK_HIGH = "#f87171"
RISK_MED = "#fbbf24"
RISK_LOW = "#34d399"


_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ============ Base ============ */
html, body, [class*="css"], .stApp,
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    letter-spacing: -0.011em;
}}

.stApp {{
    background:
        radial-gradient(1200px 600px at 0% -10%, rgba(139,92,246,0.12), transparent 60%),
        radial-gradient(900px 500px at 100% -10%, rgba(34,211,238,0.10), transparent 60%),
        radial-gradient(800px 600px at 50% 110%, rgba(244,114,182,0.06), transparent 60%),
        {BG} !important;
    background-attachment: fixed !important;
    color: {TEXT} !important;
}}

header[data-testid="stHeader"] {{ background: transparent !important; }}
footer, #MainMenu, .stDeployButton {{ display: none !important; }}
[data-testid="stToolbar"] {{ display: none !important; }}

.block-container {{
    padding-top: 1.25rem !important;
    padding-bottom: 5rem !important;
    max-width: 1440px !important;
}}

/* ============ Typography ============ */
h1, h2, h3 {{ color: {TEXT} !important; letter-spacing: -0.025em !important; }}
h1 {{ font-weight: 800 !important; font-size: 2.4rem !important; line-height: 1.1 !important; }}
h2 {{ font-weight: 700 !important; font-size: 1.5rem !important; margin-top: 1.5rem !important; }}
h3 {{ font-weight: 600 !important; font-size: 1.15rem !important; color: {MUTED} !important; text-transform: uppercase; letter-spacing: 0.08em !important; }}

/* ============ st.metric → glass stat tile ============ */
[data-testid="stMetric"] {{
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015)) !important;
    border: 1px solid {BORDER} !important;
    border-radius: 16px !important;
    padding: 1.1rem 1.25rem 1.2rem 1.25rem !important;
    backdrop-filter: blur(12px);
    transition: all 0.18s ease;
    position: relative;
    overflow: hidden;
    min-height: 110px;
}}
[data-testid="stMetric"]::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, {ACCENT_DEEP}, {ACCENT_2});
    opacity: 0;
    transition: opacity 0.18s ease;
}}
[data-testid="stMetric"]:hover {{
    border-color: {BORDER_STRONG} !important;
    transform: translateY(-1px);
}}
[data-testid="stMetric"]:hover::before {{ opacity: 1; }}

[data-testid="stMetricLabel"] {{
    color: {MUTED} !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
[data-testid="stMetricLabel"] p {{
    color: {MUTED} !important;
    font-size: 0.78rem !important;
}}
[data-testid="stMetricValue"] {{
    color: {TEXT} !important;
    font-size: 1.95rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em !important;
    font-feature-settings: 'tnum' !important;
    line-height: 1.15 !important;
    margin-top: 4px !important;
}}
[data-testid="stMetricDelta"] {{
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}}
[data-testid="stMetricDelta"] svg {{ display: none !important; }}

/* ============ st.container(border=True) → glass card ============ */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.01)) !important;
    border: 1px solid {BORDER} !important;
    border-radius: 16px !important;
    padding: 1.2rem 1.4rem !important;
    backdrop-filter: blur(12px);
}}

/* ============ Tabs → pill bar ============ */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px !important;
    background: rgba(255,255,255,0.025) !important;
    border-radius: 14px !important;
    padding: 6px !important;
    border: 1px solid {BORDER};
    margin-bottom: 1.25rem;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    border-radius: 10px !important;
    padding: 10px 22px !important;
    font-weight: 500 !important;
    color: {MUTED} !important;
    border: none !important;
    transition: all 0.15s ease;
}}
.stTabs [data-baseweb="tab"]:hover {{
    background: rgba(255,255,255,0.04) !important;
    color: {TEXT} !important;
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    background: linear-gradient(135deg, rgba(139,92,246,0.30), rgba(34,211,238,0.18)) !important;
    color: {TEXT} !important;
    box-shadow: inset 0 0 0 1px rgba(139,92,246,0.35);
}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none !important; }}

/* ============ Buttons ============ */
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {ACCENT_DEEP} 0%, {ACCENT_2} 100%) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 6px 18px rgba(139,92,246,0.35) !important;
    transition: all 0.18s ease !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.25rem !important;
}}
.stButton > button[kind="primary"]:hover {{
    transform: translateY(-1px);
    box-shadow: 0 10px 26px rgba(139,92,246,0.5) !important;
}}
.stButton > button {{
    border-radius: 10px !important;
}}

/* ============ Inputs ============ */
.stSelectbox [data-baseweb="select"] > div,
.stTextInput input,
.stNumberInput input {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    color: {TEXT} !important;
}}
.stSelectbox [data-baseweb="select"] > div:hover {{ border-color: {BORDER_STRONG} !important; }}

/* ============ Dataframes ============ */
[data-testid="stDataFrame"] {{
    border-radius: 14px !important;
    overflow: hidden;
    border: 1px solid {BORDER};
}}
[data-testid="stDataFrame"] > div {{
    background: rgba(255,255,255,0.015) !important;
}}

/* ============ Expanders ============ */
.stExpander {{
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
}}
.stExpander summary {{ color: {TEXT} !important; font-weight: 500 !important; }}
.stExpander summary:hover {{ color: {ACCENT} !important; }}

/* ============ Alerts ============ */
[data-testid="stAlert"] {{
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid {BORDER} !important;
    border-left: 3px solid {ACCENT_DEEP} !important;
    border-radius: 12px !important;
    color: {TEXT} !important;
}}

/* ============ Spinner & Progress ============ */
.stSpinner > div > div {{ border-top-color: {ACCENT} !important; }}

/* ============ Code blocks (used in KB tab) ============ */
code, pre {{
    font-family: 'JetBrains Mono', monospace !important;
    background: rgba(139,92,246,0.10) !important;
    color: {ACCENT} !important;
    padding: 2px 6px !important;
    border-radius: 6px !important;
}}

/* ============ Section divider ============ */
hr {{
    border: none !important;
    border-top: 1px solid {BORDER} !important;
    margin: 1.5rem 0 !important;
}}

/* ============ Scrollbar ============ */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.08); border-radius: 5px; }}
::-webkit-scrollbar-thumb:hover {{ background: rgba(255,255,255,0.15); }}
</style>
"""


def inject_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------- Plotly helpers ----------
def apply_plotly_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        margin=dict(l=10, r=10, t=30, b=10),
        # Force an empty title string; newer Plotly renders the literal text
        # "undefined" when title.font is set but title.text is left unset.
        # title=dict(size=14, color=TEXT, family="Inter")
        title=dict(text="", font=dict(size=14, color=TEXT, family="Inter")),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, font=dict(color=TEXT)),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.08)",
            tickfont=dict(color=MUTED),
            title_font=dict(color=MUTED, size=11),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.08)",
            tickfont=dict(color=MUTED),
            title_font=dict(color=MUTED, size=11),
        ),
    )
    return fig


VIOLET_CYAN_SCALE = [
    [0.0, "#22d3ee"],
    [0.5, "#a78bfa"],
    [1.0, "#f472b6"],
]

RISK_SCALE = [
    [0.0, "#34d399"],
    [0.5, "#fbbf24"],
    [1.0, "#f87171"],
]
