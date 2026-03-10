import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

# =========================================================
# Page
# =========================================================

st.set_page_config(
    page_title="Iran-US War Risk Tracker",
    page_icon="🌍",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "daily_signal.csv"

# =========================================================
# Load Data
# =========================================================

if not DATA_PATH.exists():
    st.error("data/daily_signal.csv 파일이 없습니다")
    st.stop()

df = pd.read_csv(DATA_PATH)

if df.empty:
    st.warning("데이터가 없습니다")
    st.stop()

df["generated_at_utc"] = pd.to_datetime(df["generated_at_utc"])

latest = df.iloc[-1]

# =========================================================
# Header
# =========================================================

st.title("🌍 Iran-US War Probability Tracker")

st.caption("AI 뉴스 분석 기반 전쟁 위험 대시보드")

# =========================================================
# War Probability Gauge
# =========================================================

war_prob = float(latest["war_smoothed_score"])

def gauge_color(score):

    if score < 25:
        return "green"

    if score < 50:
        return "yellow"

    if score < 75:
        return "orange"

    return "red"


fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=war_prob,
    title={'text': "War Probability"},
    gauge={
        'axis': {'range': [0,100]},
        'bar': {'color': gauge_color(war_prob)},
        'steps': [
            {'range':[0,25],'color':"#2ecc71"},
            {'range':[25,50],'color':"#f1c40f"},
            {'range':[50,75],'color':"#e67e22"},
            {'range':[75,100],'color':"#e74c3c"},
        ],
    }
))

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# KPI
# =========================================================

c1,c2,c3,c4 = st.columns(4)

c1.metric(
    "War Probability",
    f"{latest['war_smoothed_score']:.1f}"
)

c2.metric(
    "War Signal",
    f"{latest['war_batch_score']:.1f}"
)

c3.metric(
    "Trend",
    latest["trend_label_ko"]
)

c4.metric(
    "Relevant Articles",
    int(latest["relevant_articles"])
)

# =========================================================
# War Trend Chart
# =========================================================

st.subheader("📈 War Probability Trend")

chart_df = df.sort_values("generated_at_utc")
chart_df = chart_df.set_index("generated_at_utc")

st.line_chart(chart_df["war_smoothed_score"])

# =========================================================
# Escalation Signals
# =========================================================

st.subheader("⚠ Escalation Signals")

col1,col2,col3 = st.columns(3)

col1.metric(
    "Strike Articles",
    int(latest["strike_count"])
)

col2.metric(
    "Proxy Escalation",
    int(latest["proxy_escalation_count"])
)

col3.metric(
    "Escalation Articles",
    int(latest["escalation_count"])
)

# =========================================================
# Market Signals
# =========================================================

st.subheader("📊 Market Impact Signals")

m1,m2,m3,m4,m5 = st.columns(5)

m1.metric("Oil", latest["oil_signal"])
m2.metric("Defense", latest["defense_signal"])
m3.metric("Airline", latest["airline_signal"])
m4.metric("Equity", latest["equity_signal"])
m5.metric("Gold / Dollar", latest["gold_dollar_signal"])

# =========================================================
# AI Summary
# =========================================================

st.subheader("🧠 AI Interpretation")

st.info(latest["summary_ko"])

# =========================================================
# Raw Data
# =========================================================

with st.expander("Raw Data"):
    st.dataframe(df.tail(50))