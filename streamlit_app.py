import json
import os
import re
import html
from pathlib import Path
from textwrap import dedent

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI

# =========================================================
# Page
# =========================================================
st.set_page_config(
    page_title="Iran-US War Risk Tracker",
    page_icon="🌍",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

DAILY_SIGNAL_PATH = DATA_DIR / "daily_signal.csv"
CLASSIFIED_JSON_PATH = DATA_DIR / "classified_articles.json"
LATEST_REPORT_PATH = OUTPUT_DIR / "latest_report.json"
TITLE_KO_CACHE_PATH = DATA_DIR / "title_ko_cache.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini").strip()

# =========================================================
# Helpers
# =========================================================
def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    encodings = ["utf-8", "utf-8-sig", "cp949", "latin1"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip")
        except Exception:
            continue
    return pd.DataFrame()


def safe_read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def clean_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def to_kst(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return parsed.dt.tz_convert("Asia/Seoul")
    except Exception:
        return parsed


def risk_color(score: float) -> str:
    if score < 25:
        return "#2ecc71"
    if score < 50:
        return "#f1c40f"
    if score < 75:
        return "#e67e22"
    return "#e74c3c"


def label_color(label: str) -> str:
    palette = {
        "negotiation": "#1abc9c",
        "ceasefire": "#00b894",
        "deescalation_signal": "#2ecc71",
        "escalation": "#ff9f43",
        "proxy_escalation": "#8e44ad",
        "strike_or_retaliation": "#ff5a5f",
        "irrelevant": "#6c757d",
    }
    return palette.get(str(label).strip().lower(), "#6c757d")


def label_ko(label: str) -> str:
    mapping = {
        "negotiation": "협상",
        "ceasefire": "휴전",
        "deescalation_signal": "완화 신호",
        "escalation": "긴장 고조",
        "proxy_escalation": "대리세력 확전",
        "strike_or_retaliation": "직접 공격/보복",
        "irrelevant": "무관",
    }
    return mapping.get(str(label).strip().lower(), label)


def strength_emoji(value: float) -> str:
    try:
        v = float(value)
        n = int(round(v * 5)) if v <= 1 else int(round(v))
    except Exception:
        n = 0
    n = max(0, min(n, 5))
    return "🔥" * n if n > 0 else "·"


def format_time(ts) -> str:
    if pd.isna(ts):
        return "시간 정보 없음"
    return ts.strftime("%Y-%m-%d %H:%M")


def build_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "generated_at_kst" not in df.columns:
        return pd.DataFrame()

    temp = df.copy()
    temp = temp.dropna(subset=["generated_at_kst", "war_smoothed_score"])
    if temp.empty:
        return pd.DataFrame()

    temp["date"] = temp["generated_at_kst"].dt.strftime("%m-%d")
    temp["hour"] = temp["generated_at_kst"].dt.hour
    temp["score"] = pd.to_numeric(temp["war_smoothed_score"], errors="coerce")

    pivot = temp.pivot_table(
        index="date",
        columns="hour",
        values="score",
        aggfunc="mean",
    ).sort_index()

    return pivot


@st.cache_resource
def get_openai_client():
    if not OPENAI_API_KEY:
        return None
    try:
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        return None


def translate_title_to_korean(title: str, cache: dict) -> str:
    title = clean_text(title)
    if not title:
        return ""

    if title in cache:
        return clean_text(cache[title])

    client = get_openai_client()
    if client is None:
        return title

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Translate English news headlines into natural Korean. "
                        "Return only the translated headline text. "
                        "Do not include HTML, markdown, labels, explanation, or quotation marks."
                    ),
                },
                {"role": "user", "content": title},
            ],
        )
        out = clean_text((getattr(resp, "output_text", "") or "").strip())
        if out:
            cache[title] = out
            return out
    except Exception:
        pass

    return title


def get_title_ko(row: pd.Series, cache: dict) -> str:
    for col in ["translated_title_ko", "title_ko", "translated_title"]:
        if col in row and str(row[col]).strip():
            return clean_text(str(row[col]).strip())
    return clean_text(translate_title_to_korean(str(row.get("title", "")), cache))


# =========================================================
# Load data
# =========================================================
signal_df = safe_read_csv(DAILY_SIGNAL_PATH)
classified_articles = safe_read_json(CLASSIFIED_JSON_PATH, default=[])
latest_report = safe_read_json(LATEST_REPORT_PATH, default={})
title_ko_cache = safe_read_json(TITLE_KO_CACHE_PATH, default={})

if signal_df.empty:
    st.error("data/daily_signal.csv 파일이 없거나 비어 있습니다.")
    st.stop()

signal_df["generated_at_utc"] = pd.to_datetime(signal_df["generated_at_utc"], errors="coerce", utc=True)
signal_df["generated_at_kst"] = to_kst(signal_df["generated_at_utc"])
signal_df = signal_df.sort_values("generated_at_utc").reset_index(drop=True)

latest = signal_df.iloc[-1].copy()

articles_df = pd.DataFrame(classified_articles)
if not articles_df.empty:
    if "published_at" in articles_df.columns:
        articles_df["published_at_kst"] = to_kst(articles_df["published_at"])
    else:
        articles_df["published_at_kst"] = pd.NaT

    if "classification" in articles_df.columns:
        articles_df["label"] = articles_df["classification"].apply(
            lambda x: x.get("label", "") if isinstance(x, dict) else ""
        )
        articles_df["strength"] = articles_df["classification"].apply(
            lambda x: x.get("strength", 0.0) if isinstance(x, dict) else 0.0
        )
        articles_df["reason"] = articles_df["classification"].apply(
            lambda x: x.get("reason", "") if isinstance(x, dict) else ""
        )
        articles_df["event_key"] = articles_df["classification"].apply(
            lambda x: x.get("event_key", "") if isinstance(x, dict) else ""
        )
    else:
        articles_df["label"] = ""
        articles_df["strength"] = 0.0
        articles_df["reason"] = ""
        articles_df["event_key"] = ""

    if "source_weight" not in articles_df.columns:
        articles_df["source_weight"] = 1.0

    for col in [
        "title",
        "summary",
        "link",
        "source",
        "label",
        "reason",
        "event_key",
        "translated_title_ko",
        "title_ko",
        "translated_title",
    ]:
        if col not in articles_df.columns:
            articles_df[col] = ""
        articles_df[col] = articles_df[col].fillna("").astype(str)

    articles_df["strength"] = pd.to_numeric(articles_df["strength"], errors="coerce").fillna(0.0)
    articles_df["source_weight"] = pd.to_numeric(articles_df["source_weight"], errors="coerce").fillna(1.0)
    articles_df["weighted_strength"] = articles_df["strength"] * articles_df["source_weight"]

    articles_df["title_ko_display"] = articles_df.apply(lambda row: get_title_ko(row, title_ko_cache), axis=1)
    save_json(TITLE_KO_CACHE_PATH, title_ko_cache)

    articles_df = articles_df.sort_values(
        ["published_at_kst", "weighted_strength"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

# =========================================================
# CSS
# =========================================================
st.markdown(
    dedent("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }
    .hero {
        padding: 1.3rem 1.3rem 1.1rem 1.3rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #0f172a 0%, #172033 55%, #1d2a44 100%);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 10px 30px rgba(0,0,0,0.20);
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 900;
        line-height: 1.15;
        margin-bottom: 0.25rem;
        color: #ffffff;
    }
    .hero-sub {
        color: #c7d2e1;
        font-size: 0.98rem;
        line-height: 1.5;
    }
    .status-pill {
        display: inline-block;
        margin-top: 0.8rem;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        font-weight: 800;
        font-size: 0.85rem;
        color: white;
    }
    .mini-note {
        color: #97a6bc;
        font-size: 0.82rem;
        margin-top: 0.6rem;
    }
    .section-title {
        font-size: 1.12rem;
        font-weight: 800;
        margin-top: 1.15rem;
        margin-bottom: 0.7rem;
    }
    .chart-card {
        padding: 0.9rem 0.9rem 0.5rem 0.9rem;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.02);
        margin-bottom: 0.9rem;
    }
    .news-card {
        padding: 1rem 1rem 0.95rem 1rem;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0.015));
        margin-bottom: 0.85rem;
    }
    .badge {
        display: inline-block;
        padding: 0.30rem 0.62rem;
        border-radius: 999px;
        color: white;
        font-size: 0.73rem;
        font-weight: 800;
        margin-right: 0.38rem;
        margin-bottom: 0.38rem;
    }
    .title-ko {
        font-size: 1.05rem;
        font-weight: 900;
        line-height: 1.42;
        color: #ffffff;
        margin-top: 0.12rem;
        margin-bottom: 0.25rem;
    }
    .title-en {
        font-size: 0.88rem;
        color: #aebcd3;
        line-height: 1.4;
        margin-bottom: 0.42rem;
    }
    .summary {
        font-size: 0.92rem;
        color: #d7deea;
        line-height: 1.55;
        margin-top: 0.35rem;
        margin-bottom: 0.3rem;
    }
    .meta {
        color: #9fb0c9;
        font-size: 0.82rem;
        margin-top: 0.45rem;
    }
    .meta a {
        color: #8ecbff !important;
        text-decoration: none !important;
        font-weight: 700;
    }
    .filter-note {
        color: #95a6c0;
        font-size: 0.84rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """),
    unsafe_allow_html=True,
)

# =========================================================
# Header
# =========================================================
trend_risk = float(pd.to_numeric(latest.get("war_smoothed_score", 0), errors="coerce"))
status_color = risk_color(trend_risk)
latest_time = latest.get("generated_at_kst", pd.NaT)
latest_time_text = format_time(latest_time) + " KST" if not pd.isna(latest_time) else "시간 정보 없음"

st.markdown(
    dedent(f"""
    <div class="hero">
        <div class="hero-title">🌍 Iran-US War Risk Tracker</div>
        <div class="hero-sub">
            AI 뉴스 분류 기반 지정학 리스크 대시보드
        </div>
        <div class="status-pill" style="background:{status_color};">
            현재 추세 위험도: {trend_risk:.1f}
        </div>
        <div class="mini-note">
            마지막 업데이트: {latest_time_text}
        </div>
    </div>
    """),
    unsafe_allow_html=True,
)

# =========================================================
# KPI
# =========================================================
k1, k2, k3, k4 = st.columns(4)
k1.metric("추세 위험도", f"{float(latest.get('war_smoothed_score', 0)):.1f}")
k2.metric("즉각 위험도", f"{float(latest.get('war_batch_score', 0)):.1f}")
k3.metric("추세", str(latest.get("trend_label_ko", latest.get("trend_label", "-"))))
k4.metric("관련 기사 수", int(pd.to_numeric(latest.get("relevant_articles", 0), errors="coerce")))

# =========================================================
# AI Summary
# =========================================================
st.markdown('<div class="section-title">🧠 AI Interpretation</div>', unsafe_allow_html=True)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
summary_text = clean_text(
    latest_report.get("summary_ko", "") or latest.get("summary_ko", "요약 정보가 없습니다.")
)
alert_text = clean_text(
    latest_report.get("alert_message", "") or latest.get("alert_message", "")
)

if alert_text:
    st.warning(alert_text)

st.info(summary_text)

s1, s2, s3 = st.columns(3)
s1.metric("직접 공격/보복", int(pd.to_numeric(latest.get("strike_count", 0), errors="coerce")))
s2.metric("대리세력 확전", int(pd.to_numeric(latest.get("proxy_escalation_count", 0), errors="coerce")))
s3.metric("긴장 고조 기사", int(pd.to_numeric(latest.get("escalation_count", 0), errors="coerce")))
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# Trend + Heatmap
# =========================================================
st.markdown('<div class="section-title">📈 위험도 추세</div>', unsafe_allow_html=True)

c1, c2 = st.columns([1.35, 1])

with c1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    trend_df = signal_df.copy().dropna(subset=["generated_at_kst"])

    if not trend_df.empty:
        plot_df = trend_df.sort_values("generated_at_kst").copy()
        plot_df["generated_at_kst"] = pd.to_datetime(plot_df["generated_at_kst"], errors="coerce")
        plot_df = plot_df.dropna(subset=["generated_at_kst"])
        plot_df["time_bin"] = plot_df["generated_at_kst"].dt.floor("3H")

        plot_df = (
            plot_df.groupby("time_bin", as_index=False)
            .agg(
                war_batch_score=("war_batch_score", "last"),
                war_smoothed_score=("war_smoothed_score", "last"),
            )
            .sort_values("time_bin")
        )

        fig_trend = go.Figure()

        fig_trend.add_trace(
            go.Scatter(
                x=plot_df["time_bin"],
                y=pd.to_numeric(plot_df["war_batch_score"], errors="coerce"),
                mode="lines+markers",
                name="즉각 위험도",
                line=dict(color="#e74c3c", width=3),
                marker=dict(size=6, color="#e74c3c"),
            )
        )

        fig_trend.add_trace(
            go.Scatter(
                x=plot_df["time_bin"],
                y=pd.to_numeric(plot_df["war_smoothed_score"], errors="coerce"),
                mode="lines+markers",
                name="추세 위험도",
                line=dict(color="#3498db", width=3),
                marker=dict(size=6, color="#3498db"),
            )
        )

        fig_trend.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="시간",
            yaxis_title="위험도 점수",
            yaxis=dict(range=[0, 100]),
            hovermode="x unified",
        )

        fig_trend.update_xaxes(
            dtick=3 * 60 * 60 * 1000,
            tickformat="%m-%d %H:%M",
        )

        st.plotly_chart(fig_trend, use_container_width=True)
        st.caption("빨간선=즉각 위험도 / 파란선=추세 위험도 / 3시간 배치 기준")
    else:
        st.info("추세 데이터를 표시할 수 없습니다.")
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    heatmap_pivot = build_heatmap(signal_df.tail(120))

    if not heatmap_pivot.empty:
        heatmap_fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_pivot.values,
                x=[f"{int(h):02d}" for h in heatmap_pivot.columns],
                y=heatmap_pivot.index.tolist(),
                colorscale=[
                    [0.00, "#2ecc71"],
                    [0.25, "#f1c40f"],
                    [0.50, "#e67e22"],
                    [1.00, "#e74c3c"],
                ],
                zmin=0,
                zmax=100,
                colorbar=dict(title="Risk"),
            )
        )
        heatmap_fig.update_layout(
            title="추세 위험도 Heatmap",
            height=320,
            margin=dict(l=10, r=10, t=45, b=10),
            xaxis_title="Hour (KST)",
            yaxis_title="Date",
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)
        st.caption("날짜/시간대별 추세 위험도 분포")
    else:
        st.info("heatmap 데이터를 표시할 수 없습니다.")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# Market Signals
# =========================================================
st.markdown('<div class="section-title">📊 Market Impact Signals</div>', unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Oil", str(latest.get("oil_signal", "-")))
m2.metric("Defense", str(latest.get("defense_signal", "-")))
m3.metric("Airline", str(latest.get("airline_signal", "-")))
m4.metric("Equity", str(latest.get("equity_signal", "-")))
m5.metric("Gold / Dollar", str(latest.get("gold_dollar_signal", "-")))

# =========================================================
# Real-time news cards
# =========================================================
st.markdown('<div class="section-title">📰 실시간 뉴스 카드</div>', unsafe_allow_html=True)

if articles_df.empty:
    st.info("data/classified_articles.json에 표시할 기사 데이터가 아직 없습니다.")
else:
    label_options = [
        ("all", "전체"),
        ("negotiation", "협상"),
        ("ceasefire", "휴전"),
        ("deescalation_signal", "완화 신호"),
        ("escalation", "긴장 고조"),
        ("proxy_escalation", "대리세력 확전"),
        ("strike_or_retaliation", "직접 공격/보복"),
    ]

    if "selected_label_filter" not in st.session_state:
        st.session_state.selected_label_filter = "all"

    st.markdown('<div class="filter-note">라벨별 빠른 필터</div>', unsafe_allow_html=True)
    button_cols = st.columns(len(label_options))

    for idx, (value, text) in enumerate(label_options):
        with button_cols[idx]:
            if st.button(text, use_container_width=True):
                st.session_state.selected_label_filter = value

    selected_label = st.session_state.selected_label_filter

    filtered_articles = articles_df.copy()
    filtered_articles = filtered_articles[filtered_articles["label"].str.lower() != "irrelevant"]

    if selected_label != "all":
        filtered_articles = filtered_articles[filtered_articles["label"].str.lower() == selected_label]

    show_count = st.slider("표시 기사 수", min_value=5, max_value=30, value=12)
    st.caption(f"현재 필터: {dict(label_options).get(selected_label, '전체')}")

    if filtered_articles.empty:
        st.info("현재 필터에 해당하는 관련 기사 카드가 없습니다.")
    else:
        for _, row in filtered_articles.head(show_count).iterrows():
            badge_color = label_color(row["label"])
            label_text = label_ko(row["label"])
            strength_text = float(row["strength"])

            source_text = clean_text(row.get("source", ""))
            time_text = format_time(row["published_at_kst"])
            reason_text = clean_text(row.get("reason", ""))
            link = str(row.get("link", "")).strip()
            summary = clean_text(row.get("summary", ""))
            title_ko = clean_text(row.get("title_ko_display", ""))
            title_en = clean_text(row.get("title", ""))

            meta_parts = []
            if source_text:
                meta_parts.append(source_text)
            if time_text:
                meta_parts.append(time_text)

            meta_text = " · ".join(meta_parts)
            link_html = f'<a href="{link}" target="_blank">원문 보기 ↗</a>' if link else ""

            title_en_html = f"<div class='title-en'>{title_en}</div>" if title_en and title_en != title_ko else ""
            summary_html = f"<div class='summary'>{summary}</div>" if summary else ""
            reason_html = f"<div class='summary'><b>판단 근거:</b> {reason_text}</div>" if reason_text else ""
            meta_link_html = f" · {link_html}" if link_html else ""

            card_html = dedent(f"""
            <div class="news-card">
                <span class="badge" style="background:{badge_color};">{label_text}</span>
                <span class="badge" style="background:#334155;">Strength {strength_text:.2f}</span>
                <span class="badge" style="background:#1f2937;">{strength_emoji(strength_text)}</span>

                <div class="title-ko">{title_ko}</div>
                {title_en_html}
                {summary_html}
                {reason_html}

                <div class="meta">
                    {meta_text}{meta_link_html}
                </div>
            </div>
            """)

            st.markdown(card_html, unsafe_allow_html=True)

# =========================================================
# Raw data
# =========================================================
with st.expander("Raw signal data"):
    preview_df = signal_df.copy()
    preview_df["generated_at_kst"] = preview_df["generated_at_kst"].apply(format_time)
    show_cols = [
        c for c in [
            "generated_at_kst",
            "war_batch_score",
            "war_smoothed_score",
            "trend_label_ko",
            "relevant_articles",
            "strike_count",
            "proxy_escalation_count",
            "escalation_count",
            "oil_signal",
            "defense_signal",
        ] if c in preview_df.columns
    ]
    st.dataframe(preview_df[show_cols].tail(100), use_container_width=True)

with st.expander("Raw classified articles"):
    if articles_df.empty:
        st.write("기사 데이터 없음")
    else:
        preview_cols = [
            c for c in [
                "published_at_kst",
                "label",
                "strength",
                "source",
                "title_ko_display",
                "title",
                "reason",
                "link",
            ] if c in articles_df.columns
        ]
        temp = articles_df.copy()
        if "published_at_kst" in temp.columns:
            temp["published_at_kst"] = temp["published_at_kst"].apply(format_time)
        st.dataframe(temp[preview_cols].head(100), use_container_width=True)