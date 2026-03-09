import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np

# =========================================================
# Page
# =========================================================
st.set_page_config(
    page_title="Iran-US War Tracker",
    page_icon="🌍",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
HISTORY_PATH = BASE_DIR / "history.csv"

# =========================================================
# Utils
# =========================================================
def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    encodings = ["utf-8", "utf-8-sig", "cp949", "latin1"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip")
        except Exception:
            pass
    return pd.DataFrame()


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def to_numeric_series(series, default=0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    title_col = find_col(df, ["title", "headline", "translated_title"])
    summary_col = find_col(df, ["summary", "description", "content"])
    label_col = find_col(df, ["label", "category", "class"])
    strength_col = find_col(df, ["strength", "score", "risk_score"])
    link_col = find_col(df, ["link", "url", "article_url"])
    source_col = find_col(df, ["source", "publisher"])
    published_col = find_col(df, ["published_at", "published", "pub_date", "date", "datetime"])
    translated_col = find_col(df, ["translated_title_ko", "title_ko", "korean_title", "translated_title"])

    out = pd.DataFrame()
    out["title"] = df[title_col] if title_col else ""
    out["summary"] = df[summary_col] if summary_col else ""
    out["label"] = df[label_col] if label_col else "UNKNOWN"
    out["strength"] = df[strength_col] if strength_col else 0
    out["link"] = df[link_col] if link_col else ""
    out["source"] = df[source_col] if source_col else ""
    out["translated_title"] = df[translated_col] if translated_col else ""

    if published_col:
        parsed = pd.to_datetime(df[published_col], errors="coerce", utc=True)
        out["published_at"] = parsed
        try:
            out["published_at_kst"] = parsed.dt.tz_convert("Asia/Seoul")
        except Exception:
            out["published_at_kst"] = parsed
    else:
        out["published_at"] = pd.NaT
        out["published_at_kst"] = pd.NaT

    out["title"] = out["title"].fillna("").astype(str)
    out["summary"] = out["summary"].fillna("").astype(str)
    out["label"] = out["label"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
    out["strength"] = to_numeric_series(out["strength"], default=0)
    out["link"] = out["link"].fillna("").astype(str)
    out["source"] = out["source"].fillna("").astype(str)
    out["translated_title"] = out["translated_title"].fillna("").astype(str)

    out = out.drop_duplicates(subset=["title", "link"], keep="first").copy()

    if "published_at_kst" in out.columns:
        out = out.sort_values("published_at_kst", ascending=False, na_position="last")

    return out.reset_index(drop=True)


def label_color(label: str) -> str:
    palette = {
        "ESCALATION": "#ff5a5f",
        "MILITARY": "#ff9f43",
        "DIPLOMACY": "#2ecc71",
        "CEASEFIRE": "#00b894",
        "SANCTIONS": "#8e44ad",
        "POLITICS": "#3498db",
        "NEGOTIATION": "#1abc9c",
        "UNKNOWN": "#6c757d",
    }
    return palette.get(str(label).upper(), "#6c757d")


def strength_emoji(value: float) -> str:
    try:
        n = int(round(float(value)))
    except Exception:
        n = 0
    n = max(0, min(n, 5))
    return "🔥" * n if n > 0 else "·"


def tension_level(avg_strength: float) -> tuple[str, str]:
    if avg_strength >= 4.2:
        return "매우 높음", "#ff5a5f"
    if avg_strength >= 3.2:
        return "높음", "#ff9f43"
    if avg_strength >= 2.0:
        return "보통", "#f1c40f"
    return "낮음", "#2ecc71"


def pct_delta(current: float, previous: float) -> float:
    if previous == 0:
        if current == 0:
            return 0.0
        return 100.0
    return ((current - previous) / previous) * 100


def build_trend(df: pd.DataFrame, hours: int = 72, bin_hours: int = 6) -> pd.DataFrame:
    if df.empty or df["published_at_kst"].isna().all():
        return pd.DataFrame(columns=["time_bin", "avg_strength", "count"])

    temp = df.dropna(subset=["published_at_kst"]).copy()
    latest = temp["published_at_kst"].max()
    start_time = latest - pd.Timedelta(hours=hours)

    temp = temp[temp["published_at_kst"] >= start_time].copy()
    if temp.empty:
        return pd.DataFrame(columns=["time_bin", "avg_strength", "count"])

    temp["time_bin"] = temp["published_at_kst"].dt.floor(f"{bin_hours}H")

    trend = (
        temp.groupby("time_bin", as_index=False)
        .agg(
            avg_strength=("strength", "mean"),
            count=("title", "count"),
        )
        .sort_values("time_bin")
    )
    return trend


def build_strength_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"strength_bucket": [], "count": []})

    temp = df.copy()
    temp["strength_bucket"] = temp["strength"].round().clip(0, 5).astype(int)
    dist = (
        temp.groupby("strength_bucket", as_index=False)
        .agg(count=("title", "count"))
        .sort_values("strength_bucket")
    )

    full = pd.DataFrame({"strength_bucket": list(range(0, 6))})
    dist = full.merge(dist, on="strength_bucket", how="left").fillna(0)
    dist["count"] = dist["count"].astype(int)
    return dist


def format_time(ts):
    if pd.isna(ts):
        return "시간 정보 없음"
    return ts.strftime("%Y-%m-%d %H:%M")


# =========================================================
# Data load
# =========================================================
raw_df = safe_read_csv(HISTORY_PATH)
df = normalize_dataframe(raw_df)

# =========================================================
# CSS
# =========================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
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
        background: rgba(255,255,255,0.08);
        color: white;
    }

    .section-title {
        font-size: 1.12rem;
        font-weight: 800;
        margin-top: 1.15rem;
        margin-bottom: 0.7rem;
    }

    .kpi-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1rem 1rem 0.9rem 1rem;
        min-height: 108px;
        margin-bottom: 0.4rem;
    }

    .kpi-label {
        color: #9fb0c9;
        font-size: 0.82rem;
        margin-bottom: 0.28rem;
    }

    .kpi-value {
        font-size: 1.55rem;
        font-weight: 900;
        line-height: 1.1;
        color: #ffffff;
    }

    .kpi-delta {
        font-size: 0.82rem;
        margin-top: 0.35rem;
        color: #c9d5e7;
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
        padding: 0.28rem 0.6rem;
        border-radius: 999px;
        color: white;
        font-size: 0.72rem;
        font-weight: 800;
        margin-right: 0.38rem;
        margin-bottom: 0.38rem;
    }

    .title-ko {
        font-size: 1.04rem;
        font-weight: 900;
        line-height: 1.42;
        color: #ffffff;
        margin-top: 0.15rem;
        margin-bottom: 0.32rem;
    }

    .title-en {
        font-size: 0.9rem;
        color: #b9c5d8;
        line-height: 1.35;
        margin-bottom: 0.45rem;
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

    .mini-note {
        color: #97a6bc;
        font-size: 0.82rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 0.8rem;
        border-radius: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Empty state
# =========================================================
if df.empty:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">🌍 Iran-US War Tracker</div>
            <div class="hero-sub">history.csv를 찾지 못했거나 데이터를 읽지 못했습니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning("프로젝트 폴더에 history.csv가 있는지 확인해줘.")
    st.stop()

# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("필터")

all_labels = sorted(df["label"].dropna().unique().tolist())
selected_labels = st.sidebar.multiselect(
    "Label 선택",
    options=all_labels,
    default=all_labels,
)

min_strength_val = int(np.floor(df["strength"].min())) if not df.empty else 0
max_strength_val = int(np.ceil(df["strength"].max())) if not df.empty else 5
max_strength_val = max(max_strength_val, 5)

strength_filter = st.sidebar.slider(
    "최소 Strength",
    min_value=0,
    max_value=max_strength_val,
    value=0,
)

sort_by = st.sidebar.selectbox(
    "정렬 기준",
    ["최신순", "강도순"],
    index=0,
)

hours_window = st.sidebar.selectbox(
    "추세 기간",
    [24, 48, 72, 168],
    index=2,
    format_func=lambda x: f"최근 {x}시간" if x < 168 else "최근 7일",
)

show_count = st.sidebar.slider("기사 표시 개수", 5, 60, 18)
show_summary = st.sidebar.checkbox("요약문 표시", value=True)

filtered = df.copy()
filtered = filtered[filtered["label"].isin(selected_labels)]
filtered = filtered[filtered["strength"] >= strength_filter]

if sort_by == "최신순":
    filtered = filtered.sort_values("published_at_kst", ascending=False, na_position="last")
else:
    filtered = filtered.sort_values(["strength", "published_at_kst"], ascending=[False, False], na_position="last")

filtered = filtered.reset_index(drop=True)

# =========================================================
# Time windows
# =========================================================
valid_time_df = df.dropna(subset=["published_at_kst"]).copy()
latest_time = valid_time_df["published_at_kst"].max() if not valid_time_df.empty else None

if latest_time is not None:
    recent_24h = df[df["published_at_kst"] >= latest_time - pd.Timedelta(hours=24)].copy()
    prev_24h = df[
        (df["published_at_kst"] < latest_time - pd.Timedelta(hours=24)) &
        (df["published_at_kst"] >= latest_time - pd.Timedelta(hours=48))
    ].copy()
else:
    recent_24h = df.copy()
    prev_24h = pd.DataFrame(columns=df.columns)

recent_count = len(recent_24h)
prev_count = len(prev_24h)

recent_avg_strength = float(recent_24h["strength"].mean()) if not recent_24h.empty else 0.0
prev_avg_strength = float(prev_24h["strength"].mean()) if not prev_24h.empty else 0.0

count_delta_pct = pct_delta(recent_count, prev_count)
strength_delta = recent_avg_strength - prev_avg_strength

dominant_label = (
    recent_24h["label"].value_counts().idxmax()
    if not recent_24h.empty and recent_24h["label"].notna().any()
    else "UNKNOWN"
)

level_text, level_color = tension_level(recent_avg_strength)
latest_text = format_time(latest_time) + " KST" if latest_time is not None else "시간 정보 없음"

# =========================================================
# Header
# =========================================================
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">🌍 Iran-US War Tracker</div>
        <div class="hero-sub">
            실시간 뉴스 흐름을 바탕으로 긴장도와 변화 방향을 빠르게 보는 대시보드
        </div>
        <div class="status-pill" style="background:{level_color};">
            현재 긴장도: {level_text}
        </div>
        <div class="mini-note" style="margin-top:0.65rem;">
            마지막 기사 시각: {latest_text}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# KPI cards
# =========================================================
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">최근 24시간 기사 수</div>
            <div class="kpi-value">{recent_count}</div>
            <div class="kpi-delta">직전 24시간 대비 {count_delta_pct:+.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">최근 24시간 평균 Strength</div>
            <div class="kpi-value">{recent_avg_strength:.2f}</div>
            <div class="kpi-delta">직전 24시간 대비 {strength_delta:+.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    dominant_color = label_color(dominant_label)
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">우세 Label</div>
            <div class="kpi-value" style="font-size:1.25rem;">{dominant_label}</div>
            <div class="kpi-delta">
                <span class="badge" style="background:{dominant_color}; margin-top:0.15rem;">{dominant_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k4:
    latest_strength = float(filtered.iloc[0]["strength"]) if not filtered.empty else 0.0
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">가장 최근 기사 강도</div>
            <div class="kpi-value">{latest_strength:.1f}</div>
            <div class="kpi-delta">{strength_emoji(latest_strength)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# Trends
# =========================================================
st.markdown('<div class="section-title">📈 추세 분석</div>', unsafe_allow_html=True)

trend_df = build_trend(df, hours=hours_window, bin_hours=6)
strength_dist_df = build_strength_distribution(filtered)

left_col, right_col = st.columns([1.55, 1])

with left_col:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.subheader("긴장도 추세")
    if not trend_df.empty:
        line_df = trend_df.set_index("time_bin")[["avg_strength"]]
        st.line_chart(line_df, height=290, use_container_width=True)
        st.caption("6시간 단위 평균 Strength")
    else:
        st.info("추세선에 필요한 시간 데이터가 부족합니다.")
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.subheader("Label 분포")
    label_dist = filtered["label"].value_counts()
    if not label_dist.empty:
        st.bar_chart(label_dist, height=290, use_container_width=True)
        st.caption("현재 필터 기준 기사 수")
    else:
        st.info("표시할 Label 데이터가 없습니다.")
    st.markdown('</div>', unsafe_allow_html=True)

left_col2, right_col2 = st.columns([1.55, 1])

with left_col2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.subheader("기사량 추세")
    if not trend_df.empty:
        area_df = trend_df.set_index("time_bin")[["count"]]
        st.area_chart(area_df, height=240, use_container_width=True)
        st.caption("6시간 단위 기사 수")
    else:
        st.info("기사량 추세를 표시할 데이터가 없습니다.")
    st.markdown('</div>', unsafe_allow_html=True)

with right_col2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.subheader("Strength 분포")
    if not strength_dist_df.empty:
        chart_df = strength_dist_df.set_index("strength_bucket")[["count"]]
        st.bar_chart(chart_df, height=240, use_container_width=True)
        st.caption("반올림한 Strength 구간별 기사 수")
    else:
        st.info("Strength 분포 데이터가 없습니다.")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# Quick insights
# =========================================================
st.markdown('<div class="section-title">🧭 빠른 해석</div>', unsafe_allow_html=True)

insight_cols = st.columns(3)

trend_message = "데이터 부족"
if not trend_df.empty and len(trend_df) >= 2:
    first_val = float(trend_df["avg_strength"].iloc[0])
    last_val = float(trend_df["avg_strength"].iloc[-1])
    diff = last_val - first_val
    if diff > 0.35:
        trend_message = f"긴장도 상승 흐름 (+{diff:.2f})"
    elif diff < -0.35:
        trend_message = f"긴장도 완화 흐름 ({diff:.2f})"
    else:
        trend_message = f"큰 변화 없음 ({diff:+.2f})"

recent_escalation_share = 0.0
if not recent_24h.empty:
    recent_escalation_share = (
        (recent_24h["label"] == "ESCALATION").mean() * 100
    )

top_source = (
    filtered["source"].value_counts().idxmax()
    if not filtered.empty and filtered["source"].astype(str).str.strip().ne("").any()
    else "정보 없음"
)

with insight_cols[0]:
    st.metric("추세 해석", trend_message)

with insight_cols[1]:
    st.metric("ESCALATION 비중", f"{recent_escalation_share:.1f}%")

with insight_cols[2]:
    st.metric("가장 많이 보이는 출처", top_source)

# =========================================================
# Table
# =========================================================
with st.expander("데이터 테이블 보기"):
    preview = filtered.copy()
    preview["published_at_kst"] = preview["published_at_kst"].apply(format_time)
    view_cols = [c for c in ["published_at_kst", "label", "strength", "source", "title", "link"] if c in preview.columns]
    st.dataframe(preview[view_cols].head(200), use_container_width=True)

# =========================================================
# News cards
# =========================================================
st.markdown('<div class="section-title">📰 최신 기사</div>', unsafe_allow_html=True)

cards_df = filtered.head(show_count).copy()

if cards_df.empty:
    st.warning("현재 필터 조건에 맞는 기사가 없습니다.")
else:
    for _, row in cards_df.iterrows():
        badge_color = label_color(row["label"])

        title_ko = row["translated_title"].strip() if row["translated_title"].strip() else row["title"]
        title_en = row["title"].strip() if title_ko.strip() != row["title"].strip() else ""
        summary = row["summary"].strip()
        source_text = row["source"].strip()
        time_text = format_time(row["published_at_kst"])
        link = row["link"].strip()

        meta_parts = []
        if source_text:
            meta_parts.append(source_text)
        if time_text:
            meta_parts.append(time_text)

        meta_text = " · ".join(meta_parts)

        link_html = f'<a href="{link}" target="_blank">원문 보기 ↗</a>' if link else ""

        st.markdown(
            f"""
            <div class="news-card">
                <span class="badge" style="background:{badge_color};">{row["label"]}</span>
                <span class="badge" style="background:#334155;">Strength {float(row["strength"]):.1f}</span>
                <span class="badge" style="background:#1f2937;">{strength_emoji(row["strength"])}</span>

                <div class="title-ko">{title_ko}</div>
                {"<div class='title-en'>" + title_en + "</div>" if title_en else ""}
                {"<div class='summary'>" + summary + "</div>" if (show_summary and summary) else ""}

                <div class="meta">
                    {meta_text}
                    {" · " + link_html if link_html else ""}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )