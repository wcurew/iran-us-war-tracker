# streamlit_app.py

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# =========================
# 기본 설정
# =========================
st.set_page_config(
    page_title="Iran-US War Tracker",
    page_icon="🔥",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

CSV_PATH = DATA_DIR / "daily_signal.csv"
ARTICLES_PATH = DATA_DIR / "classified_articles.json"
REPORT_PATH = OUTPUT_DIR / "latest_report.json"
TITLE_CACHE_PATH = DATA_DIR / "title_ko_cache.json"


# =========================
# 유틸
# =========================
def safe_load_json(path: Path, default: Any):
    try:
        if not path.exists():
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def parse_dt_column(df: pd.DataFrame) -> pd.Series:
    """
    timestamp / collected_at / batch_time / datetime / created_at 등
    가능한 시간 컬럼을 찾아 datetime으로 반환.
    """
    if df.empty:
        return pd.Series(dtype="datetime64[ns]")

    candidate_cols = [
        "timestamp",
        "collected_at",
        "batch_time",
        "datetime",
        "created_at",
        "published_at",
        "time",
        "date",
    ]

    for col in candidate_cols:
        if col in df.columns:
            return pd.to_datetime(df[col], errors="coerce")

    return pd.Series([pd.NaT] * len(df), index=df.index)


def first_existing(d: dict, keys: list[str], default: Any = "") -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def is_mostly_ascii(text: str) -> bool:
    if not text:
        return True
    ascii_count = sum(1 for ch in text if ord(ch) < 128)
    return ascii_count / max(len(text), 1) >= 0.85


def has_hangul(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text))


def looks_like_broken_or_code(text: str) -> bool:
    if not text:
        return True

    lowered = text.lower()

    suspicious_patterns = [
        "<div",
        "</div",
        "<span",
        "</span",
        "<p>",
        "</p>",
        "<br",
        "function(",
        "return {",
        "json",
        '{"',
        '["',
        "traceback",
        "error:",
        "nan",
        "null",
        "none",
    ]

    if any(p in lowered for p in suspicious_patterns):
        return True

    # 괄호/기호 비율이 너무 높으면 깨진 문자열로 간주
    special_count = sum(1 for ch in text if ch in "{}[]<>|\\`~")
    if special_count >= 4:
        return True

    return False


def sanitize_korean_title(title_ko: Any, title_en: Any) -> str:
    """
    번역 실패 / 영어 그대로 / cache 오염 / HTML / 이상 문자열 방어
    최종적으로 유효한 한글 제목이 아니면 '번역 없음'
    """
    ko = normalize_text(title_ko)
    en = normalize_text(title_en)

    if not ko:
        return "번역 없음"

    if looks_like_broken_or_code(ko):
        return "번역 없음"

    if en and ko.strip().lower() == en.strip().lower():
        return "번역 없음"

    # 한글이 전혀 없고 ASCII 비율 높으면 번역 실패로 간주
    if not has_hangul(ko) and is_mostly_ascii(ko):
        return "번역 없음"

    # 너무 짧거나 의미 없는 값 차단
    if ko in {"-", "--", "N/A", "n/a", "없음"}:
        return "번역 없음"

    return ko


def sanitize_summary(text: Any) -> str:
    s = normalize_text(text)
    if not s or looks_like_broken_or_code(s):
        return "요약 없음"
    return s


def sanitize_reason(text: Any) -> str:
    s = normalize_text(text)
    if not s or looks_like_broken_or_code(s):
        return "판단 근거 없음"
    return s


def sanitize_source(text: Any) -> str:
    s = normalize_text(text)
    return s if s else "출처 미상"


def sanitize_link(text: Any) -> str:
    s = normalize_text(text)
    if s.startswith("http://") or s.startswith("https://"):
        return s
    return ""


def format_strength(value: Any) -> str:
    if value is None:
        return "-"
    try:
        v = float(value)
        return f"{v:.2f}"
    except Exception:
        return normalize_text(value) or "-"


def format_score(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "-"


def fire_emoji_from_strength(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return "🔥"

    if v >= 0.90:
        return "🔥🔥🔥"
    if v >= 0.70:
        return "🔥🔥"
    if v >= 0.40:
        return "🔥"
    return "·"


def format_datetime(value: Any) -> str:
    if value is None:
        return "-"
    try:
        dt = pd.to_datetime(value, errors="coerce")
        if pd.isna(dt):
            return "-"
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "-"


# =========================
# 데이터 로드
# =========================
@st.cache_data(ttl=300)
def load_history() -> pd.DataFrame:
    df = safe_read_csv(CSV_PATH)
    if df.empty:
        return df

    dt = parse_dt_column(df)
    df = df.copy()
    df["timestamp_parsed"] = dt
    df = df.dropna(subset=["timestamp_parsed"]).sort_values("timestamp_parsed")

    # 숫자 컬럼 정리
    for col in ["war_batch_score", "war_smoothed_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data(ttl=300)
def load_articles() -> list[dict]:
    data = safe_load_json(ARTICLES_PATH, default=[])
    if not isinstance(data, list):
        return []

    # title_ko_cache.json 이 있더라도 직접 신뢰하지 않고,
    # 기사 데이터 안의 title_ko도 sanitize해서만 사용
    _ = safe_load_json(TITLE_CACHE_PATH, default={})  # 깨져도 무시

    cleaned: list[dict] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        title_en = first_existing(item, ["title", "title_en", "headline"], "")
        title_ko_raw = first_existing(item, ["title_ko", "translated_title_ko"], "")

        article = {
            "label": normalize_text(first_existing(item, ["label", "classification"], "")) or "UNKNOWN",
            "strength": first_existing(item, ["strength", "score", "confidence"], None),
            "title_en": normalize_text(title_en) or "제목 없음",
            "title_ko": sanitize_korean_title(title_ko_raw, title_en),
            "summary": sanitize_summary(first_existing(item, ["summary", "summary_en", "desc"], "")),
            "reason": sanitize_reason(
                first_existing(item, ["reason", "rationale", "judgement_reason", "why"], "")
            ),
            "source": sanitize_source(first_existing(item, ["source", "publisher", "press"], "")),
            "published_at": first_existing(item, ["published_at", "time", "created_at"], None),
            "link": sanitize_link(first_existing(item, ["link", "url", "original_link"], "")),
        }
        cleaned.append(article)

    # 최신순 정렬
    def sort_key(x: dict):
        try:
            ts = pd.to_datetime(x.get("published_at"), errors="coerce")
            if pd.isna(ts):
                return pd.Timestamp.min
            return ts
        except Exception:
            return pd.Timestamp.min

    cleaned.sort(key=sort_key, reverse=True)
    return cleaned


@st.cache_data(ttl=300)
def load_report() -> dict:
    data = safe_load_json(REPORT_PATH, default={})
    return data if isinstance(data, dict) else {}


# =========================
# 가공
# =========================
def prepare_chart_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "timestamp_parsed" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "war_batch_score", "war_smoothed_score"])

    chart_df = df.copy()
    chart_df = chart_df.set_index("timestamp_parsed")

    keep_cols = [c for c in ["war_batch_score", "war_smoothed_score"] if c in chart_df.columns]
    if not keep_cols:
        return pd.DataFrame(columns=["timestamp", "war_batch_score", "war_smoothed_score"])

    chart_df = chart_df[keep_cols]

    # 3시간 배치 기준으로 집계
    # 배치 데이터라면 거의 그대로 유지되고, 더 촘촘한 경우엔 3시간 단위 last
    chart_df = chart_df.resample("3H").last().dropna(how="all").reset_index()
    chart_df = chart_df.rename(columns={"timestamp_parsed": "timestamp"})

    return chart_df


def latest_scores_from_df(df: pd.DataFrame) -> tuple[Any, Any, str]:
    if df.empty:
        return None, None, "-"

    last = df.iloc[-1]
    batch = last["war_batch_score"] if "war_batch_score" in df.columns else None
    smooth = last["war_smoothed_score"] if "war_smoothed_score" in df.columns else None
    ts = format_datetime(last.get("timestamp_parsed"))
    return batch, smooth, ts


def latest_from_report_or_df(report: dict, df: pd.DataFrame) -> tuple[Any, Any]:
    batch = report.get("war_batch_score", report.get("immediate_risk"))
    smooth = report.get("war_smoothed_score", report.get("trend_risk"))

    if batch is None or smooth is None:
        df_batch, df_smooth, _ = latest_scores_from_df(df)
        batch = batch if batch is not None else df_batch
        smooth = smooth if smooth is not None else df_smooth

    return batch, smooth


# =========================
# 시각화
# =========================
def render_score_chart(chart_df: pd.DataFrame):
    st.subheader("위험도 추세")

    if chart_df.empty:
        st.info("표시할 위험도 이력 데이터가 없습니다.")
        return

    fig, ax = plt.subplots(figsize=(12, 4.8))

    if "war_batch_score" in chart_df.columns:
        ax.plot(
            chart_df["timestamp"],
            chart_df["war_batch_score"],
            label="즉각 위험도",
            linewidth=2.2,
            color="red",
        )

    if "war_smoothed_score" in chart_df.columns:
        ax.plot(
            chart_df["timestamp"],
            chart_df["war_smoothed_score"],
            label="추세 위험도",
            linewidth=2.2,
            color="blue",
        )

    ax.set_title("3시간 배치 기준 위험도 추세", fontsize=14)
    ax.set_xlabel("시간")
    ax.set_ylabel("점수")
    ax.grid(alpha=0.25)
    ax.legend()

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)


def render_news_cards(articles: list[dict], limit: int):
    st.subheader("실시간 뉴스 카드")

    if not articles:
        st.info("표시할 기사 데이터가 없습니다.")
        return

    for idx, article in enumerate(articles[:limit], start=1):
        label = normalize_text(article.get("label")) or "UNKNOWN"
        strength = format_strength(article.get("strength"))
        fire = fire_emoji_from_strength(article.get("strength"))
        title_ko = normalize_text(article.get("title_ko")) or "번역 없음"
        title_en = normalize_text(article.get("title_en")) or "제목 없음"
        summary = sanitize_summary(article.get("summary"))
        reason = sanitize_reason(article.get("reason"))
        source = sanitize_source(article.get("source"))
        published_at = format_datetime(article.get("published_at"))
        link = sanitize_link(article.get("link"))

        with st.container(border=True):
            top_left, top_right = st.columns([0.8, 0.2])

            with top_left:
                st.markdown(f"### {idx}. {title_ko}")
                st.caption(title_en)

            with top_right:
                st.metric("strength", strength)

            meta_col1, meta_col2, meta_col3 = st.columns([0.22, 0.18, 0.60])
            with meta_col1:
                st.write(f"**라벨**: {label}")
            with meta_col2:
                st.write(f"**열기**: {fire}")
            with meta_col3:
                st.write(f"**출처/시간**: {source} / {published_at}")

            st.write(f"**Summary**: {summary}")
            st.write(f"**판단 근거**: {reason}")

            if link:
                st.link_button("원문 링크 열기", link, use_container_width=False)
            else:
                st.caption("원문 링크 없음")


# =========================
# 메인
# =========================
def main():
    st.title("Iran-US War Tracker")

    history_df = load_history()
    articles = load_articles()
    report = load_report()

    latest_batch, latest_smooth = latest_from_report_or_df(report, history_df)
    _, _, latest_ts = latest_scores_from_df(history_df)

    top1, top2, top3 = st.columns(3)
    with top1:
        st.metric("즉각 위험도", format_score(latest_batch))
    with top2:
        st.metric("추세 위험도", format_score(latest_smooth))
    with top3:
        st.metric("마지막 업데이트", latest_ts)

    with st.expander("데이터 상태", expanded=False):
        st.write(f"- daily_signal.csv: {'존재' if CSV_PATH.exists() else '없음'}")
        st.write(f"- classified_articles.json: {'존재' if ARTICLES_PATH.exists() else '없음'}")
        st.write(f"- latest_report.json: {'존재' if REPORT_PATH.exists() else '없음'}")
        st.write(f"- title_ko_cache.json: {'존재' if TITLE_CACHE_PATH.exists() else '없음'}")
        st.write(
            "- 번역 제목은 표시 전에 sanitize 처리되며, 깨진 값/영문 그대로/HTML/코드 조각은 모두 '번역 없음'으로 치환됩니다."
        )

    st.divider()

    chart_df = prepare_chart_df(history_df)
    render_score_chart(chart_df)

    st.divider()

    control_left, control_right = st.columns([0.3, 0.7])
    with control_left:
        article_limit = st.selectbox(
            "표시 기사 수",
            options=[10, 20, 30, 50],
            index=1,
        )
    with control_right:
        st.caption(
            "카드는 HTML 직접 렌더링 없이 Streamlit 기본 컴포넌트만 사용합니다. "
            "그래서 카드 내부에 HTML 코드가 노출되던 문제를 방지합니다."
        )

    render_news_cards(articles, limit=article_limit)


if __name__ == "__main__":
    main()