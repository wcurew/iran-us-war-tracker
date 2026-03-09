import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------
# 기본 설정
# ---------------------------------------------------

st.set_page_config(
    page_title="Iran-US War Tracker",
    page_icon="🔥",
    layout="centered",
)

LATEST_REPORT_PATH = Path("output/latest_report.json")
DAILY_CSV_PATH = Path("data/daily_signal.csv")


# ---------------------------------------------------
# 데이터 로드
# ---------------------------------------------------

@st.cache_data(ttl=300)
def load_report():
    if not LATEST_REPORT_PATH.exists():
        return None

    with open(LATEST_REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_history():
    if not DAILY_CSV_PATH.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(DAILY_CSV_PATH)
    except Exception:
        try:
            df = pd.read_csv(DAILY_CSV_PATH, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()

    if "window_end_utc" in df.columns:
        try:
            df["window_end_utc"] = pd.to_datetime(df["window_end_utc"], errors="coerce")
            df = df.sort_values("window_end_utc")
        except Exception:
            pass

    return df


report = load_report()
history_df = load_history()


# ---------------------------------------------------
# 스타일
# ---------------------------------------------------

st.markdown(
    """
    <style>
    .big-score {
        font-size: 3rem;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }
    .muted {
        color: #888;
        font-size: 0.95rem;
    }
    .section-card {
        padding: 0.8rem 0.2rem 0.2rem 0.2rem;
    }
    .signal-box {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 14px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------
# 빈 상태
# ---------------------------------------------------

if report is None:
    st.title("Iran-US War Tracker")
    st.warning("아직 latest_report.json 파일이 없습니다.")
    st.stop()


# ---------------------------------------------------
# 헤더
# ---------------------------------------------------

st.title("Iran-US War Tracker")
st.caption("6시간마다 자동 갱신")

st.markdown(
    f"""
    <div class="big-score">{report.get('war_smoothed_score', 0):.1f}</div>
    <div class="muted">상태 기반 전쟁 확률</div>
    """,
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)
with c1:
    st.metric(
        "실시간 신호",
        f"{report.get('war_batch_score', 0):.1f}",
    )
with c2:
    st.metric(
        "24시간 추세",
        report.get("trend_label_ko", "-"),
        f"{report.get('trend_delta', 0):+.1f}",
    )

st.info(f"경보: {report.get('alert_message', '-')}")


# ---------------------------------------------------
# 해석
# ---------------------------------------------------

st.subheader("한줄 해석")
st.write(report.get("summary_ko", "-"))


# ---------------------------------------------------
# 실행 구간
# ---------------------------------------------------

with st.expander("실행 정보", expanded=False):
    st.write(f"업데이트 시각: {report.get('generated_at_utc', '-')}")
    st.write(
        f"구간: {report.get('window_start_utc', '-')}"
        f"  ~  {report.get('window_end_utc', '-')}"
    )


# ---------------------------------------------------
# 투자 시그널
# ---------------------------------------------------

st.subheader("투자 시그널")

signals = [
    ("유가", report.get("oil_signal", "-")),
    ("방산", report.get("defense_signal", "-")),
    ("항공", report.get("airline_signal", "-")),
    ("증시", report.get("equity_signal", "-")),
    ("금/달러", report.get("gold_dollar_signal", "-")),
]

for name, value in signals:
    st.markdown(
        f"""
        <div class="signal-box">
            <div><strong>{name}</strong></div>
            <div>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------
# 기사 분포
# ---------------------------------------------------

st.subheader("이번 구간 기사")

r1, r2, r3 = st.columns(3)
r1.metric("협상", report.get("negotiation_count", 0))
r2.metric("휴전", report.get("ceasefire_count", 0))
r3.metric("완화", report.get("deescalation_count", 0))

r4, r5, r6 = st.columns(3)
r4.metric("긴장고조", report.get("escalation_count", 0))
r5.metric("대리세력", report.get("proxy_escalation_count", 0))
r6.metric("직접공격", report.get("strike_count", 0))


# ---------------------------------------------------
# 그래프
# ---------------------------------------------------

if not history_df.empty:
    st.subheader("전쟁 확률 추세")

    chart_df = history_df.copy()

    cols = []
    if "window_end_utc" in chart_df.columns:
        chart_df["window_end_utc"] = pd.to_datetime(chart_df["window_end_utc"], errors="coerce")
        chart_df = chart_df.sort_values("window_end_utc").set_index("window_end_utc")

    if "war_smoothed_score" in chart_df.columns:
        cols.append("war_smoothed_score")
    if "war_batch_score" in chart_df.columns:
        cols.append("war_batch_score")

    if cols:
        st.line_chart(chart_df[cols], height=260, use_container_width=True)

    with st.expander("최근 기록 보기", expanded=False):
        show_cols = [
            c for c in [
                "war_batch_score",
                "war_smoothed_score",
                "trend_label_ko",
                "alert_message",
                "negotiation_count",
                "ceasefire_count",
                "proxy_escalation_count",
                "strike_count",
            ]
            if c in chart_df.columns
        ]
        if show_cols:
            st.dataframe(
                chart_df[show_cols].tail(15),
                use_container_width=True,
            )


# ---------------------------------------------------
# 기사 표시 함수
# ---------------------------------------------------

def show_articles(title: str, rows: list):
    if not rows:
        return

    with st.expander(title, expanded=False):
        for row in rows:
            st.markdown(f"**{row.get('title', '')}**")
            st.caption(
                f"{row.get('source', '')} | "
                f"strength={row.get('strength', '')} | "
                f"weight={row.get('source_weight', '')}"
            )
            st.write(row.get("reason", ""))
            if row.get("link"):
                st.markdown(f"[기사 링크]({row['link']})")
            st.divider()


# ---------------------------------------------------
# Top 기사
# ---------------------------------------------------

show_articles("Top Negotiation", report.get("top_negotiation_articles", []))
show_articles("Top Ceasefire", report.get("top_ceasefire_articles", []))
show_articles("Top Proxy Escalation", report.get("top_proxy_articles", []))
show_articles("Top Strike", report.get("top_strike_articles", []))


# ---------------------------------------------------
# 하단
# ---------------------------------------------------

st.caption("모바일 보기 최적화 버전")