import csv
import json
import os
import re
import time
import hashlib
import html
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import requests
from openai import OpenAI


# =========================================================
# Config
# =========================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini").strip()

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
RAW_DIR = DATA_DIR / "raw"

CLASSIFIED_JSON_PATH = DATA_DIR / "classified_articles.json"
LATEST_REPORT_PATH = OUTPUT_DIR / "latest_report.json"
RUN_LOG_PATH = OUTPUT_DIR / "last_run_log.json"
STATE_PATH = OUTPUT_DIR / "state.json"
DAILY_CSV_PATH = DATA_DIR / "daily_signal.csv"

REQUEST_TIMEOUT = 25
MAX_ARTICLES_PER_FEED = 30
MAX_CLASSIFY_ARTICLES = 80
CLASSIFY_RETRIES = 3
SLEEP_BETWEEN_CALLS = 0.15

DEFAULT_LOOKBACK_HOURS = 6
STATE_HISTORY_LIMIT = 50
SEEN_ARTICLE_IDS_LIMIT = 500

# 기사 분류 점수(전쟁 확률용 raw score)
BATCH_WEIGHTS = {
    "negotiation": -2.0,
    "ceasefire": -4.0,
    "deescalation_signal": -1.0,
    "escalation": 2.0,
    "strike_or_retaliation": 5.0,
    "irrelevant": 0.0,
}

# raw score를 0~100 배치 점수로 정규화할 때 사용하는 범위
RAW_SCORE_MIN = -10.0
RAW_SCORE_MAX = 10.0

RSS_QUERIES = [
    '("Iran" OR "Iranian") ("United States" OR U.S. OR US OR American) ceasefire',
    '("Iran" OR "Iranian") ("United States" OR U.S. OR US OR American) negotiation',
    '("Iran" OR "Iranian") ("United States" OR U.S. OR US OR American) talks',
    '("Iran" OR "Iranian") ("United States" OR U.S. OR US OR American) diplomacy',
    '("Iran" OR "Iranian") ("United States" OR U.S. OR US OR American) retaliation',
    '("Iran" OR "Iranian") ("United States" OR U.S. OR US OR American) strike',
    '("Iran" OR "Iranian") ("United States" OR U.S. OR US OR American) missile',
    '("Iran" OR "Iranian") ("United States" OR U.S. OR US OR American) escalation',
    '("Iran" OR "Iranian") ("United States" OR U.S. OR US OR American) de-escalation',
    '("Iran" OR "Iranian") ("United States" OR U.S. OR US OR American) warning',
]

CLASSIFIER_PROMPT = """
You classify news articles about Iran-US conflict developments.

Choose exactly one label:
- negotiation
- ceasefire
- deescalation_signal
- escalation
- strike_or_retaliation
- irrelevant

Definitions:
- negotiation:
  active diplomatic talks, mediation, backchannel contacts, envoys meeting,
  proposal exchanges, talks resuming, negotiation framework discussion
- ceasefire:
  explicit ceasefire, truce, pause in fighting, halt in attacks, suspension
  of hostilities, agreed military pause
- deescalation_signal:
  softer calming signals without a clear ceasefire, restraint, willingness
  to avoid wider war, indirect contact, delaying retaliation, reduction in tone
- escalation:
  threats, warnings, breakdown of diplomacy, rejection of proposals,
  rising tensions, mobilization, hostile rhetoric
- strike_or_retaliation:
  actual strike, bombing, missile launch, attack, retaliation, military action
- irrelevant:
  not materially about Iran-US conflict diplomacy or military developments

Extra rules:
- ceasefire is only for explicit pause/truce/halt in hostilities, not vague hope
- articles mainly about oil prices, domestic politics, or broad commentary are irrelevant
  unless they directly discuss Iran-US diplomacy or military action
- proxy activity should be classified cautiously unless clearly framed as Iran-US escalation

Return ONLY valid JSON with this exact schema:
{
  "label": "one of the labels above",
  "strength": 0.0,
  "reason": "brief explanation under 25 words",
  "event_key": "short normalized event tag"
}

Rules:
- strength must be a number from 0.0 to 1.0
- event_key should group duplicate stories about the same underlying event
- never return markdown
- if uncertain, choose the closest label and lower the strength
""".strip()


# =========================================================
# Utilities
# =========================================================

def ensure_dirs() -> None:
    for p in [DATA_DIR, OUTPUT_DIR, RAW_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().replace(microsecond=0).isoformat()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def strip_html(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return normalize_space(text)


def normalize_title(text: str) -> str:
    text = strip_html(text).lower()
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return normalize_space(text)


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        if x < 0:
            return 0.0
        if x > 1:
            return 1.0
        return x
    except Exception:
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    m = n // 2
    if n % 2 == 1:
        return s[m]
    return round((s[m - 1] + s[m]) / 2.0, 4)


def parse_iso_datetime(text: str) -> Optional[datetime]:
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def parse_pubdate(text: str) -> Optional[str]:
    if not text:
        return None
    try:
        dt = parsedate_to_datetime(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return None


def save_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


# =========================================================
# State / Time Window
# =========================================================

def load_state() -> Dict[str, Any]:
    state = load_json(STATE_PATH, default={})
    if not isinstance(state, dict):
        return {}
    return state


def save_state(state: Dict[str, Any]) -> None:
    save_json(STATE_PATH, state)


def get_time_window() -> tuple[datetime, datetime]:
    now_utc = utc_now()
    state = load_state()
    last_run = parse_iso_datetime(state.get("last_run_time_utc", ""))

    if last_run is None:
        last_run = now_utc - timedelta(hours=DEFAULT_LOOKBACK_HOURS)

    # 미래시간 방지
    if last_run > now_utc:
        last_run = now_utc - timedelta(hours=DEFAULT_LOOKBACK_HOURS)

    return last_run, now_utc


def get_seen_article_ids_from_state() -> set[str]:
    state = load_state()
    raw_ids = state.get("seen_article_ids", [])
    if not isinstance(raw_ids, list):
        return set()
    return {str(x) for x in raw_ids if x}


def get_history_from_state() -> List[Dict[str, Any]]:
    state = load_state()
    history = state.get("history", [])
    if not isinstance(history, list):
        return []
    output = []
    for row in history:
        if isinstance(row, dict):
            output.append(row)
    return output


# =========================================================
# RSS Fetch
# =========================================================

def google_news_rss_url(query: str) -> str:
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


def fetch_rss(url: str, timeout: int = REQUEST_TIMEOUT) -> List[Dict[str, Any]]:
    resp = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 (compatible; IranUSTracker/2.0)"},
    )
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    items = []
    channel = root.find("channel")
    if channel is None:
        return items

    for item in channel.findall("item"):
        title = item.findtext("title", default="")
        link = item.findtext("link", default="")
        pub_date = item.findtext("pubDate", default="")
        description = item.findtext("description", default="")
        source_el = item.find("source")
        source = source_el.text.strip() if source_el is not None and source_el.text else ""

        items.append({
            "title": strip_html(title),
            "summary": strip_html(description),
            "link": link.strip(),
            "source": source,
            "published_at": parse_pubdate(pub_date),
            "raw_pub_date": pub_date,
            "feed_url": url,
        })
    return items


def fetch_articles() -> List[Dict[str, Any]]:
    all_articles: List[Dict[str, Any]] = []

    for query in RSS_QUERIES:
        url = google_news_rss_url(query)
        try:
            batch = fetch_rss(url)
        except Exception as e:
            print(f"[WARN] RSS fetch failed: {query} / {e}")
            batch = []

        if batch:
            all_articles.extend(batch[:MAX_ARTICLES_PER_FEED])

        time.sleep(0.1)

    return all_articles


# =========================================================
# Filtering / Dedupe
# =========================================================

def filter_articles_by_time_window(
    articles: List[Dict[str, Any]],
    start_time: datetime,
    end_time: datetime,
) -> List[Dict[str, Any]]:
    output = []

    for article in articles:
        published_at = parse_iso_datetime(article.get("published_at", ""))
        if published_at is None:
            continue

        if start_time < published_at <= end_time:
            output.append(article)

    return output


def is_relevant_keyword(title: str, summary: str) -> bool:
    text = f"{title} {summary}".lower()

    iran_terms = [
        "iran", "iranian", "tehran", "iran's", "iranian-backed"
    ]
    us_terms = [
        "u.s.", " us ", " us-", "united states", "american", "america", "washington", "pentagon"
    ]
    conflict_terms = [
        "ceasefire", "truce", "negotiation", "talks", "diplomacy", "diplomatic",
        "retaliation", "retaliate", "strike", "attack", "missile", "military",
        "de-escalation", "escalation", "warning", "hostilities", "bombing",
        "airstrike", "envoy", "mediator", "proposal", "conflict", "threat"
    ]

    has_iran = any(t in text for t in iran_terms)
    has_us = any(t in text for t in us_terms)
    has_conflict = any(t in text for t in conflict_terms)

    return has_iran and has_us and has_conflict


def dedupe_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output = []

    for a in articles:
        title_key = normalize_title(a.get("title", ""))
        link_key = (a.get("link") or "").strip().lower()
        source_key = normalize_title(a.get("source", ""))

        if link_key:
            key = f"link::{link_key}"
        else:
            key = f"title_source::{title_key}::{source_key}"

        if key in seen:
            continue

        seen.add(key)
        a["article_id"] = sha1_text(key)
        output.append(a)

    return output


def remove_seen_articles(articles: List[Dict[str, Any]], seen_article_ids: set[str]) -> List[Dict[str, Any]]:
    output = []
    for a in articles:
        article_id = a.get("article_id", "")
        if article_id and article_id in seen_article_ids:
            continue
        output.append(a)
    return output


# =========================================================
# OpenAI Classification
# =========================================================

def build_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=OPENAI_API_KEY)


def extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model output")

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Could not parse JSON from: {text[:300]}")


def classify_article(client: OpenAI, title: str, summary: str) -> Dict[str, Any]:
    user_input = f"Title: {title}\nSummary: {summary}\nClassify this article."

    last_err = None
    for attempt in range(1, CLASSIFY_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=[
                    {"role": "system", "content": CLASSIFIER_PROMPT},
                    {"role": "user", "content": user_input},
                ],
            )

            text = getattr(resp, "output_text", "") or ""
            data = extract_json_object(text)

            label = str(data.get("label", "irrelevant")).strip()
            if label not in BATCH_WEIGHTS:
                label = "irrelevant"

            strength = safe_float(data.get("strength", 0.0), default=0.0)
            reason = normalize_space(str(data.get("reason", "")))[:200]
            event_key = normalize_title(str(data.get("event_key", "")))[:80]

            if not event_key:
                event_key = normalize_title(title)[:80]

            return {
                "label": label,
                "strength": strength,
                "reason": reason,
                "event_key": event_key,
            }

        except Exception as e:
            last_err = e
            time.sleep(min(2 * attempt, 5))

    return {
        "label": "irrelevant",
        "strength": 0.0,
        "reason": f"classification_failed: {last_err}",
        "event_key": normalize_title(title)[:80],
    }


def classify_articles(client: OpenAI, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    target_count = min(len(articles), MAX_CLASSIFY_ARTICLES)

    for idx, article in enumerate(articles[:MAX_CLASSIFY_ARTICLES], start=1):
        cls = classify_article(
            client=client,
            title=article.get("title", ""),
            summary=article.get("summary", ""),
        )
        output.append({
            **article,
            "classification": cls,
        })
        print(f"[INFO] Classified {idx}/{target_count}: {cls['label']}")
        time.sleep(SLEEP_BETWEEN_CALLS)

    return output


# =========================================================
# Score / Trend / Market Signal
# =========================================================

def normalize_raw_to_100(raw_score: float) -> float:
    raw_score = clamp(raw_score, RAW_SCORE_MIN, RAW_SCORE_MAX)
    normalized = ((raw_score - RAW_SCORE_MIN) / (RAW_SCORE_MAX - RAW_SCORE_MIN)) * 100.0
    return round(clamp(normalized, 0.0, 100.0), 2)


def build_event_stats(classified_articles: List[Dict[str, Any]]) -> Dict[str, int]:
    unique_by_label: Dict[str, set[str]] = {}

    for article in classified_articles:
        cls = article["classification"]
        label = cls["label"]
        if label == "irrelevant":
            continue

        event_key = cls.get("event_key", "") or normalize_title(article.get("title", ""))
        unique_by_label.setdefault(label, set()).add(event_key)

    return {
        "negotiation_events": len(unique_by_label.get("negotiation", set())),
        "ceasefire_events": len(unique_by_label.get("ceasefire", set())),
        "deescalation_events": len(unique_by_label.get("deescalation_signal", set())),
        "escalation_events": len(unique_by_label.get("escalation", set())),
        "strike_events": len(unique_by_label.get("strike_or_retaliation", set())),
    }


def compute_batch_metrics(classified_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter()
    strengths: List[float] = []
    positive_strengths: List[float] = []
    negative_strengths: List[float] = []
    batch_raw_score = 0.0

    positive_labels = {"negotiation", "ceasefire", "deescalation_signal"}
    negative_labels = {"escalation", "strike_or_retaliation"}

    relevant_articles = []

    for article in classified_articles:
        cls = article["classification"]
        label = cls["label"]
        strength = safe_float(cls.get("strength", 0.0))

        counts[label] += 1
        batch_raw_score += BATCH_WEIGHTS[label] * strength

        if label != "irrelevant":
            relevant_articles.append(article)
            strengths.append(strength)

        if label in positive_labels:
            positive_strengths.append(strength)
        elif label in negative_labels:
            negative_strengths.append(strength)

    relevant_total = len(relevant_articles)
    positive_count = sum(counts[x] for x in positive_labels)
    negative_count = sum(counts[x] for x in negative_labels)

    negotiation_ratio = round(positive_count / relevant_total, 4) if relevant_total else 0.0
    war_ratio = round(negative_count / relevant_total, 4) if relevant_total else 0.0

    event_stats = build_event_stats(classified_articles)
    war_batch_score = normalize_raw_to_100(batch_raw_score)

    return {
        "total_articles_classified": len(classified_articles),
        "relevant_articles": relevant_total,
        "negotiation_count": counts["negotiation"],
        "ceasefire_count": counts["ceasefire"],
        "deescalation_count": counts["deescalation_signal"],
        "escalation_count": counts["escalation"],
        "strike_count": counts["strike_or_retaliation"],
        "irrelevant_count": counts["irrelevant"],
        "negotiation_ratio": negotiation_ratio,
        "war_ratio": war_ratio,
        "avg_strength": round(mean(strengths), 4),
        "median_strength": median(strengths),
        "avg_positive_strength": round(mean(positive_strengths), 4),
        "avg_negative_strength": round(mean(negative_strengths), 4),
        "batch_raw_score": round(batch_raw_score, 4),
        "war_batch_score": war_batch_score,
        **event_stats,
    }


def compute_smoothed_score(current_batch_score: float, history: List[Dict[str, Any]]) -> float:
    prev_scores = []
    for row in reversed(history):
        val = row.get("war_batch_score")
        if isinstance(val, (int, float)):
            prev_scores.append(float(val))
        if len(prev_scores) >= 2:
            break

    prev1 = prev_scores[0] if len(prev_scores) >= 1 else current_batch_score
    prev2 = prev_scores[1] if len(prev_scores) >= 2 else prev1

    smoothed = 0.6 * current_batch_score + 0.3 * prev1 + 0.1 * prev2
    return round(clamp(smoothed, 0.0, 100.0), 2)


def compute_trend(history: List[Dict[str, Any]], current_batch_score: float) -> tuple[float, str]:
    batch_scores = []
    for row in history:
        val = row.get("war_batch_score")
        if isinstance(val, (int, float)):
            batch_scores.append(float(val))

    batch_scores.append(float(current_batch_score))

    recent = batch_scores[-4:]
    previous = batch_scores[-8:-4]

    recent_avg = mean(recent)
    previous_avg = mean(previous) if previous else recent_avg
    trend_delta = round(recent_avg - previous_avg, 2)

    if trend_delta >= 10:
        trend_label = "strong_up"
    elif trend_delta >= 4:
        trend_label = "up"
    elif trend_delta <= -10:
        trend_label = "strong_down"
    elif trend_delta <= -4:
        trend_label = "down"
    else:
        trend_label = "flat"

    return trend_delta, trend_label


def trend_label_ko(trend_label: str) -> str:
    mapping = {
        "strong_up": "가파른 상승",
        "up": "상승",
        "flat": "보합",
        "down": "하락",
        "strong_down": "가파른 하락",
    }
    return mapping.get(trend_label, "보합")


def build_market_signal(war_smoothed_score: float) -> Dict[str, str]:
    if war_smoothed_score >= 75:
        return {
            "oil_signal": "급등 리스크",
            "defense_signal": "매우 강세",
            "airline_signal": "악재",
            "equity_signal": "강한 리스크오프",
            "gold_dollar_signal": "강한 선호",
        }
    if war_smoothed_score >= 50:
        return {
            "oil_signal": "강세",
            "defense_signal": "강세",
            "airline_signal": "부담",
            "equity_signal": "리스크오프",
            "gold_dollar_signal": "선호",
        }
    if war_smoothed_score >= 25:
        return {
            "oil_signal": "약한 강세",
            "defense_signal": "약한 강세",
            "airline_signal": "중립",
            "equity_signal": "중립",
            "gold_dollar_signal": "중립~선호",
        }
    return {
        "oil_signal": "중립~약세",
        "defense_signal": "중립",
        "airline_signal": "우호적",
        "equity_signal": "리스크온",
        "gold_dollar_signal": "중립",
    }


def build_summary_ko(report: Dict[str, Any]) -> str:
    score = report["war_smoothed_score"]
    trend = report["trend_label"]
    strike_count = report["strike_count"]
    negotiation = report["negotiation_count"]
    ceasefire = report["ceasefire_count"]

    if report["relevant_articles"] == 0:
        return "이번 실행 구간에는 전쟁 확률을 의미 있게 바꿀 만한 새 관련 기사가 거의 없었다."

    if score >= 75:
        if strike_count >= 1:
            return "실제 공격·보복 보도가 강하게 반영돼 고위험 구간이다. 단기 확전 리스크를 강하게 경계해야 한다."
        return "긴장 고조 흐름이 매우 강하다. 시장은 방어적 포지션과 에너지 강세 쪽 해석이 유효하다."

    if score >= 50:
        if trend in {"strong_up", "up"}:
            return "전쟁 위험이 높아지는 방향이다. 공격·경고 기사 비중이 협상 신호보다 우세하다."
        return "아직 긴장 상태가 우세하다. 다만 즉각적 확전 여부는 추가 기사 확인이 필요하다."

    if score >= 25:
        if negotiation + ceasefire >= 2:
            return "협상·완화 신호가 일부 보이지만 아직 중립권이다. 확전과 완화 신호가 혼재한다."
        return "상태 점수는 중립권이다. 이번 구간만으로 방향성을 단정하긴 어렵다."

    return "완화 신호가 우세하다. 협상 또는 휴전 관련 보도가 전쟁 위험을 낮추는 방향으로 작용하고 있다."


def top_articles_by_label(classified_articles: List[Dict[str, Any]], label: str, n: int = 5) -> List[Dict[str, Any]]:
    rows = [a for a in classified_articles if a["classification"]["label"] == label]
    rows.sort(key=lambda x: x["classification"]["strength"], reverse=True)

    output = []
    for a in rows[:n]:
        output.append({
            "title": a.get("title", ""),
            "source": a.get("source", ""),
            "published_at": a.get("published_at"),
            "link": a.get("link", ""),
            "strength": a["classification"]["strength"],
            "reason": a["classification"]["reason"],
            "event_key": a["classification"]["event_key"],
        })
    return output


# =========================================================
# CSV / Report Helpers
# =========================================================

def append_daily_csv(path: Path, row: Dict[str, Any]) -> None:
    file_exists = path.exists()
    fieldnames = list(row.keys())

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_empty_report(window_start: datetime, window_end: datetime, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    war_batch_score = 50.0
    war_smoothed_score = compute_smoothed_score(war_batch_score, history)
    trend_delta, trend_label = compute_trend(history, war_batch_score)
    market = build_market_signal(war_smoothed_score)

    report = {
        "date_utc": window_end.date().isoformat(),
        "generated_at_utc": utc_now_iso(),
        "window_start_utc": window_start.isoformat(),
        "window_end_utc": window_end.isoformat(),

        "total_articles_classified": 0,
        "relevant_articles": 0,
        "negotiation_count": 0,
        "ceasefire_count": 0,
        "deescalation_count": 0,
        "escalation_count": 0,
        "strike_count": 0,
        "irrelevant_count": 0,

        "negotiation_ratio": 0.0,
        "war_ratio": 0.0,
        "avg_strength": 0.0,
        "median_strength": 0.0,
        "avg_positive_strength": 0.0,
        "avg_negative_strength": 0.0,

        "negotiation_events": 0,
        "ceasefire_events": 0,
        "deescalation_events": 0,
        "escalation_events": 0,
        "strike_events": 0,

        "batch_raw_score": 0.0,
        "war_batch_score": war_batch_score,
        "war_smoothed_score": war_smoothed_score,
        "trend_delta": trend_delta,
        "trend_label": trend_label,
        "trend_label_ko": trend_label_ko(trend_label),

        **market,

        "summary_ko": "이번 실행 구간에는 새로 올라온 관련 기사가 거의 없었다.",

        "top_negotiation_articles": [],
        "top_ceasefire_articles": [],
        "top_escalation_articles": [],
        "top_strike_articles": [],
    }
    return report


def build_report(
    window_start: datetime,
    window_end: datetime,
    classified_articles: List[Dict[str, Any]],
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    metrics = compute_batch_metrics(classified_articles)

    war_batch_score = metrics["war_batch_score"]
    war_smoothed_score = compute_smoothed_score(war_batch_score, history)
    trend_delta, trend_label = compute_trend(history, war_batch_score)
    market = build_market_signal(war_smoothed_score)

    report = {
        "date_utc": window_end.date().isoformat(),
        "generated_at_utc": utc_now_iso(),
        "window_start_utc": window_start.isoformat(),
        "window_end_utc": window_end.isoformat(),

        **metrics,

        "war_smoothed_score": war_smoothed_score,
        "trend_delta": trend_delta,
        "trend_label": trend_label,
        "trend_label_ko": trend_label_ko(trend_label),

        **market,
    }

    report["summary_ko"] = build_summary_ko(report)
    report["top_negotiation_articles"] = top_articles_by_label(classified_articles, "negotiation", n=5)
    report["top_ceasefire_articles"] = top_articles_by_label(classified_articles, "ceasefire", n=5)
    report["top_escalation_articles"] = top_articles_by_label(classified_articles, "escalation", n=5)
    report["top_strike_articles"] = top_articles_by_label(classified_articles, "strike_or_retaliation", n=5)

    return report


# =========================================================
# Main
# =========================================================

def run_tracker() -> Dict[str, Any]:
    ensure_dirs()

    window_start, window_end = get_time_window()
    state = load_state()
    history = get_history_from_state()
    seen_article_ids = get_seen_article_ids_from_state()

    run_log: Dict[str, Any] = {
        "started_at_utc": utc_now_iso(),
        "openai_model": OPENAI_MODEL,
        "window_start_utc": window_start.isoformat(),
        "window_end_utc": window_end.isoformat(),
        "steps": {},
    }

    print("[INFO] Fetching articles...")
    raw_articles = fetch_articles()
    run_log["steps"]["raw_fetched"] = len(raw_articles)

    raw_dump_name = RAW_DIR / f"raw_{window_end.strftime('%Y%m%dT%H%M%SZ')}.json"
    save_json(raw_dump_name, raw_articles)

    print(f"[INFO] Raw fetched: {len(raw_articles)}")
    print(f"[INFO] Time window: {window_start.isoformat()} ~ {window_end.isoformat()}")

    time_filtered = filter_articles_by_time_window(
        raw_articles,
        start_time=window_start,
        end_time=window_end,
    )
    run_log["steps"]["time_filtered"] = len(time_filtered)

    keyword_filtered = [
        a for a in time_filtered
        if is_relevant_keyword(a.get("title", ""), a.get("summary", ""))
    ]
    run_log["steps"]["keyword_filtered"] = len(keyword_filtered)

    deduped = dedupe_articles(keyword_filtered)
    run_log["steps"]["deduped"] = len(deduped)

    unseen = remove_seen_articles(deduped, seen_article_ids)
    run_log["steps"]["after_seen_filter"] = len(unseen)

    print(f"[INFO] After time filter   : {len(time_filtered)}")
    print(f"[INFO] After keyword      : {len(keyword_filtered)}")
    print(f"[INFO] After dedupe       : {len(deduped)}")
    print(f"[INFO] After seen filter  : {len(unseen)}")

    if not unseen:
        report = build_empty_report(window_start, window_end, history)

        save_json(CLASSIFIED_JSON_PATH, [])
        save_json(LATEST_REPORT_PATH, report)
        save_json(RUN_LOG_PATH, {
            **run_log,
            "finished_at_utc": utc_now_iso(),
            "summary": report["summary_ko"],
        })

        csv_row = {k: v for k, v in report.items() if not isinstance(v, list)}
        append_daily_csv(DAILY_CSV_PATH, csv_row)

        new_history = history + [{
            "window_start_utc": window_start.isoformat(),
            "window_end_utc": window_end.isoformat(),
            "war_batch_score": report["war_batch_score"],
            "war_smoothed_score": report["war_smoothed_score"],
            "trend_label": report["trend_label"],
        }]
        new_history = new_history[-STATE_HISTORY_LIMIT:]

        save_state({
            "last_run_time_utc": window_end.isoformat(),
            "history": new_history,
            "seen_article_ids": list(seen_article_ids)[-SEEN_ARTICLE_IDS_LIMIT:],
        })

        return report

    client = build_client()

    print("[INFO] Classifying articles...")
    classified_articles = classify_articles(client, unseen)
    run_log["steps"]["classified"] = len(classified_articles)

    report = build_report(
        window_start=window_start,
        window_end=window_end,
        classified_articles=classified_articles,
        history=history,
    )

    save_json(CLASSIFIED_JSON_PATH, classified_articles)
    save_json(LATEST_REPORT_PATH, report)
    save_json(RUN_LOG_PATH, {
        **run_log,
        "finished_at_utc": utc_now_iso(),
        "summary": report["summary_ko"],
    })

    csv_row = {k: v for k, v in report.items() if not isinstance(v, list)}
    append_daily_csv(DAILY_CSV_PATH, csv_row)

    new_history = history + [{
        "window_start_utc": window_start.isoformat(),
        "window_end_utc": window_end.isoformat(),
        "war_batch_score": report["war_batch_score"],
        "war_smoothed_score": report["war_smoothed_score"],
        "trend_label": report["trend_label"],
    }]
    new_history = new_history[-STATE_HISTORY_LIMIT:]

    new_seen_ids = list(seen_article_ids)
    for article in unseen:
        article_id = article.get("article_id", "")
        if article_id:
            new_seen_ids.append(article_id)
    new_seen_ids = new_seen_ids[-SEEN_ARTICLE_IDS_LIMIT:]

    save_state({
        "last_run_time_utc": window_end.isoformat(),
        "history": new_history,
        "seen_article_ids": new_seen_ids,
    })

    return report


# =========================================================
# Console Output
# =========================================================

def print_article_section(title: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    print(f"\n[{title}]")
    for i, row in enumerate(rows, start=1):
        print(f"{i}. {row.get('title', '')}")
        print(f"   - source   : {row.get('source', '')}")
        print(f"   - time     : {row.get('published_at', '')}")
        print(f"   - strength : {row.get('strength', '')}")
        print(f"   - reason   : {row.get('reason', '')}")
        print(f"   - link     : {row.get('link', '')}")


def print_console_report(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("IRAN-US WAR TRACKER")
    print("=" * 72)
    print(f"실행 구간              : {report['window_start_utc']}  ~  {report['window_end_utc']}")
    print(f"생성 시각 UTC          : {report['generated_at_utc']}")
    print("-" * 72)
    print(f"실시간 전쟁 신호       : {report['war_batch_score']}")
    print(f"상태 기반 전쟁 확률    : {report['war_smoothed_score']}")
    print(f"24시간 추세            : {report['trend_label_ko']} ({report['trend_delta']:+})")
    print("-" * 72)
    print(f"관련 기사 수           : {report['relevant_articles']}")
    print(f"협상 기사              : {report['negotiation_count']}")
    print(f"휴전 기사              : {report['ceasefire_count']}")
    print(f"완화 신호 기사         : {report['deescalation_count']}")
    print(f"긴장 고조 기사         : {report['escalation_count']}")
    print(f"실제 공격/보복 기사    : {report['strike_count']}")
    print("-" * 72)
    print(f"투자 시그널 - 유가     : {report['oil_signal']}")
    print(f"투자 시그널 - 방산     : {report['defense_signal']}")
    print(f"투자 시그널 - 항공     : {report['airline_signal']}")
    print(f"투자 시그널 - 증시     : {report['equity_signal']}")
    print(f"투자 시그널 - 금/달러  : {report['gold_dollar_signal']}")
    print("-" * 72)
    print(f"해석                   : {report['summary_ko']}")
    print("=" * 72)

    print_article_section("TOP NEGOTIATION", report.get("top_negotiation_articles", []))
    print_article_section("TOP CEASEFIRE", report.get("top_ceasefire_articles", []))
    print_article_section("TOP ESCALATION", report.get("top_escalation_articles", []))
    print_article_section("TOP STRIKE", report.get("top_strike_articles", []))


if __name__ == "__main__":
    try:
        report = run_tracker()
        print_console_report(report)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"[ERROR] {e}")
        raise