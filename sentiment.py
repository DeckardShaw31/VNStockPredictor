"""
sentiment.py — LLM-powered news sentiment for Vietnam stocks.

Architecture:
  1. Fetch recent Vietnamese financial news headlines via free RSS/web sources
  2. Score each headline with Claude (claude-haiku-4-5 for speed/cost)
  3. Aggregate into a daily sentiment score per symbol
  4. Cache results to avoid re-scoring the same headlines

Why LLM beats simple keyword matching:
  - Handles negations ("không tăng" = "not rising" → bearish)
  - Understands context ("lãi suất tăng" = "interest rates rise" → bearish for growth stocks)
  - Can read mixed Vietnamese/English financial text
  - Understands sector-specific nuance

Sentiment score: float in [-1, +1]
  -1 = very bearish, 0 = neutral, +1 = very bullish

Data sources (all free, no API key needed):
  - VnExpress Finance RSS: https://vnexpress.net/rss/kinh-doanh.rss
  - CafeF RSS: various RSS feeds
  - VietStock news: web scraping
  - Google News search for ticker symbols
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests
import pandas as pd
import numpy as np

logger = logging.getLogger("sentiment")

SENTIMENT_CACHE_DIR = Path("data/sentiment_cache")
SENTIMENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Claude API endpoint — key is handled by the environment/proxy
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL   = "claude-haiku-4-5-20251001"   # fast + cheap for batch scoring

# News sources (RSS feeds — free, no auth)
NEWS_SOURCES = [
    "https://vnexpress.net/rss/kinh-doanh.rss",
    "https://vnexpress.net/rss/chung-khoan.rss",
    "https://cafef.vn/thi-truong-chung-khoan.rss",
]

# Map stock symbols to company names / keywords for relevance filtering
SYMBOL_KEYWORDS = {
    "VNM":  ["vinamilk", "vnm", "sữa"],
    "VIC":  ["vingroup", "vic", "vinhomes", "vinpearl"],
    "HPG":  ["hòa phát", "hpg", "thép"],
    "VCB":  ["vietcombank", "vcb"],
    "FPT":  ["fpt", "công nghệ thông tin"],
    "MWG":  ["thế giới di động", "mwg", "điện máy xanh"],
    "MSN":  ["masan", "msn"],
    "TCB":  ["techcombank", "tcb"],
    "SSI":  ["ssi", "chứng khoán ssi"],
    "GAS":  ["pv gas", "gas", "khí"],
    "PLX":  ["petrolimex", "plx", "xăng dầu"],
    "SAB":  ["sabeco", "sab", "bia"],
    "VHM":  ["vinhomes", "vhm", "bất động sản"],
    "HPA":  ["hpa"],
    "SHS":  ["shs", "chứng khoán shs"],
    "VGI":  ["viettel global", "vgi"],
    "GEX":  ["gelex", "gex"],
    "CTG":  ["vietinbank", "ctg"],
    "BID":  ["bidv", "bid"],
}


def _headline_cache_key(headline: str) -> str:
    return hashlib.md5(headline.encode()).hexdigest()[:12]


def _load_sentiment_cache() -> dict:
    p = SENTIMENT_CACHE_DIR / "scored_headlines.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_sentiment_cache(cache: dict):
    p = SENTIMENT_CACHE_DIR / "scored_headlines.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def fetch_news_headlines(max_per_source: int = 30) -> List[dict]:
    """
    Fetch recent headlines from Vietnamese financial RSS feeds.
    Returns list of {title, link, published, source}
    """
    headlines = []
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; VNStockAI/1.0)"
    }

    for url in NEWS_SOURCES:
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()

            # Simple RSS XML parsing without feedparser dependency
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.content)
            ns = {"": ""}

            items = root.findall(".//item")
            for item in items[:max_per_source]:
                title = item.findtext("title", "")
                link  = item.findtext("link", "")
                pub   = item.findtext("pubDate", "")
                if title:
                    headlines.append({
                        "title":     title.strip(),
                        "link":      link.strip(),
                        "published": pub.strip(),
                        "source":    url.split("/")[2],
                    })
        except Exception as e:
            logger.warning(f"[news] Failed to fetch {url}: {e}")

    logger.info(f"[news] Fetched {len(headlines)} headlines from {len(NEWS_SOURCES)} sources")
    return headlines


def score_headlines_with_llm(
    headlines: List[str],
    api_key: Optional[str] = None,
    batch_size: int = 20,
) -> Dict[str, float]:
    """
    Score a list of headlines using Claude.
    Returns {headline: sentiment_score} where score in [-1, +1].

    Batches headlines to minimise API calls.
    """
    import os
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        logger.warning("[sentiment] No ANTHROPIC_API_KEY found — skipping LLM scoring")
        return {}

    cache = _load_sentiment_cache()
    results = {}
    to_score = []

    # Check cache first
    for h in headlines:
        ck = _headline_cache_key(h)
        if ck in cache:
            results[h] = cache[ck]
        else:
            to_score.append(h)

    if not to_score:
        return results

    # Score in batches
    for i in range(0, len(to_score), batch_size):
        batch = to_score[i: i + batch_size]
        numbered = "\n".join(f"{j+1}. {h}" for j, h in enumerate(batch))

        prompt = f"""You are a financial sentiment analyst specialising in the Vietnam stock market.

Score each headline below for its sentiment impact on Vietnam stocks.
Respond ONLY with a JSON array of numbers, one per headline, in range [-1.0, +1.0]:
  -1.0 = very bearish (strong negative market impact)
   0.0 = neutral
  +1.0 = very bullish (strong positive market impact)

Consider: earnings beats/misses, regulatory news, macro indicators (GDP, inflation, interest rates),
foreign investment flows, sector-specific news, management changes, legal issues.

Headlines:
{numbered}

Respond with ONLY a JSON array like: [-0.3, 0.8, 0.0, ...]
No explanation, just the array."""

        try:
            resp = requests.post(
                CLAUDE_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": CLAUDE_MODEL,
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            resp.raise_for_status()
            text = resp.json()["content"][0]["text"].strip()

            # Parse JSON array
            scores = json.loads(text)
            if len(scores) != len(batch):
                logger.warning(f"[sentiment] Got {len(scores)} scores for {len(batch)} headlines")
                scores = scores[:len(batch)] + [0.0] * (len(batch) - len(scores))

            for headline, score in zip(batch, scores):
                score = float(np.clip(score, -1.0, 1.0))
                results[headline] = score
                cache[_headline_cache_key(headline)] = score

            time.sleep(0.3)  # gentle rate limiting

        except Exception as e:
            logger.warning(f"[sentiment] LLM scoring failed for batch {i//batch_size}: {e}")
            for h in batch:
                results[h] = 0.0

    _save_sentiment_cache(cache)
    return results


def build_sentiment_features(
    symbols: List[str],
    daily_index: pd.DatetimeIndex,
    api_key: Optional[str] = None,
    lookback_days: int = 7,
) -> Dict[str, pd.DataFrame]:
    """
    Build a daily sentiment score DataFrame for each symbol.

    Returns {symbol: DataFrame} where each DataFrame has columns:
        sentiment_mean  – mean score of relevant headlines today
        sentiment_max   – most bullish headline score
        sentiment_min   – most bearish headline score
        sentiment_count – number of relevant headlines
        sentiment_7d    – 7-day rolling mean sentiment
    """
    headlines = fetch_news_headlines(max_per_source=40)
    if not headlines:
        logger.warning("[sentiment] No headlines fetched — sentiment features will be zero")

    texts = [h["title"] for h in headlines]
    scores_map = score_headlines_with_llm(texts, api_key=api_key) if texts else {}

    results = {}
    today = pd.Timestamp.today().normalize()

    for sym in symbols:
        keywords = SYMBOL_KEYWORDS.get(sym, [sym.lower()])

        # Filter headlines relevant to this symbol
        relevant_scores = []
        for h in headlines:
            title_lower = h["title"].lower()
            if any(kw.lower() in title_lower for kw in keywords):
                score = scores_map.get(h["title"], 0.0)
                relevant_scores.append(score)

        # Build daily series (all today since we don't have per-day news history)
        feat = pd.DataFrame(index=daily_index)
        feat["sentiment_mean"]  = 0.0
        feat["sentiment_max"]   = 0.0
        feat["sentiment_min"]   = 0.0
        feat["sentiment_count"] = 0.0

        if relevant_scores and today in feat.index:
            feat.loc[today, "sentiment_mean"]  = float(np.mean(relevant_scores))
            feat.loc[today, "sentiment_max"]   = float(np.max(relevant_scores))
            feat.loc[today, "sentiment_min"]   = float(np.min(relevant_scores))
            feat.loc[today, "sentiment_count"] = len(relevant_scores)

        # Rolling mean sentiment
        feat["sentiment_7d"] = feat["sentiment_mean"].rolling(
            min(7, len(feat)), min_periods=1
        ).mean()

        results[sym] = feat

    return results


def load_or_build_sentiment(
    symbols: List[str],
    daily_index: pd.DatetimeIndex,
    api_key: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load from cache if fresh (< 6h), otherwise rebuild.
    """
    cache_file = SENTIMENT_CACHE_DIR / "daily_sentiment.pkl"
    import pickle

    if cache_file.exists():
        age_h = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() / 3600
        if age_h < 6:
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            logger.info("[sentiment] Loaded from cache")
            # Re-align to current daily_index
            return {
                sym: df.reindex(daily_index)
                for sym, df in cached.items()
                if sym in symbols
            }

    logger.info("[sentiment] Building fresh sentiment features...")
    result = build_sentiment_features(symbols, daily_index, api_key)

    import pickle
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)

    return result
