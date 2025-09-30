# app.py
"""
VN Stock News–Price Analyzer (Vietnam-native prices via vnstock)

- Prices: vnstock (daily OHLCV in VND) using the new Vnstock() interface
- News: Google News RSS (Vietnamese)
- Sentiment: lightweight lexicon (optional transformer if you install it)
"""

import sys
import platform
import datetime as dt
import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import feedparser

# --- Optional transformer sentiment (leave uninstalled on Streamlit Cloud for speed) ---
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
    import torch  # type: ignore
    _HAS_TFM = True
except Exception:
    _HAS_TFM = False

# -------------------------- Page & quick diagnostics -------------------------- #

st.set_page_config(page_title="VN Stock News–Price Analyzer", layout="wide")

with st.expander("Diagnostics (environment)"):
    st.json({
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    })

# -------------------------- Data classes & helpers -------------------------- #

@dataclass
class NewsItem:
    published: dt.datetime
    title: str
    summary: str
    link: str
    source: str
    sentiment: Optional[float] = None  # -1 .. 1

def _coerce_dt(x: str) -> dt.datetime:
    try:
        return dt.datetime(*feedparser._parse_date(x)[:6])  # type: ignore[attr-defined]
    except Exception:
        return dt.datetime.utcnow()

# -------------------------- VN price backend (vnstock) -------------------------- #

def _clean_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize vnstock output to: date, Open, High, Low, Close, Volume (numeric), ascending by date.
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # vnstock usually returns 'time' as the datetime column
    time_col = "time" if "time" in df.columns else ("date" if "date" in df.columns else None)
    if time_col is None:
        raise ValueError("vnstock returned data without a 'time' or 'date' column.")
    df.rename(columns={time_col: "date"}, inplace=True)

    rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close",
                  "adj_close": "Adj Close", "volume": "Volume"}
    for k, v in rename_map.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    if keep_cols:
        df = df.dropna(subset=keep_cols)
    if "Close" in df.columns:
        df = df[df["Close"] > 0]

    df = df.sort_values("date").reset_index(drop=True)
    return df

def load_prices_vietnam(ticker: str, start: dt.date, end: dt.date, source: str = "VCI"
                        ) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    Fetch daily OHLCV for a VN ticker using vnstock (Vnstock().stock(...).quote.history).
    Returns (resolved_symbol, prices_df, debug_log_df).
    """
    attempts: List[Dict[str, Any]] = []
    try:
        # New unified interface
        from vnstock import Vnstock  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import 'vnstock'. Make sure requirements.txt lists 'vnstock' and "
            "that the Python/NumPy/Pandas versions are compatible.\n"
            f"Import error: {type(e).__name__}: {e}"
        )

    symbol = ticker.strip().upper()
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    try:
        stock = Vnstock().stock(symbol=symbol, source=source)
        df = stock.quote.history(start=start_str, end=end_str, interval="1D")
        n = int(len(df)) if isinstance(df, pd.DataFrame) else 0
        attempts.append({"symbol": symbol, "method": f"Vnstock().stock(..., source='{source}').quote.history(1D)", "rows": n})
        if n > 0:
            cleaned = _clean_price_df(df)
            if len(cleaned) > 0:
                return symbol, cleaned, pd.DataFrame(attempts)
    except Exception as e:
        attempts.append({"symbol": symbol, "method": "vnstock history ERROR", "rows": 0, "msg": str(e)})

    debug_df = pd.DataFrame(attempts)
    raise RuntimeError(
        f"No VN data returned for {symbol} (source={source}). "
        "Check the ticker (e.g., HPG, VNM, FPT, MBS) and try a longer date range."
    )

# -------------------------- News & Sentiment -------------------------- #

def google_news_rss(query: str, days: int = 30, lang: str = "vi") -> str:
    q = re.sub(r"\s+", "+", query.strip())
    return f"https://news.google.com/rss/search?q={q}+when:{days}d&hl={lang}&gl=VN&ceid=VN:{lang}"

def fetch_news(query: str, days: int = 30) -> List[NewsItem]:
    url = google_news_rss(query, days=days)
    feed = feedparser.parse(url)
    items: List[NewsItem] = []
    for e in feed.entries:
        title = e.get("title", "").strip()
        summary = re.sub(r"<[^>]+>", " ", e.get("summary", "").strip())
        link = e.get("link", "")
        src = e.get("source", {})
        source_title = src.get("title", src) if isinstance(src, dict) else src
        published = _coerce_dt(e.get("published", ""))
        items.append(NewsItem(published, title, summary, link, source_title or ""))
    # dedup by title
    dedup = {}
    for it in items:
        key = it.title.lower()
        if key not in dedup:
            dedup[key] = it
    return sorted(dedup.values(), key=lambda x: x.published)

# Simple Vietnamese lexicon (fallback if no transformer)
VI_POS = set("tăng|kỷ lục|tích cực|lợi nhuận|vượt|bứt phá|khả quan|mua ròng|đỉnh|bùng nổ|thuận lợi".split("|"))
VI_NEG = set("giảm|tiêu cực|thua lỗ|lỗ|suy giảm|sụt|bán ròng|khó khăn|điều tra|phạt|rủi ro|đình chỉ|suy thoái|nợ xấu".split("|"))

@st.cache_resource(show_spinner=False)
def _load_model():
    if not _HAS_TFM:
        return None, None
    try:
        name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForSequenceClassification.from_pretrained(name)
        mdl.eval()
        return tok, mdl
    except Exception:
        return None, None

def score_sentiment(text: str) -> float:
    text = (text or "").strip()
    if not text:
        return 0.0
    tok, mdl = _load_model()
    if tok is not None and mdl is not None:
        try:
            with torch.no_grad():  # type: ignore[attr-defined]
                inputs = tok(text[:512], return_tensors="pt", truncation=True)
                logits = mdl(**inputs).logits[0]
                probs = torch.softmax(logits, dim=0).cpu().numpy()
                return float(probs[2] - probs[0])  # pos - neg
        except Exception:
            pass
    toks = re.findall(r"\w+", text.lower())
    pos = sum(1 for t in toks if t in VI_POS)
    neg = sum(1 for t in toks if t in VI_NEG)
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / max(1, pos + neg)

def attach_sentiment(items: List[NewsItem]) -> List[NewsItem]:
    for it in items:
        it.sentiment = score_sentiment(f"{it.title}. {it.summary}")
    return items

# ------------------------------ UI ------------------------------ #

st.title("🇻🇳 VN Stock News–Price Analyzer")
st.caption("Dữ liệu giá từ nguồn Việt Nam (thư viện `vnstock`). Tin tức từ Google News (tiếng Việt).")

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    raw_ticker = st.text_input("Mã cổ phiếu (VD: HPG, VNM, MSN, FPT, MBS...)", value="HPG")
with col2:
    lookback_days = st.number_input("Số ngày xem tin", value=30, min_value=7, max_value=365)
with col3:
    date_range = st.date_input(
        "Khoảng thời gian giá",
        value=(dt.date.today() - dt.timedelta(days=365), dt.date.today()),
        min_value=dt.date(2010, 1, 1),
        max_value=dt.date.today(),
    )

col4, col5 = st.columns([1, 1])
with col4:
    provider = st.selectbox("Nguồn dữ liệu (vnstock)", ["VCI"], index=0)
with col5:
    company_hint = st.text_input("Tên công ty (tùy chọn, tăng độ chính xác tìm tin)", value="Hòa Phát")

run = st.button("Phân tích ngay", type="primary")

if run:
    try:
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start, end = date_range
        else:
            start, end = dt.date.today() - dt.timedelta(days=365), dt.date.today()

        # --- Prices from vnstock ---
        resolved, prices, debug_log = load_prices_vietnam(raw_ticker, start, end, source=provider)
        st.success(f"Đã tìm thấy dữ liệu giá: {resolved} ({len(prices)} phiên)")
        with st.expander("Debug: thông tin tải giá"):
            st.dataframe(debug_log, use_container_width=True, hide_index=True)

        # Candlestick
        st.subheader("Giá lịch sử")
        fig_price = go.Figure()
        fig_price.add_trace(go.Candlestick(
            x=prices["date"],
            open=prices["Open"],
            high=prices["High"],
            low=prices["Low"],
            close=prices["Close"],
            name="Giá",
        ))
        fig_price.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_price, use_container_width=True)

        # Returns
        pxdf = prices.copy()
        pxdf["ret1"] = pxdf["Close"].pct_change()
        pxdf["ret5"] = pxdf["Close"].pct_change(5)

        # --- News + sentiment ---
        query = f"{raw_ticker} {company_hint}".strip()
        news = attach_sentiment(fetch_news(query, days=int(lookback_days)))
        if not news:
            st.warning("Không tìm thấy bài viết nào cho từ khóa đã chọn.")
        else:
            nd = pd.DataFrame([{
                "date": pd.to_datetime(n.published).tz_localize(None).date(),
                "published": pd.to_datetime(n.published).tz_localize(None),
                "title": n.title, "summary": n.summary, "link": n.link,
                "source": n.source, "sentiment": n.sentiment
            } for n in news])

            st.subheader("Tin tức & cảm xúc thị trường")
            st.dataframe(
                nd[["published", "title", "source", "sentiment", "link"]].sort_values("published", ascending=False),
                use_container_width=True, hide_index=True
            )

            # Daily sentiment → merge
            daily_sent = nd.groupby("date")["sentiment"].mean().reset_index().rename(columns={"sentiment": "sent_daily"})
            daily_sent = daily_sent.rename(columns={"date": "news_date"})
            pxdf["date_only"] = pd.to_datetime(pxdf["date"]).dt.date
            merged = pxdf.merge(daily_sent, left_on="date_only", right_on="news_date", how="left")
            merged["sent_daily"] = merged["sent_daily"].fillna(0.0)

            # Rolling sentiment + forward returns
            merged["sent_roll3"] = merged["sent_daily"].rolling(3, min_periods=1).mean()
            merged["fwd_ret1"] = merged["ret1"].shift(-1)
            merged["fwd_ret5"] = merged["ret5"].shift(-5)

            # Correlations
            def _safe_corr(a: str, b: str) -> float:
                try:
                    val = merged[[a, b]].corr().iloc[0, 1]
                    return 0.0 if math.isnan(val) else float(val)
                except Exception:
                    return 0.0
            c1 = _safe_corr("sent_daily", "fwd_ret1")
            c5 = _safe_corr("sent_roll3", "fwd_ret5")
            st.markdown(
                f"**Tương quan**: cùng ngày → lợi suất ngày kế tiếp = **{c1:.3f}** · "
                f"MA3 sentiment → lợi suất 5 ngày tới = **{c5:.3f}**"
            )

            # Plot: Close + sentiment
            from plotly.subplots import make_subplots
            fig1 = make_subplots(specs=[[{"secondary_y": True}]])
            fig1.add_trace(go.Scatter(x=merged["date"], y=merged["Close"], name="Close"), secondary_y=False)
            fig1.add_trace(go.Scatter(x=merged["date"], y=merged["sent_roll3"], name="Sentiment (MA3)"), secondary_y=True)
            fig1.update_yaxes(title_text="Close", secondary_y=False)
            fig1.update_yaxes(title_text="Sentiment (MA3)", secondary_y=True)
            fig1.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig1, use_container_width=True)

            # Scatter (no trendline dependency)
            scat = px.scatter(
                merged,
                x="sent_roll3",
                y="fwd_ret5",
                labels={"sent_roll3": "Sentiment (MA3)", "fwd_ret5": "Lợi suất 5 ngày tới"},
                title="Sentiment (MA3) vs. Lợi suất 5 ngày tới",
            )
            st.plotly_chart(scat, use_container_width=True)

        # Exports
        csv_prices = pxdf.to_csv(index=False).encode("utf-8")
        st.download_button("Tải CSV giá", csv_prices, file_name=f"{raw_ticker}_prices.csv", mime="text/csv")
        if 'nd' in locals():
            csv_news = nd.to_csv(index=False).encode("utf-8")
            st.download_button("Tải CSV tin tức", csv_news, file_name=f"{raw_ticker}_news.csv", mime="text/csv")

        with st.expander("Nguồn & mẹo truy vấn"):
            st.markdown(
                "- Giá: Nguồn Việt Nam qua thư viện `vnstock` (VND).\n"
                "- Tin tức: Google News RSS — thêm tên DN/từ khóa ‘kqkd’, ‘cổ tức’, ‘trái phiếu’ để chính xác hơn."
            )

    except Exception as e:
        st.error(f"Lỗi: {e}")

else:
    st.info("Nhập mã cổ phiếu và bấm **Phân tích ngay** để bắt đầu. Ví dụ: HPG, VNM, MSN, FPT, MBS.")
