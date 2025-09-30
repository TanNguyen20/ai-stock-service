"""
VN Stock News–Price Analyzer

A Streamlit app that lets a user enter a Vietnam stock ticker (e.g., HPG, MSR) and
then:
  • Pulls recent price history from Yahoo Finance (via yfinance)
  • Collects recent Vietnamese news via Google News RSS
  • Scores headline/summary sentiment (lightweight rule-based + optional HF model)
  • Correlates sentiment vs. next‑day returns (lag analysis)
  • Visualizes everything and exports CSVs

Run locally:
  pip install streamlit yfinance feedparser pandas numpy plotly scikit-learn underthesea==6.8.4
  # (Optional, for multilingual transformer sentiment)
  pip install transformers torch --extra-index-url https://download.pytorch.org/whl/cpu

Start the app:
  streamlit run app.py

Note:
  • Yahoo Finance tickers for Vietnam often look like HPG.VN, MSN.VN, VNM.VN.
    This app will auto-try the common suffixes for you.
  • Google News RSS is used for headlines; results depend on what’s publicly indexed.
  • Transformer model is optional and attempted only if available; otherwise the app
    falls back to a simple Vietnamese lexicon approach.
"""

import datetime as dt
import io
import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import feedparser

# Optional deps
try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

# Optional transformer sentiment
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    _HAS_TFM = True
except Exception:
    _HAS_TFM = False

# Optional Vietnamese NLP helpers (very lightweight lemmatization/tokenization)
try:
    from underthesea import word_tokenize
except Exception:
    def word_tokenize(text, format="text"):
        return text


# -------------------------- Config & Helpers -------------------------- #

st.set_page_config(
    page_title="VN Stock News–Price Analyzer",
    layout="wide",
)

VN_NEWS_SOURCES_HINT = [
    "cafef.vn", "vietstock.vn", "vnexpress.net", "nld.com.vn", "tuoitre.vn", "thanhnien.vn",
]

COMMON_SUFFIXES = [".VN", ".HM", ".HSX", ".HNX", ".UPCOM"]

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
        return dt.datetime(*feedparser._parse_date(x)[:6])
    except Exception:
        return dt.datetime.utcnow()


def guess_yf_ticker(raw: str) -> List[str]:
    raw = raw.strip().upper()
    if raw.endswith(tuple(s for s in COMMON_SUFFIXES)):
        return [raw]
    return [raw + s for s in COMMON_SUFFIXES] + [raw]


def load_prices(raw_ticker: str, start: dt.date, end: dt.date) -> Tuple[str, pd.DataFrame]:
    if yf is None:
        raise RuntimeError("yfinance is not installed. Please `pip install yfinance`.")
    for candidate in guess_yf_ticker(raw_ticker):
        try:
            df = yf.download(
                candidate,
                start=start.isoformat(),
                end=(end + dt.timedelta(days=1)).isoformat(),
                progress=False,
            )
            if isinstance(df, pd.DataFrame) and len(df) > 5:
                df = df.rename(columns=str.title)
                df = df.reset_index().rename(columns={"Date": "date"})
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                # Ensure numeric dtypes so Plotly doesn't treat as categorical
                for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=[c for c in ["Open", "High", "Low", "Close"] if c in df.columns])
                if "Close" in df.columns:
                    df = df[df["Close"] > 0]
                if len(df) > 5:
                    return candidate, df
        except Exception:
            pass
    raise RuntimeError("Could not fetch price data. Try another ticker or date range.")
    for candidate in guess_yf_ticker(raw_ticker):
        try:
            df = yf.download(candidate, start=start.isoformat(), end=(end + dt.timedelta(days=1)).isoformat(), progress=False)
            if isinstance(df, pd.DataFrame) and len(df) > 5:
                df = df.rename(columns=str.title)
                df = df.reset_index().rename(columns={"Date": "date"})
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                # --- Ensure numeric dtypes (avoid Plotly categorical axis 0..4 bug) ---
                for col in ["Open","High","Low","Close","Adj Close","Volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                # drop rows with bad/zero prices
                df = df.dropna(subset=[c for c in ["Open","High","Low","Close"] if c in df.columns])
                df = df[df["Close"] > 0]
                if len(df) > 5:
                    return candidate, df
        except Exception:
            pass
    raise RuntimeError("Could not fetch price data. Try another ticker or date range.")
        except Exception:
            pass
    raise RuntimeError("Could not fetch price data. Try another ticker or date range.")


def google_news_rss(query: str, days: int = 30, lang: str = "vi") -> str:
    # Example: https://news.google.com/rss/search?q=HPG+H%C3%B2a+Ph%C3%A1t+when:30d&hl=vi&gl=VN&ceid=VN:vi
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
        source = e.get("source", {}).get("title", e.get("source", "")) if isinstance(e.get("source"), dict) else e.get("source", "")
        published = _coerce_dt(e.get("published", ""))
        items.append(NewsItem(published=published, title=title, summary=summary, link=link, source=source))
    # Deduplicate by title
    dedup = {}
    for it in items:
        key = it.title.lower()
        if key not in dedup:
            dedup[key] = it
    return sorted(dedup.values(), key=lambda x: x.published)


# ---- Sentiment ---- #
VI_POS = set("tăng|kỷ lục|tích cực|lợi nhuận|vượt|bứt phá|khá quan|khả quan|mua ròng|kỷ lục|đỉnh|bùng nổ|thuận lợi".split("|"))
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
    """Return polarity in [-1, 1]. Uses transformer if available, else lexicon."""
    text = (text or "").strip()
    if not text:
        return 0.0

    tok, mdl = _load_model()
    if tok is not None and mdl is not None:
        try:
            with torch.no_grad():
                inputs = tok(text[:512], return_tensors="pt", truncation=True)
                logits = mdl(**inputs).logits[0]
                probs = torch.softmax(logits, dim=0).cpu().numpy()
                # labels: [negative, neutral, positive]
                return float(probs[2] - probs[0])
        except Exception:
            pass

    # Fallback: simple Vietnamese lexicon count
    toks = re.findall(r"\w+", word_tokenize(text.lower()))
    pos = sum(1 for t in toks if t in VI_POS)
    neg = sum(1 for t in toks if t in VI_NEG)
    if pos == neg == 0:
        return 0.0
    return (pos - neg) / max(1, (pos + neg))


def attach_sentiment(items: List[NewsItem]) -> List[NewsItem]:
    for it in items:
        it.sentiment = score_sentiment(f"{it.title}. {it.summary}")
    return items


# ------------------------------ UI ------------------------------ #
st.title("🇻🇳 VN Stock News–Price Analyzer")
st.caption("Nhập mã cổ phiếu (VD: HPG, VNM, MSN, MWG, FPT, VCB…). Ứng dụng sẽ lấy giá lịch sử và tin tức liên quan để phân tích tương quan.")

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    raw_ticker = st.text_input("Mã cổ phiếu", value="HPG")
with col2:
    lookback_days = st.number_input("Số ngày xem tin", value=30, min_value=7, max_value=365)
with col3:
    date_range = st.date_input(
        "Khoảng thời gian giá",
        value=(dt.date.today() - dt.timedelta(days=365), dt.date.today()),
        min_value=dt.date(2010, 1, 1),
        max_value=dt.date.today(),
    )

company_hint = st.text_input("Tên công ty (tùy chọn, tăng độ chính xác tìm tin)", value="Hòa Phát")

run = st.button("Phân tích ngay", type="primary")

if run:
    try:
        start, end = (date_range if isinstance(date_range, (list, tuple)) else (dt.date.today() - dt.timedelta(days=365), dt.date.today()))
        yf_symbol, prices = load_prices(raw_ticker, start, end)

        st.success(f"Đã tìm thấy dữ liệu giá: {yf_symbol} ({len(prices)} phiên)")
        fig_price = go.Figure()
        fig_price.add_trace(go.Candlestick(x=prices["date"], open=prices["Open"], high=prices["High"], low=prices["Low"], close=prices["Close"], name="Giá"))
        fig_price.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
        st.subheader("Giá lịch sử")
        st.plotly_chart(fig_price, use_container_width=True)

        # Compute returns
        pxdf = prices.copy()
        pxdf["ret1"] = pxdf["Close"].pct_change()
        pxdf["ret5"] = pxdf["Close"].pct_change(5)

        # Fetch and score news
        query = f"{raw_ticker} {company_hint}".strip()
        news = attach_sentiment(fetch_news(query, days=int(lookback_days)))
        if not news:
            st.warning("Không tìm thấy bài viết nào cho từ khóa đã chọn.")
        else:
            nd = pd.DataFrame([{ 
                "date": pd.to_datetime(n.published).tz_localize(None).date(),
                "published": pd.to_datetime(n.published).tz_localize(None),
                "title": n.title,
                "summary": n.summary,
                "link": n.link,
                "source": n.source,
                "sentiment": n.sentiment,
            } for n in news])

            st.subheader("Tin tức & cảm xúc thị trường")
            st.dataframe(nd[["published","title","source","sentiment","link"]].sort_values("published", ascending=False), use_container_width=True, hide_index=True)

            # Daily agg sentiment
            daily_sent = nd.groupby("date")["sentiment"].mean().reset_index().rename(columns={"sentiment":"sent_daily"})
            # avoid name clash with pxdf['date'] by renaming the news key
            daily_sent = daily_sent.rename(columns={"date":"news_date"})

            pxdf["date_only"] = pd.to_datetime(pxdf["date"]).dt.date
            merged = pxdf.merge(
                daily_sent,
                left_on="date_only",
                right_on="news_date",
                how="left"
            )
            merged["sent_daily"] = merged["sent_daily"].fillna(0.0)

            # Rolling avg and lag relationship
            merged["sent_roll3"] = merged["sent_daily"].rolling(3, min_periods=1).mean()
            merged["fwd_ret1"] = merged["ret1"].shift(-1)
            merged["fwd_ret5"] = merged["ret5"].shift(-5)

            # Correlations
            corr1 = merged[["sent_daily","fwd_ret1"]].corr().iloc[0,1]
            corr5 = merged[["sent_roll3","fwd_ret5"]].corr().iloc[0,1]

            c1 = 0.0 if math.isnan(corr1) else float(corr1)
            c5 = 0.0 if math.isnan(corr5) else float(corr5)
            st.markdown(f"**Tương quan**: cùng ngày vs. lợi suất ngày kế tiếp = **{c1:.3f}** · trung bình 3 ngày vs. lợi suất 5 ngày tới = **{c5:.3f}**")

            # Plot: sentiment vs. price (use separate axes to avoid internal merges)
            from plotly.subplots import make_subplots
            fig1 = make_subplots(specs=[[{"secondary_y": True}]])
            fig1.add_trace(go.Scatter(x=merged["date"], y=merged["Close"], name="Close"), secondary_y=False)
            fig1.add_trace(go.Scatter(x=merged["date"], y=merged["sent_roll3"], name="Sentiment (MA3)"), secondary_y=True)
            fig1.update_yaxes(title_text="Close", secondary_y=False)
            fig1.update_yaxes(title_text="Sentiment (MA3)", secondary_y=True)
            st.plotly_chart(fig1, use_container_width=True)

            # Scatter with optional trendline (only if statsmodels is present)
            try:
                import statsmodels.api as sm  # noqa: F401
                scat = px.scatter(merged, x="sent_roll3", y="fwd_ret5", trendline="ols", labels={"sent_roll3":"Sentiment (MA3)", "fwd_ret5":"Lợi suất 5 ngày tới"})
            except Exception:
                scat = px.scatter(merged, x="sent_roll3", y="fwd_ret5", labels={"sent_roll3":"Sentiment (MA3)", "fwd_ret5":"Lợi suất 5 ngày tới"})
            st.plotly_chart(scat, use_container_width=True)

            # Export buttons
            csv_prices = pxdf.to_csv(index=False).encode("utf-8")
            csv_news = nd.to_csv(index=False).encode("utf-8")
            st.download_button("Tải CSV giá", csv_prices, file_name=f"{raw_ticker}_prices.csv", mime="text/csv")
            st.download_button("Tải CSV tin tức", csv_news, file_name=f"{raw_ticker}_news.csv", mime="text/csv")

        with st.expander("Nguồn & mẹo truy vấn"):
            st.markdown(
                "- Giá: Yahoo Finance (qua `yfinance`).\n"
                "- Tin tức: Google News RSS (lọc theo từ khóa bạn nhập). Hãy thêm tên doanh nghiệp hoặc từ khóa như 'kqkd', 'cổ tức', 'trái phiếu' để nâng độ chính xác.\n"
                "- Sentiment: mô hình đa ngữ (nếu cài) hoặc từ điển tiếng Việt đơn giản."
            )

    except Exception as e:
        st.error(f"Lỗi: {e}")

else:
    st.info("Nhập mã cổ phiếu và bấm **Phân tích ngay** để bắt đầu. Ví dụ: HPG (Hòa Phát), VNM (Vinamilk), MSN (Masan).")
