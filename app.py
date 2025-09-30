# app.py
"""
VN Stock News–Price Analyzer

Enter a Vietnam stock ticker (e.g., HPG, MSN, VNM) to:
  • Fetch price history from Yahoo Finance (via yfinance)
  • Pull recent Vietnamese news via Google News RSS
  • Score sentiment (simple, multilingual transformer optional)
  • Correlate daily sentiment vs. forward returns
  • Visualize and export CSVs

Minimal deps:
  pip install streamlit yfinance feedparser pandas numpy plotly

Optional (multilingual transformer sentiment):
  # pip install transformers torch --extra-index-url https://download.pytorch.org/whl/cpu

Run:
  streamlit run app.py
"""

import datetime as dt
import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import feedparser

# Optional deps
try:
    import yfinance as yf
except Exception:
    yf = None

# Optional transformer sentiment (multilingual)
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
    import torch  # type: ignore

    _HAS_TFM = True
except Exception:
    _HAS_TFM = False

# -------------------------- Config & Helpers -------------------------- #

st.set_page_config(page_title="VN Stock News–Price Analyzer", layout="wide")

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
        return dt.datetime(*feedparser._parse_date(x)[:6])  # type: ignore[attr-defined]
    except Exception:
        return dt.datetime.utcnow()

def guess_yf_ticker(raw: str) -> List[str]:
    raw = raw.strip().upper()
    if raw.endswith(tuple(COMMON_SUFFIXES)):
        return [raw]
    # Try most likely codes first (.VN then raw)
    return [raw + ".VN", raw] + [raw + s for s in [".HM", ".HSX", ".HNX", ".UPCOM"]]

def _clean_price_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.title)
    if "Date" in df.columns:
        df = df.reset_index().rename(columns={"Date": "date"})
    elif "Datetime" in df.columns:
        df = df.reset_index().rename(columns={"Datetime": "date"})
    elif "date" not in df.columns:
        # Some yfinance versions already return a column named 'date'
        df = df.rename_axis("date").reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    if keep_cols:
        df = df.dropna(subset=keep_cols)
    if "Close" in df.columns:
        df = df[df["Close"] > 0]
    return df

def load_prices(raw_ticker: str, start: dt.date, end: dt.date) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    Try multiple methods and suffixes to fetch data from Yahoo Finance.
    Returns (resolved_symbol, dataframe, debug_log_df)
    """
    if yf is None:
        raise RuntimeError("yfinance is not installed. Please `pip install yfinance`.")

    attempts: List[Dict[str, Any]] = []
    # Add a small buffer to end date for inclusive ranges
    end_plus = end + dt.timedelta(days=1)

    for candidate in guess_yf_ticker(raw_ticker):
        # Method 1: yf.download with start/end
        try:
            df1 = yf.download(candidate, start=start.isoformat(), end=end_plus.isoformat(), progress=False)
            n1 = int(len(df1)) if isinstance(df1, pd.DataFrame) else 0
            attempts.append({"symbol": candidate, "method": "download(start/end)", "rows": n1})
            if n1 > 0:
                cleaned = _clean_price_df(df1)
                if len(cleaned) > 0:
                    return candidate, cleaned, pd.DataFrame(attempts)
        except Exception as e:
            attempts.append({"symbol": candidate, "method": "download(start/end) ERROR", "rows": 0, "msg": str(e)})

        # Method 2: Ticker.history with start/end
        try:
            tkr = yf.Ticker(candidate)
            df2 = tkr.history(start=start.isoformat(), end=end_plus.isoformat(), interval="1d", auto_adjust=False)
            n2 = int(len(df2)) if isinstance(df2, pd.DataFrame) else 0
            attempts.append({"symbol": candidate, "method": "history(start/end)", "rows": n2})
            if n2 > 0:
                cleaned = _clean_price_df(df2)
                if len(cleaned) > 0:
                    return candidate, cleaned, pd.DataFrame(attempts)
        except Exception as e:
            attempts.append({"symbol": candidate, "method": "history(start/end) ERROR", "rows": 0, "msg": str(e)})

        # Method 3: Ticker.history with period (fallback, ignores specific start/end)
        try:
            # choose period based on requested span (cap at 5y)
            span_days = max(1, (end - start).days)
            if span_days <= 365:
                period = "1y"
            elif span_days <= 365 * 2:
                period = "2y"
            elif span_days <= 365 * 5:
                period = "5y"
            else:
                period = "10y"
            tkr = yf.Ticker(candidate)
            df3 = tkr.history(period=period, interval="1d", auto_adjust=False)
            n3 = int(len(df3)) if isinstance(df3, pd.DataFrame) else 0
            attempts.append({"symbol": candidate, "method": f"history(period={period})", "rows": n3})
            if n3 > 0:
                cleaned = _clean_price_df(df3)
                # filter to requested window if possible
                mask = (cleaned["date"].dt.date >= start) & (cleaned["date"].dt.date <= end)
                cleaned = cleaned.loc[mask].reset_index(drop=True)
                if len(cleaned) == 0:
                    # If filtering removed all rows, still return the unfiltered data as a last resort
                    cleaned = _clean_price_df(df3)
                if len(cleaned) > 0:
                    return candidate, cleaned, pd.DataFrame(attempts)
        except Exception as e:
            attempts.append({"symbol": candidate, "method": "history(period) ERROR", "rows": 0, "msg": str(e)})

    # If we got here, all attempts failed
    debug_df = pd.DataFrame(attempts)
    raise RuntimeError(f"No data returned for {raw_ticker}. Tried: {', '.join(debug_df['symbol'].unique())}")

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
    # Deduplicate by title
    dedup = {}
    for it in items:
        key = it.title.lower()
        if key not in dedup:
            dedup[key] = it
    return sorted(dedup.values(), key=lambda x: x.published)

# ---- Sentiment ---- #
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
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start, end = date_range
        else:
            start, end = dt.date.today() - dt.timedelta(days=365), dt.date.today()

        # Prices (with debug log)
        resolved, prices, debug_log = load_prices(raw_ticker, start, end)
        st.success(f"Đã tìm thấy dữ liệu giá: {resolved} ({len(prices)} phiên)")

        with st.expander("Debug: các phương án tải giá đã thử"):
            st.dataframe(debug_log, use_container_width=True, hide_index=True)

        # Candlestick
        st.subheader("Giá lịch sử")
        fig_price = go.Figure()
        fig_price.add_trace(go.Candlestick(
            x=prices["date"], open=prices["Open"], high=prices["High"], low=prices["Low"], close=prices["Close"], name="Giá"
        ))
        fig_price.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_price, use_container_width=True)

        # Returns
        pxdf = prices.copy()
        if "Close" not in pxdf.columns:
            st.warning("Thiếu cột Close trong dữ liệu giá.")
        pxdf["ret1"] = pxdf["Close"].pct_change()
        pxdf["ret5"] = pxdf["Close"].pct_change(5)

        # News + sentiment
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
            st.dataframe(nd[["published", "title", "source", "sentiment", "link"]].sort_values("published", ascending=False),
                        use_container_width=True, hide_index=True)

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

        # Export buttons
        csv_prices = pxdf.to_csv(index=False).encode("utf-8")
        st.download_button("Tải CSV giá", csv_prices, file_name=f"{raw_ticker}_prices.csv", mime="text/csv")
        if 'nd' in locals():
            csv_news = nd.to_csv(index=False).encode("utf-8")
            st.download_button("Tải CSV tin tức", csv_news, file_name=f"{raw_ticker}_news.csv", mime="text/csv")

        with st.expander("Nguồn & mẹo truy vấn"):
            st.markdown(
                "- Giá: Yahoo Finance (qua `yfinance`).\n"
                "- Tin tức: Google News RSS (lọc theo từ khóa bạn nhập). Thêm tên doanh nghiệp hoặc từ khóa như 'kqkd', 'cổ tức', 'trái phiếu' để nâng độ chính xác.\n"
                "- Sentiment: mô hình đa ngữ (nếu cài) hoặc từ điển tiếng Việt đơn giản."
            )

    except Exception as e:
        st.error(f"Lỗi: {e}")

else:
    st.info("Nhập mã cổ phiếu và bấm **Phân tích ngay** để bắt đầu. Ví dụ: HPG (Hòa Phát), VNM (Vinamilk), MSN (Masan).")
