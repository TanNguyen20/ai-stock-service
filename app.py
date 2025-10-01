import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
from dateutil import parser
import re
import feedparser

# ---------------------------
# Helpers: vnstock adapters
# ---------------------------
from vnstock import Vnstock, Quote, Listing  # vnstock unified API (2025)  # see docs
# Docs examples show:
# stock = Vnstock().stock(symbol='ACB', source='VCI')
# stock.quote.history(start='2024-01-01', end='2025-03-19', interval='1D')
# :contentReference[oaicite:1]{index=1}

st.set_page_config(page_title="Vietnam Stock + News Lens", layout="wide")

# ---------- Cache -----------
@st.cache_data(show_spinner=False)
def get_all_symbols_df():
    try:
        listing = Listing()
        symbols = listing.all_symbols()  # ticker, exchange, name, etc.
        # normalize for search
        if isinstance(symbols, pd.DataFrame):
            # Expect columns like 'symbol','comName','floor' etc depending on source
            cols_lower = {c: c.lower() for c in symbols.columns}
            symbols.rename(columns=cols_lower, inplace=True)
        return symbols
    except Exception as e:
        st.warning(f"Could not fetch symbol listing: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def resolve_symbol(user_text: str) -> str | None:
    """Accepts 'HPG', 'Hòa Phát', 'MSR', 'Masan', etc. Returns ticker if found."""
    s = user_text.strip().upper()
    # If already looks like a 2-5 char code, try it directly first
    if re.fullmatch(r"[A-Z]{2,5}", s):
        return s
    df = get_all_symbols_df()
    if df.empty:
        return None

    # prepare name column guesses
    name_cols = [c for c in df.columns if "name" in c or "company" in c or "org" in c]
    symbol_col = "symbol" if "symbol" in df.columns else df.columns[0]

    # hard exact match in names
    for nc in name_cols:
        hit = df[df[nc].str.upper().str.contains(s, na=False)]
        if not hit.empty:
            return hit.iloc[0][symbol_col]

    # fallback: includes ascii remove accents?
    try:
        import unicodedata
        def strip_accents(text):
            return "".join(c for c in unicodedata.normalize("NFD", text)
                           if unicodedata.category(c) != 'Mn')
        s_ascii = strip_accents(user_text).upper()
        for nc in name_cols:
            series_ascii = df[nc].fillna("").apply(lambda x: strip_accents(str(x)).upper())
            hit = df[series_ascii.str.contains(s_ascii, na=False)]
            if not hit.empty:
                return hit.iloc[0][symbol_col]
    except Exception:
        pass

    return None

@st.cache_data(show_spinner=False)
def load_history(ticker: str, start: str, end: str, source: str = "VCI") -> pd.DataFrame:
    stock = Vnstock().stock(symbol=ticker, source=source)
    df = stock.quote.history(start=start, end=end, interval='1D')  # OHLCV
    # Expect columns: time, open, high, low, close, volume  (per docs)
    # :contentReference[oaicite:2]{index=2}
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    # Ensure datetime index
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
        df.set_index('time', inplace=True)
    return df

# ---------------------------
# News via Google News RSS
# ---------------------------
VN_NEWS_SITES = [
    "site:vietstock.vn", "site:cafef.vn", "site:ndh.vn",
    "site:vneconomy.vn", "site:vnexpress.net"
]

@st.cache_data(show_spinner=False)
def fetch_news_headlines(query_text: str, limit: int = 20):
    # Build a Google News RSS query
    q = f"{query_text} ({' OR '.join(VN_NEWS_SITES)})"
    rss = f"https://news.google.com/rss/search?q={q}&hl=vi-VN&gl=VN&ceid=VN:vi"
    feed = feedparser.parse(rss)
    items = []
    for entry in (feed.entries or [])[:limit]:
        # Parse date if present
        pub = None
        for key in ['published', 'updated', 'pubDate']:
            if key in entry:
                try:
                    pub = parser.parse(entry[key])
                    break
                except Exception:
                    pass
        items.append({
            "title": entry.get("title"),
            "link": entry.get("link"),
            "published": pub,
            "source": entry.get("source", {}).get("title") if isinstance(entry.get("source"), dict) else None,
            "summary": entry.get("summary")
        })
    news_df = pd.DataFrame(items)
    if not news_df.empty and 'published' in news_df:
        news_df = news_df.sort_values('published', ascending=False)
    return news_df

# ---------------------------
# Lightweight indicators
# ---------------------------
def add_indicators(df: pd.DataFrame, ma_fast=20, ma_slow=50):
    out = df.copy()
    if "close" in out.columns:
        out[f"SMA{ma_fast}"] = out["close"].rolling(ma_fast).mean()
        out[f"SMA{ma_slow}"] = out["close"].rolling(ma_slow).mean()
        out["Return_1W"] = out["close"].pct_change(5)
        out["Return_1M"] = out["close"].pct_change(21)
        out["Return_3M"] = out["close"].pct_change(63)
    return out

def plot_candles(df: pd.DataFrame, symbol: str, ma_fast=20, ma_slow=50):
    if df.empty:
        st.info("No price data to chart.")
        return
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name=symbol
    ))
    if f"SMA{ma_fast}" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA{ma_fast}"], mode='lines', name=f"SMA {ma_fast}"))
    if f"SMA{ma_slow}" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA{ma_slow}"], mode='lines', name=f"SMA {ma_slow}"))
    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title=None, yaxis_title="Price"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# UI
# ---------------------------
st.title("🇻🇳 Vietnam Stock Pricing + News Analyzer")

colA, colB = st.columns([2, 1])
with colA:
    user_input = st.text_input(
        "Enter ticker or company name (e.g., `HPG` or `Hòa Phát`, `MSR` or `Masan High-Tech Materials`):",
        value="HPG"
    )
with colB:
    source = st.selectbox("Data source", options=["VCI", "TCBS"], index=0)  # docs recommend VCI coverage  :contentReference[oaicite:3]{index=3}

# Resolve to ticker
resolved = resolve_symbol(user_input) or (user_input.strip().upper() if re.fullmatch(r"[A-Z]{2,5}", user_input.strip().upper()) else None)

if not resolved:
    st.warning("Could not detect a valid ticker. Try the exact code (e.g., HPG, MSN, VNM) or a different company name.")
    st.stop()

# Dates
default_end = date.today()
default_start = default_end - timedelta(days=365)
c1, c2, c3 = st.columns([1.2, 1.2, 1])
with c1:
    start_date = st.date_input("Start", value=default_start, max_value=default_end)
with c2:
    end_date = st.date_input("End", value=default_end, max_value=default_end)
with c3:
    ma_fast = st.number_input("SMA Fast", min_value=5, max_value=100, value=20, step=1)
    ma_slow = st.number_input("SMA Slow", min_value=10, max_value=250, value=50, step=5)

# Data
hist = load_history(resolved, start_date.isoformat(), end_date.isoformat(), source=source)
if hist.empty:
    st.error("No historical data returned. Try another ticker or date range.")
    st.stop()

hist = add_indicators(hist, ma_fast=ma_fast, ma_slow=ma_slow)

# Header KPIs
last_close = hist["close"].iloc[-1]
ret_1w = hist["Return_1W"].iloc[-1]
ret_1m = hist["Return_1M"].iloc[-1]
ret_3m = hist["Return_3M"].iloc[-1]

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"{resolved} Last Close", f"{last_close:,.0f}")
k2.metric("1W %", f"{ret_1w*100:,.2f}%" if pd.notna(ret_1w) else "—")
k3.metric("1M %", f"{ret_1m*100:,.2f}%" if pd.notna(ret_1m) else "—")
k4.metric("3M %", f"{ret_3m*100:,.2f}%" if pd.notna(ret_3m) else "—")

# Chart
plot_candles(hist, resolved, ma_fast=ma_fast, ma_slow=ma_slow)

# Price table
with st.expander("Show price table"):
    st.dataframe(hist.reset_index().rename(columns={"time":"date"}), use_container_width=True)

# ---------------------------
# News panel
# ---------------------------
st.subheader("📰 Recent news & disclosures (auto-fetched)")
news_query = st.text_input(
    "News search query",
    value=f"{resolved} OR {user_input}",
    help="You can refine this (e.g., add the parent group name, sector, product keywords)."
)
news_df = fetch_news_headlines(news_query, limit=30)

if news_df.empty:
    st.info("No news found from common VN finance sources. Try broadening the query.")
else:
    # Try to highlight items around big price moves in the chosen window
    # Compute daily returns to pair dates roughly with news days
    daily = hist["close"].pct_change().rename("ret")
    def nearest_return(dt):
        if pd.isna(dt): return None
        d = pd.to_datetime(dt.date())
        # choose same day or previous market day
        if d in daily.index:
            return daily.loc[d]
        prev = daily.index[daily.index <= d]
        return daily.loc[prev[-1]] if len(prev) else None

    news_df["~price_return_near"] = news_df["published"].apply(nearest_return)
    # Render
    for _, row in news_df.iterrows():
        title = row["title"] or ""
        link = row["link"] or "#"
        pub = row["published"]
        tag = f" · {row['~price_return_near']*100:,.2f}% daily return" if pd.notna(row["~price_return_near"]) else ""
        sub = (pub.strftime("%Y-%m-%d %H:%M") if pd.notna(pub) else "—") + tag
        st.markdown(f"- [{title}]({link})  \n  <span style='color:gray;font-size:0.9em'>{sub}</span>", unsafe_allow_html=True)

st.caption("Price data via vnstock unified API (Quote.history); headlines via Google News RSS. Data is for research/education only.")
