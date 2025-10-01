# app.py
# Vietnam Stock Pricing + News Analyzer (Streamlit + vnstock)
# -----------------------------------------------------------
# - Ticker resolver: enter code (HPG) or company name (“Hòa Phát”)
# - Price history via vnstock with source fallback (VCI → TCBS → SSI)
# - Candlestick + SMA overlays, 1W/1M/3M returns
# - News via Google News RSS (Vietstock, CafeF, NDH, VnEconomy, VnExpress Biz)
# - FIX: URL-encode Google News query to avoid http.client.InvalidURL
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
from dateutil import parser as dtparser
import time
import re
import feedparser
from urllib.parse import urlencode, quote_plus

# vnstock unified API
# If your vnstock is older, upgrade: pip install -U vnstock
from vnstock import Vnstock

st.set_page_config(page_title="Vietnam Stock + News Analyzer", layout="wide")


# =========================
# Utilities & Caching
# =========================

_CONTROL_CHARS_REGEX = re.compile(r"[\x00-\x1f\x7f]")  # remove control chars that break URLs


def _strip_accents(text: str) -> str:
    try:
        import unicodedata
        return "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )
    except Exception:
        return text


def _sanitize_query_for_url(q: str) -> str:
    """Remove control chars and collapse whitespace for safer URL building."""
    if not isinstance(q, str):
        q = str(q)
    q = _CONTROL_CHARS_REGEX.sub(" ", q)
    q = " ".join(q.split())
    return q


@st.cache_data(show_spinner=False)
def get_all_symbols_df() -> pd.DataFrame:
    """
    Pull all symbols for lookup. vnstock provides various listing helpers
    depending on version; we try a unified approach first.
    """
    try:
        from vnstock import Listing
        listing = Listing()
        df = listing.all_symbols()  # expected DataFrame
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.copy()
            df.columns = [c.lower() for c in df.columns]
            return df
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def resolve_symbol(user_text: str) -> str | None:
    """
    Accepts: 'HPG', 'Hòa Phát', 'MSR', 'Masan', ...
    Returns: ticker code if found else None
    """
    s = (user_text or "").strip()
    if not s:
        return None

    # If user typed a likely ticker already
    if re.fullmatch(r"[A-Za-z]{2,5}", s):
        return s.upper()

    df = get_all_symbols_df()
    if df.empty:
        return None

    # best-guess column names
    symbol_col = "symbol" if "symbol" in df.columns else df.columns[0]
    name_cols = [c for c in df.columns if any(k in c for k in ["name", "company", "org", "organ"])]

    # direct uppercase match on name columns
    s_up = s.upper()
    for nc in name_cols:
        try:
            hit = df[df[nc].astype(str).str.upper().str.contains(s_up, na=False)]
            if not hit.empty:
                return str(hit.iloc[0][symbol_col]).upper()
        except Exception:
            pass

    # accent-insensitive contains
    s_ascii = _strip_accents(s).upper()
    for nc in name_cols:
        try:
            series_ascii = df[nc].fillna("").map(lambda x: _strip_accents(str(x)).upper())
            hit = df[series_ascii.str.contains(s_ascii, na=False)]
            if not hit.empty:
                return str(hit.iloc[0][symbol_col]).upper()
        except Exception:
            pass

    return None


@st.cache_data(show_spinner=False)
def load_history(
    ticker: str,
    start: str,
    end: str,
    source: str = "Auto",
    retries_per_source: int = 2,
    backoff_sec: float = 0.8,
) -> pd.DataFrame:
    """
    Get daily OHLCV for [start, end] with resilient fallback across sources.
    Returns a DataFrame indexed by datetime with columns: open, high, low, close, volume, ...
    Adds df.attrs['source'] = effective_source
    """
    vn = Vnstock()

    sources = [source] if source not in (None, "", "Auto") else ["VCI", "TCBS", "SSI"]
    last_err = None
    attempts_log = []

    for src in sources:
        for attempt in range(1, retries_per_source + 1):
            try:
                stock = vn.stock(symbol=ticker, source=src)
                df = stock.quote.history(start=start, end=end, interval="1D")
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df = df.copy()
                    # normalize index
                    if "time" in df.columns:
                        df["time"] = pd.to_datetime(df["time"])
                        df.sort_values("time", inplace=True)
                        df.set_index("time", inplace=True)
                    # basic cleanup
                    for col in ["open", "high", "low", "close", "volume"]:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                    df.attrs["source"] = src
                    return df
                else:
                    attempts_log.append(f"{src} attempt {attempt}: empty dataframe")
            except Exception as e:
                last_err = e
                attempts_log.append(f"{src} attempt {attempt}: {e}")
                if attempt < retries_per_source:
                    time.sleep(backoff_sec)

    msg = " | ".join(attempts_log[-6:])  # shorten
    raise RuntimeError(f"Failed to fetch {ticker} from sources {sources}. {msg}") from last_err


def add_indicators(df: pd.DataFrame, ma_fast=20, ma_slow=50) -> pd.DataFrame:
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
        x=df.index,
        open=df.get("open"),
        high=df.get("high"),
        low=df.get("low"),
        close=df.get("close"),
        name=symbol,
    ))
    if f"SMA{ma_fast}" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA{ma_fast}"], mode="lines", name=f"SMA {ma_fast}"))
    if f"SMA{ma_slow}" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA{ma_slow}"], mode="lines", name=f"SMA {ma_slow}"))

    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title=None,
        yaxis_title="Price",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================
# News via Google News RSS
# =========================

VN_NEWS_SITES = [
    "site:vietstock.vn", "site:cafef.vn", "site:ndh.vn",
    "site:vneconomy.vn", "site:vnexpress.net"
]


@st.cache_data(show_spinner=False)
def fetch_news_headlines(query_text: str, limit: int = 20):
    """
    Return (DataFrame, error_str). On success, error_str=None.
    Build a URL-encoded Google News RSS query to prevent InvalidURL.
    """
    try:
        # Build a clean, encoded query
        base = "https://news.google.com/rss/search"
        # Remove control characters & tidy whitespace
        safe_query = _sanitize_query_for_url(query_text or "")
        sites_qualifier = " OR ".join(VN_NEWS_SITES)
        q_raw = f"{safe_query} ({sites_qualifier})"

        # urlencode with quote_plus to safely encode spaces, parentheses, accents, etc.
        params = {
            "q": q_raw,
            "hl": "vi-VN",
            "gl": "VN",
            "ceid": "VN:vi",
        }
        rss_url = base + "?" + urlencode(params, quote_via=quote_plus)

        feed = feedparser.parse(rss_url)
        items = []
        for entry in (feed.entries or [])[:limit]:
            pub = None
            for key in ("published", "updated", "pubDate"):
                if key in entry:
                    try:
                        pub = dtparser.parse(entry[key])
                        break
                    except Exception:
                        pass
            src_title = None
            if isinstance(entry.get("source"), dict):
                src_title = entry["source"].get("title")
            items.append({
                "title": entry.get("title"),
                "link": entry.get("link"),
                "published": pub,
                "source": src_title,
                "summary": entry.get("summary"),
            })
        df = pd.DataFrame(items)
        if not df.empty and "published" in df:
            df = df.sort_values("published", ascending=False)
        return df, None
    except Exception as e:
        # Return empty df + error string (do not raise inside cache)
        return pd.DataFrame(), f"{type(e).__name__}: {e}"


# =========================
# UI
# =========================

st.title("🇻🇳 Vietnam Stock Pricing + News Analyzer")

colA, colB = st.columns([2, 1])
with colA:
    user_input = st.text_input(
        "Enter ticker or company name (e.g., HPG / Hòa Phát, MSR / Masan High-Tech Materials):",
        value="HPG",
        placeholder="HPG or Hòa Phát"
    )
with colB:
    source = st.selectbox(
        "Data source",
        options=["Auto", "VCI", "TCBS", "SSI"],
        index=0,
        help="Auto will try VCI → TCBS → SSI until one succeeds."
    )

# Resolve to ticker
resolved = resolve_symbol(user_input) or (
    user_input.strip().upper() if re.fullmatch(r"[A-Za-z]{2,5}", user_input.strip()) else None
)

if not resolved:
    st.warning("⚠️ Could not detect a valid ticker. Try the exact code (HPG, MSN, VNM) or a different company name.")
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

# Data fetch with robust error handling
try:
    hist = load_history(resolved, start_date.isoformat(), end_date.isoformat(), source=source)
except Exception as e:
    st.error(
        "⚠️ Unable to fetch price history right now.\n\n"
        f"Details: {e}\n\n"
        "Tips: select a different source (e.g., TCBS), shorten the date range, or retry shortly."
    )
    st.stop()

hist = add_indicators(hist, ma_fast=ma_fast, ma_slow=ma_slow)
effective_source = hist.attrs.get("source", source)
st.caption(f"Data source in use: **{effective_source}**  ·  Resolved ticker: **{resolved}**")

# Header KPIs
last_close = hist["close"].dropna().iloc[-1] if "close" in hist.columns and not hist["close"].dropna().empty else None
ret_1w = hist["Return_1W"].iloc[-1] if "Return_1W" in hist.columns else None
ret_1m = hist["Return_1M"].iloc[-1] if "Return_1M" in hist.columns else None
ret_3m = hist["Return_3M"].iloc[-1] if "Return_3M" in hist.columns else None

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"{resolved} Last Close", f"{last_close:,.0f}" if last_close is not None else "—")
k2.metric("1W %", f"{ret_1w*100:,.2f}%" if pd.notna(ret_1w) else "—")
k3.metric("1M %", f"{ret_1m*100:,.2f}%" if pd.notna(ret_1m) else "—")
k4.metric("3M %", f"{ret_3m*100:,.2f}%" if pd.notna(ret_3m) else "—")

# Chart
plot_candles(hist, resolved, ma_fast=ma_fast, ma_slow=ma_slow)

# Table
with st.expander("Show price table"):
    tbl = hist.reset_index().rename(columns={"time": "date"})
    st.dataframe(tbl, use_container_width=True)

# =========================
# News
# =========================
st.subheader("📰 Recent news & disclosures")
news_query = st.text_input(
    "News search query",
    value=f"{resolved} OR {user_input}",
    help="Refine if needed (e.g., add parent group name, sector, or keywords)."
)

# Fetch headlines (returns df, error_str)
news_df, news_err = fetch_news_headlines(news_query, limit=30)

if news_err:
    st.warning(
        "Could not fetch news via Google News RSS right now.\n\n"
        f"Details: {news_err}\n\n"
        "Tip: Try simplifying the query (letters/numbers only) or retry shortly."
    )

if news_df.empty:
    st.info("No recent news found from common VN finance sources. Try broadening the query.")
else:
    # Pair headlines with nearest daily return
    daily_ret = hist["close"].pct_change().rename("ret") if "close" in hist.columns else pd.Series(dtype=float)

    def nearest_return(dt):
        try:
            if pd.isna(dt) or daily_ret.empty:
                return None
            d = pd.to_datetime(dt.date())
            if d in daily_ret.index:
                return float(daily_ret.loc[d])
            prev = daily_ret.index[daily_ret.index <= d]
            return float(daily_ret.loc[prev[-1]]) if len(prev) else None
        except Exception:
            return None

    news_df["~price_return_near"] = news_df["published"].apply(nearest_return)

    for _, row in news_df.iterrows():
        title = row.get("title") or ""
        link = row.get("link") or "#"
        pub = row.get("published")
        pub_txt = pub.strftime("%Y-%m-%d %H:%M") if pd.notna(pub) else "—"
        pr = row.get("~price_return_near")
        tag = f" · {pr*100:,.2f}% daily return" if pr is not None else ""
        st.markdown(f"- [{title}]({link})  \n  <span style='color:gray;font-size:0.9em'>{pub_txt}{tag}</span>", unsafe_allow_html=True)

st.caption("Disclaimer: Data is for research/education only. Prices via vnstock; headlines via Google News RSS.")
