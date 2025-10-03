# app.py
# Vietnam Stock Pricing + News Analyzer (Streamlit + vnstock)
# -----------------------------------------------------------
# - Resolve ticker from Vietnamese/English company names (alphanumeric tickers OK: TV2, FRT…)
# - Price history via vnstock with source fallback (VCI → TCBS → SSI) + retries
# - Candlestick + SMA overlays, 1W/1M/3M returns
# - News via Google News RSS (URL-encoded to avoid InvalidURL)
# - Forecasts:
#    * Price-only: Holt-Winters, ARIMA(1,1,1), SMA, Naive
#    * News-aware (optional): ARIMAX with headline sentiment as exogenous input
# - Recommender: N-day-low + rebound screener
# - Robust listings: retries, TTL cache, manual refresh, disk fallback universe
# - UI debug logger (instead of print) to view diagnostics in the page
# -----------------------------------------------------------

import os
import re
import time
import feedparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from datetime import date, timedelta, datetime, UTC
from dateutil import parser as dtparser
from urllib.parse import urlencode, quote_plus

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

_TRANSFORMERS_OK = True
_SARIMAX_OK = True
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
except Exception:
    _TRANSFORMERS_OK = False
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    _SARIMAX_OK = False

from vnstock import Vnstock

st.set_page_config(page_title="Vietnam Stock + News + Forecast", layout="wide")

# =========================
# UI Logger (instead of print)
# =========================
def _log(msg: str):
    """Append debug text to a session-scoped buffer for showing in the UI."""
    buf = st.session_state.get("_debug_log", [])
    buf.append(str(msg))
    st.session_state["_debug_log"] = buf

# =========================
# Utilities & Caching
# =========================

# Fallback universe (VN30 / liquid HOSE names; tweak as you like)
FALLBACK_UNIVERSE = [
    "VCB","BID","CTG","TCB","VPB","MBB","STB","HDB","VIB","ACB","SHB",
    "VIC","VHM","VRE","NVL","PDR","KDH",
    "VNM","MSN","SAB","HPG","FPT","MWG","GVR","GAS","PLX","POW","REE","VJC","HVN",
    "SSI","VND","TCI","SHS","VIX","HCM","VCI","FTS"
]

# On-disk symbols cache (survives cold boots on Streamlit Cloud)
SYMBOLS_CACHE_PATH = "symbols_cache.parquet"

def _load_symbols_cache() -> pd.DataFrame:
    try:
        if os.path.exists(SYMBOLS_CACHE_PATH):
            df = pd.read_parquet(SYMBOLS_CACHE_PATH)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.copy()
                df.columns = [c.lower() for c in df.columns]
                _log(f"[cache] Loaded symbols_cache.parquet with shape={df.shape}")
                return df
    except Exception as e:
        _log(f"[cache] Failed to read symbols_cache.parquet: {e}")
    return pd.DataFrame()

def _save_symbols_cache(df: pd.DataFrame) -> None:
    try:
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_parquet(SYMBOLS_CACHE_PATH, index=False)
            _log(f"[cache] Saved symbols_cache.parquet with shape={df.shape}")
    except Exception as e:
        _log(f"[cache] Failed to save symbols_cache.parquet: {e}")

_CONTROL_CHARS_REGEX = re.compile(r"[\x00-\x1f\x7f]")  # remove control chars that break URLs

def _strip_accents(text: str) -> str:
    # Remove Vietnamese diacritics for fuzzy matching
    try:
        import unicodedata
        return "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )
    except Exception:
        return text

def _sanitize_query_for_url(q: str) -> str:
    # Remove control chars and collapse whitespace for safe URLs
    if not isinstance(q, str):
        q = str(q)
    q = _CONTROL_CHARS_REGEX.sub(" ", q)
    q = " ".join(q.split())
    return q

# TTL so it refreshes periodically; on failure use disk cache; raise if nothing
@st.cache_data(show_spinner=False, ttl=3600, persist="disk")
def get_all_symbols_df() -> pd.DataFrame:
    """
    Try live vnstock Listing() (with retries). On failure, use a disk cache if available.
    If both fail, raise so caller can fall back to VN30 universe.
    """
    try:
        from vnstock import Listing
        listing = Listing()
        last_exc = None
        for attempt in range(1, 4):
            try:
                _log(f"[listing] Attempt {attempt} → Listing().all_symbols()")
                df = listing.all_symbols()
                _log(f"[listing] Raw shape: {getattr(df, 'shape', None)}")
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df = df.copy()
                    df.columns = [c.lower() for c in df.columns]
                    st.session_state["symbols_error"] = None
                    _save_symbols_cache(df)  # Persist for offline/rate-limited runs
                    return df
                time.sleep(0.5)
            except Exception as e:
                last_exc = e
                _log(f"[listing] Attempt {attempt} failed: {type(e).__name__}: {e}")
                time.sleep(0.7)

        msg = f"Empty/failed listing response. Last error: {type(last_exc).__name__}: {last_exc}" if last_exc else "Empty listing response."
        st.session_state["symbols_error"] = msg
        _log(f"[listing] Live fetch failed. {msg}")

        # Disk cache fallback
        cache_df = _load_symbols_cache()
        if not cache_df.empty:
            st.session_state["symbols_error"] = f"{msg} · Using cached symbols on disk."
            return cache_df

        raise RuntimeError(msg)  # nothing to return
    except Exception as e:
        st.session_state["symbols_error"] = f"{type(e).__name__}: {e}"
        _log(f"[listing] Unexpected error: {e}")
        cache_df = _load_symbols_cache()
        if not cache_df.empty:
            st.session_state["symbols_error"] += " · Using cached symbols on disk."
            return cache_df
        raise

@st.cache_data(show_spinner=False)
def resolve_symbol(user_text: str) -> str | None:
    # Accept: HPG, Hòa Phát, MSR, Masan, TV2, etc. Return ticker if found
    s = (user_text or "").strip()
    if not s:
        return None

    # Allow alphanumeric tickers (2–6 chars), e.g., TV2
    if re.fullmatch(r"[A-Za-z0-9]{2,6}", s):
        return s.upper()

    # Try listing-based fuzzy match but survive listing failures
    try:
        df = get_all_symbols_df()
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        return None

    symbol_col = "symbol" if "symbol" in df.columns else df.columns[0]
    name_cols = [c for c in df.columns if any(k in c for k in ["name", "company", "org", "organ"])]

    # direct uppercase contains
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

@st.cache_resource(show_spinner=False, ttl=3600)
def get_vnstock():
    return Vnstock()

@st.cache_data(show_spinner=False)
def load_history(
    ticker: str,
    start: str,
    end: str,
    source: str = "Auto",
    retries_per_source: int = 2,
    backoff_sec: float = 0.8,
) -> pd.DataFrame:
    # Fetch OHLCV for [start, end] with fallback sources
    vn = get_vnstock()
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
                    if "time" in df.columns:
                        df["time"] = pd.to_datetime(df["time"])
                        df.sort_values("time", inplace=True)
                        df.set_index("time", inplace=True)
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

    msg = " | ".join(attempts_log[-6:])
    raise RuntimeError(f"Failed to fetch {ticker} from {sources}. {msg}") from last_err

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
        open=df.get("open"), high=df.get("high"),
        low=df.get("low"), close=df.get("close"),
        name=symbol,
    ))
    if f"SMA{ma_fast}" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA{ma_fast}"], mode="lines", name=f"SMA {ma_fast}"))
    if f"SMA{ma_slow}" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA{ma_slow}"], mode="lines", name=f"SMA {ma_slow}"))
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=0),
                      xaxis_title=None, yaxis_title="Price", hovermode="x unified")
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
    # Safely build a Google News RSS query (URL-encoded)
    try:
        base = "https://news.google.com/rss/search"
        safe_query = _sanitize_query_for_url(query_text or "")
        sites_qualifier = " OR ".join(VN_NEWS_SITES)
        q_raw = f"{safe_query} ({sites_qualifier})"
        params = {"q": q_raw, "hl": "vi-VN", "gl": "VN", "ceid": "VN:vi"}
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
        return pd.DataFrame(), f"{type(e).__name__}: {e}"

# =========================
# Price-only Forecast Helpers
# =========================

def _prep_close_series(df: pd.DataFrame):
    # Clean daily close series on business days
    s = df["close"].dropna().copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    bdays = pd.date_range(s.index.min(), s.index.max(), freq="B")
    s = s.reindex(bdays).ffill()
    return s

def _recent_volatility(s: pd.Series, window: int = 21):
    r = s.pct_change().dropna()
    if len(r) < 2:
        return 0.0
    return float(r.tail(window).std())

def make_forecast(s: pd.Series, horizon_days: int, method: str = "Holt-Winters"):
    # Return DataFrame (yhat, lower, upper) for next business days
    horizon_days = int(horizon_days)
    future_idx = pd.bdate_range(s.index[-1] + pd.tseries.offsets.BDay(1), periods=horizon_days, freq="B")
    vol = _recent_volatility(s)
    last = float(s.iloc[-1])

    if method == "Naive (last value)":
        yhat = pd.Series(last, index=future_idx)
        steps = np.arange(1, horizon_days + 1)
        band = 1.65 * last * vol * np.sqrt(steps)
        return pd.DataFrame({"yhat": yhat, "lower": yhat.values - band, "upper": yhat.values + band}, index=future_idx)

    if method == "SMA (window=20)":
        sma = s.rolling(20).mean().iloc[-1]
        yhat = pd.Series(float(sma), index=future_idx)
        steps = np.arange(1, horizon_days + 1)
        band = 1.65 * float(s.iloc[-1]) * vol * np.sqrt(steps)
        return pd.DataFrame({"yhat": yhat, "lower": yhat.values - band, "upper": yhat.values + band}, index=future_idx)

    if method == "Holt-Winters":
        try:
            model = ExponentialSmoothing(s, trend="add", seasonal=None, initialization_method="estimated")
            fit = model.fit(optimized=True)
            fc = fit.forecast(horizon_days)
            resid = s - fit.fittedvalues.reindex_like(s)
            sigma = float(resid.dropna().std())
            steps = np.arange(1, horizon_days + 1)
            band = 1.65 * sigma * np.sqrt(steps)
            return pd.DataFrame({"yhat": fc, "lower": fc.values - band, "upper": fc.values + band}, index=future_idx)
        except Exception:
            pass

    try:
        model = ARIMA(s, order=(1, 1, 1))
        fit = model.fit()
        pred = fit.get_forecast(steps=horizon_days)
        mean = pred.predicted_mean
        conf = pred.conf_int(alpha=0.10)  # ~90%
        return pd.DataFrame({"yhat": mean, "lower": conf.iloc[:, 0], "upper": conf.iloc[:, 1]}, index=future_idx)
    except Exception:
        yhat = pd.Series(last, index=future_idx)
        steps = np.arange(1, horizon_days + 1)
        band = 1.65 * last * vol * np.sqrt(steps)
        return pd.DataFrame({"yhat": yhat, "lower": yhat.values - band, "upper": yhat.values + band}, index=future_idx)

def plot_forecast(hist_close: pd.Series, fc_df: pd.DataFrame, symbol: str, label_suffix: str = ""):
    if fc_df is None or fc_df.empty:
        st.info("No forecast to plot.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_close.index, y=hist_close.values, mode="lines", name=f"{symbol} Close"))
    fig.add_trace(go.Scatter(
        x=list(fc_df.index) + list(fc_df.index[::-1]),
        y=list(fc_df["upper"].values) + list(fc_df["lower"].values[::-1]),
        fill="toself", opacity=0.2, line=dict(width=0), name=f"Forecast band{label_suffix}"
    ))
    fig.add_trace(go.Scatter(x=fc_df.index, y=fc_df["yhat"], mode="lines", name=f"Forecast{label_suffix}"))
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0),
                      xaxis_title=None, yaxis_title="Price", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# Sentiment / News-aware Forecast
# =========================

@st.cache_resource(show_spinner=False)
def _load_sentiment_pipeline():
    # Lazy-load multilingual sentiment if transformers is available. Returns pipeline or None.
    if not _TRANSFORMERS_OK:
        return None
    try:
        model_id = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
        return TextClassificationPipeline(model=mdl, tokenizer=tok, framework="pt", return_all_scores=True)
    except Exception:
        return None

def _score_headlines_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    # Return DataFrame: published_date (date), sent in [-1,1]
    if news_df is None or news_df.empty:
        return pd.DataFrame(columns=["published_date", "sent"])

    pipe = _load_sentiment_pipeline()
    if pipe is None:
        df = news_df.dropna(subset=["published"]).copy()
        df["published_date"] = pd.to_datetime(df["published"]).dt.date
        return df[["published_date"]].assign(sent=0.0).drop_duplicates()

    df = news_df.dropna(subset=["title", "published"]).copy()
    df["published_date"] = pd.to_datetime(df["published"]).dt.date
    texts = df["title"].astype(str).tolist()

    scores = []
    B = 16
    for i in range(0, len(texts), B):
        chunk = texts[i:i+B]
        out = pipe(chunk)
        for res in out:
            d = {x["label"].lower(): x["score"] for x in res}
            s = d.get("positive", 0) - d.get("negative", 0)
            scores.append(float(s))
    df["sent"] = scores[:len(df)]
    daily = df.groupby("published_date")["sent"].mean().rename("sent").to_frame()
    return daily.reset_index()

def _build_daily_sentiment_series(news_df: pd.DataFrame, hist_index: pd.DatetimeIndex) -> pd.Series:
    # Align daily mean sentiment to price index; ffill a few days (weekend news)
    if news_df is None or news_df.empty:
        return pd.Series(0.0, index=hist_index)

    daily_sent_df = _score_headlines_sentiment(news_df)
    if daily_sent_df.empty:
        return pd.Series(0.0, index=hist_index)

    s = daily_sent_df.set_index(pd.to_datetime(daily_sent_df["published_date"]))["sent"]
    s = s.reindex(hist_index, method=None).fillna(method="ffill", limit=3).fillna(0.0)
    return s.astype(float)

def make_arimax_with_sentiment(close_series: pd.Series,
                               sentiment_series: pd.Series,
                               horizon_days: int):
    # Train SARIMAX (ARIMA) with exogenous daily sentiment and forecast
    if not _SARIMAX_OK:
        raise RuntimeError("statsmodels.sarimax not available. Install statsmodels >= 0.13.")

    s = close_series.dropna().copy()
    s.index = pd.to_datetime(s.index)
    s = s.asfreq("B").ffill()
    exog = sentiment_series.reindex(s.index).fillna(0.0)

    order = (1, 1, 1)
    model = SARIMAX(s, order=order, exog=exog, enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)

    future_idx = pd.bdate_range(s.index[-1] + pd.tseries.offsets.BDay(1), periods=horizon_days, freq="B")
    future_exog = pd.Series(0.0, index=future_idx)

    pred = fit.get_forecast(steps=horizon_days, exog=future_exog.values.reshape(-1, 1))
    mean = pred.predicted_mean
    conf = pred.conf_int(alpha=0.10)
    out = pd.DataFrame({"yhat": mean, "lower": conf.iloc[:, 0], "upper": conf.iloc[:, 1]}, index=future_idx)
    return out

# =========================
# Simple indicators for screener
# =========================

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    s = series.dropna().astype(float)
    delta = s.diff()
    up = (delta.clip(lower=0)).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / (down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.reindex(series.index)

def _near_n_day_low(df: pd.DataFrame, n: int, tol_pct: float) -> tuple[bool, float, float]:
    # Return (is_near_low, n_low, dist_pct)
    if df.empty or "close" not in df.columns:
        return False, np.nan, np.nan
    closes = df["close"].astype(float)
    if closes.shape[0] < n:
        return False, np.nan, np.nan
    n_low = closes.tail(n).min()
    last = closes.iloc[-1]
    dist = (last / n_low - 1.0) * 100.0
    return dist <= tol_pct, n_low, dist

def _rebound_signal(df: pd.DataFrame) -> tuple[bool, dict]:
    # Rebound: Close > SMA20, SMA20 rising, RSI(14) rising vs 3 days ago
    meta = {"close_gt_sma20": False, "sma20_rising": False, "rsi_rising": False, "rsi": np.nan}
    if df.empty or "close" not in df.columns:
        return False, meta
    s = df["close"].astype(float)
    sma20 = s.rolling(20).mean()
    rsi = _rsi(s, 14)
    if s.shape[0] < 25:
        meta["rsi"] = float(rsi.iloc[-1]) if not rsi.dropna().empty else np.nan
        return False, meta
    close_gt_sma20 = bool(s.iloc[-1] > sma20.iloc[-1])
    sma20_rising = bool(sma20.iloc[-1] > sma20.iloc[-2])
    rsi_rising = False
    if rsi.dropna().shape[0] >= 4:
        rsi_rising = bool(rsi.iloc[-1] > rsi.iloc[-4])
    meta.update({
        "close_gt_sma20": close_gt_sma20,
        "sma20_rising": sma20_rising,
        "rsi_rising": rsi_rising,
        "rsi": float(rsi.iloc[-1]) if not rsi.dropna().empty else np.nan
    })
    ok = close_gt_sma20 and sma20_rising and rsi_rising
    return ok, meta

# =========================
# UI
# =========================

st.title("🇻🇳 Vietnam Stock Pricing + News + Forecast")

colA, colB = st.columns([2, 1])
with colA:
    user_input = st.text_input(
        "Enter ticker or company name (e.g., HPG / Hòa Phát, MSR / Masan High-Tech Materials):",
        value="HPG",
        placeholder="HPG, TV2, or Hòa Phát"
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
    user_input.strip().upper() if re.fullmatch(r"[A-Za-z0-9]{2,6}", user_input.strip()) else None
)
if not resolved:
    st.warning("⚠️ Could not detect a valid ticker. Try the exact code (HPG, TV2, MSN, VNM) or a different company name.")
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

# Data fetch
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

# KPIs
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

# Price-only Forecast
st.subheader("🔮 Price-only forecast (experimental)")
with st.expander("Show forecast options"):
    horizon = st.slider("Forecast horizon (business days)", 5, 60, 20, step=5)
    method = st.selectbox("Method", ["Holt-Winters", "ARIMA(1,1,1)", "SMA (window=20)", "Naive (last value)"])
    s_close = _prep_close_series(hist)
    fc = make_forecast(s_close, horizon_days=horizon, method=method)
    plot_forecast(s_close, fc, resolved)
    st.caption("Forecasts are illustrative only. They are not investment advice.")

# Table
with st.expander("Show price table"):
    st.dataframe(hist.reset_index().rename(columns={"time": "date"}), use_container_width=True)

# News
st.subheader("📰 Recent news & disclosures")
news_query = st.text_input(
    "News search query",
    value=f"{resolved} OR {user_input}",
    help="Refine if needed (e.g., add parent group name, sector, or keywords)."
)
news_df, news_err = fetch_news_headlines(news_query, limit=30)

if news_err:
    st.warning(
        "Could not fetch news via Google News RSS right now.\n\n"
        f"Details: {news_err}\n\n"
        "Tip: Try simplifying the query or retry shortly."
    )

if news_df.empty:
    st.info("No recent news found from common VN finance sources. Try broadening the query.")
else:
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

# News-aware Forecast
st.subheader("🧠 News-aware forecast (experimental)")
with st.expander("Use headlines sentiment as an exogenous feature"):
    st.caption("We convert recent Vietnamese headlines into a daily sentiment score and feed it into an ARIMAX model.")
    horizon_news = st.slider("Horizon (business days)", 5, 60, 20, step=5)
    run_news_forecast = st.checkbox("Enable news-aware forecast", value=False,
                                    help="Requires 'transformers' & 'torch' on first use (model download).")

    if run_news_forecast:
        s_close2 = hist["close"].dropna().copy()
        s_close2.index = pd.to_datetime(s_close2.index)
        s_close2 = s_close2.asfreq("B").ffill()

        sent_series = _build_daily_sentiment_series(news_df, s_close2.index)

        try:
            fc_news = make_arimax_with_sentiment(s_close2, sent_series, horizon_days=horizon_news)
            plot_forecast(s_close2, fc_news, resolved, label_suffix=" (news-aware)")
            with st.expander("Show recent daily sentiment used in the model"):
                df_view = pd.DataFrame({
                    "date": s_close2.index,
                    "sentiment": sent_series.reindex(s_close2.index).values
                }).tail(40)
                st.dataframe(df_view, use_container_width=True)
        except Exception as e:
            if not _TRANSFORMERS_OK:
                st.error("Transformers not installed. Run:\n\n"
                         "`pip install -U transformers torch --extra-index-url https://download.pytorch.org/whl/cpu`")
            elif not _SARIMAX_OK:
                st.error("SARIMAX not available. Ensure statsmodels is installed and up to date:\n\n"
                         "`pip install -U statsmodels`")
            else:
                st.error(f"Could not run news-aware forecast: {e}")

# ============================================================
# Recommender: Buy-the-dip candidates (N-day low + rebound)
# ============================================================

st.subheader("🎯 Buy-the-dip candidates (N-day low + rebound)")

with st.expander("Scan settings"):
    colr1, colr2 = st.columns([1,3])
    with colr1:
        if st.button("↻ Refresh symbols"):
            st.cache_data.clear()   # clear cached listing & other data
            st.rerun()

    # Optional diagnostics
    err = st.session_state.get("symbols_error")
    if err:
        st.caption(f"Listing diagnostics: {err}")

    # Show UI debug logs
    logs = st.session_state.get("_debug_log", [])
    if logs:
        with st.expander("Debug logs"):
            st.text("\n".join(logs[-400:]))

    col1, col2, col3 = st.columns(3)
    with col1:
        lookback_n = st.number_input("Lookback N (days)", min_value=20, max_value=250, value=60, step=5)
        near_low_pct = st.number_input("Near-low threshold (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    with col2:
        history_days = st.number_input("History window (days)", min_value=int(lookback_n), max_value=800, value=max(180, int(lookback_n)+20), step=10)
        min_avg_vol = st.number_input("Min avg volume (last 20d)", min_value=0, max_value=10_000_000, value=50_000, step=10_000)
    with col3:
        universe_mode = st.selectbox("Universe", ["All symbols (auto)", "Custom list"])
        max_symbols = st.number_input("Max symbols to scan", min_value=10, max_value=500, value=120, step=10)

    custom_list = ""
    if universe_mode == "Custom list":
        custom_list = st.text_area("Tickers (comma / space separated, e.g., HPG, VNM, TV2)", value="HPG, VNM, TV2, FPT")

    run_scan = st.button("Run scan")

def _parse_custom_list(text: str) -> list[str]:
    if not text:
        return []
    ticks = re.split(r"[,\s]+", text.strip().upper())
    ticks = [t for t in ticks if re.fullmatch(r"[A-Z0-9]{2,6}", t)]
    return list(dict.fromkeys(ticks))  # unique, keep order

@st.cache_data(show_spinner=False)
def _get_universe(universe_mode: str, custom_text: str, limit: int) -> list[str]:
    if universe_mode == "Custom list":
        arr = _parse_custom_list(custom_text)
        return arr[:limit] if limit else arr

    # Try live listings; on failure/empty, we fall back
    try:
        df = get_all_symbols_df()
        sym_col = "symbol" if "symbol" in df.columns else df.columns[0]
        syms = df[sym_col].astype(str).str.upper().tolist()
        syms = [s for s in syms if re.fullmatch(r"[A-Z0-9]{2,6}", s)]
        if syms:
            return syms[:limit]
    except Exception:
        pass

    # Fallback universe
    return FALLBACK_UNIVERSE[:limit] if limit else FALLBACK_UNIVERSE

def _safe_load_hist_for_screener(ticker: str, days: int, source: str) -> pd.DataFrame:
    # Use timezone-aware UTC (no utcnow deprecation)
    today_utc = datetime.now(UTC).date()
    end = today_utc.isoformat()
    start = (today_utc - timedelta(days=days)).isoformat()
    try:
        return load_history(ticker, start, end, source=source)
    except Exception:
        return pd.DataFrame()

def _avg_volume(df: pd.DataFrame, window: int = 20) -> float:
    if df.empty or "volume" not in df.columns:
        return 0.0
    return float(pd.to_numeric(df["volume"], errors="coerce").tail(window).mean())

if run_scan:
    universe = _get_universe(universe_mode, custom_list, int(max_symbols))
    if not universe:
        st.warning("No symbols to scan. Try a custom list (e.g., HPG, VNM, FPT).")
        err = st.session_state.get("symbols_error")
        if err:
            st.caption(f"Listing diagnostics: {err}")
    else:
        # Note when fallback was used
        used_fallback = False
        try:
            _ = get_all_symbols_df()  # will raise if listing failed (meaning fallback used)
        except Exception:
            used_fallback = True
        if universe_mode == "All symbols (auto)" and used_fallback:
            st.caption("Using fallback universe (VN30/liquid HOSE) because listings were unavailable.")

        rows = []
        progress = st.progress(0.0, text="Scanning…")
        for i, sym in enumerate(universe, start=1):
            progress.progress(i/len(universe), text=f"Scanning {sym} ({i}/{len(universe)})")
            df = _safe_load_hist_for_screener(sym, int(history_days), source=source)
            if df.empty or df.shape[0] < max(lookback_n, 25):
                continue
            if _avg_volume(df, 20) < min_avg_vol:
                continue

            is_near, nlow, dist = _near_n_day_low(df, int(lookback_n), float(near_low_pct))
            if not is_near:
                continue
            rebound_ok, meta = _rebound_signal(df)
            if not rebound_ok:
                continue

            last_close_val = float(df["close"].iloc[-1])
            ret_5d = float(df["close"].pct_change(5).iloc[-1]) if df.shape[0] >= 6 else np.nan
            vol20 = _avg_volume(df, 20)

            rows.append({
                "Ticker": sym,
                "Close": round(last_close_val, 2),
                f"N({lookback_n}) Low": round(nlow, 2) if pd.notna(nlow) else np.nan,
                "Dist to N-low (%)": round(dist, 2),
                "SMA20 rising": meta["sma20_rising"],
                "RSI(14)": round(meta["rsi"], 1) if pd.notna(meta["rsi"]) else np.nan,
                "Close>SMA20": meta["close_gt_sma20"],
                "5D Return (%)": round(ret_5d*100, 2) if pd.notna(ret_5d) else np.nan,
                "Avg Vol(20d)": int(vol20),
            })

        progress.empty()

        if not rows:
            st.info("No candidates found with the current filters. Try increasing the near-low threshold or lowering min volume.")
        else:
            res = pd.DataFrame(rows)
            res["rank_score"] = (-res["Dist to N-low (%)"].fillna(999)
                                 + res["5D Return (%)"].fillna(0)
                                 + (res["Avg Vol(20d)"].fillna(0) / 100000))
            res = res.sort_values("rank_score", ascending=False).drop(columns=["rank_score"])

            st.success(f"Found {len(res)} candidates.")
            st.dataframe(res, use_container_width=True)

            csv = res.to_csv(index=False).encode("utf-8")
            st.download_button("Download results (CSV)", data=csv, file_name="vn_candidates_near_low.csv", mime="text/csv")

st.caption("Disclaimer: Educational tool only. Not investment advice. Prices via vnstock; headlines via Google News RSS.")
