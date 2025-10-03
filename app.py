# app.py
# Vietnam Stock Pricing + News Analyzer (Streamlit + vnstock)
# -----------------------------------------------------------
# - Resolve ticker from Vietnamese/English company names
# - Price history via vnstock with source fallback (VCI → TCBS → SSI) + retries
# - Candlestick + SMA overlays, 1W/1M/3M returns
# - News via Google News RSS (URL-encoded to avoid InvalidURL)
# - Forecasts:
#    * Price-only: Holt-Winters, ARIMA(1,1,1), SMA, Naive
#    * News-aware (optional): ARIMAX (SARIMAX) with headline sentiment as exogenous input
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

# Numeric/Stats
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

# Optional (loaded lazily / guarded)
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

# vnstock unified API
from vnstock import Vnstock

st.set_page_config(page_title="Vietnam Stock + News + Forecast", layout="wide")


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
    Pull all symbols for lookup via vnstock Listing().
    """
    try:
        from vnstock import Listing
        listing = Listing()
        df = listing.all_symbols()
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
    Accepts: 'HPG', 'Hòa Phát', 'MSR', 'Masan', 'TV2', ...
    Returns: ticker code if found else None
    """
    s = (user_text or "").strip()
    if not s:
        return None

    # If user typed a likely ticker already (letters or digits, 2–6)
    if re.fullmatch(r"[A-Za-z0-9]{2,6}", s):   # <-- CHANGED
        return s.upper()

    df = get_all_symbols_df()
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
    """
    Return (DataFrame, error_str). On success, error_str=None.
    URL-encodes query to avoid InvalidURL.
    """
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
    """
    Return a DataFrame with index future business days and columns: yhat, lower, upper
    """
    horizon_days = int(horizon_days)
    future_idx = pd.bdate_range(s.index[-1] + pd.tseries.offsets.BDay(1), periods=horizon_days, freq="B")
    vol = _recent_volatility(s)
    last = float(s.iloc[-1])

    if method == "Naive (last value)":
        yhat = pd.Series(last, index=future_idx)
        steps = np.arange(1, horizon_days + 1)
        band = 1.65 * last * vol * np.sqrt(steps)
        lower = yhat.values - band
        upper = yhat.values + band
        return pd.DataFrame({"yhat": yhat, "lower": lower, "upper": upper}, index=future_idx)

    if method == "SMA (window=20)":
        sma = s.rolling(20).mean().iloc[-1]
        yhat = pd.Series(float(sma), index=future_idx)
        steps = np.arange(1, horizon_days + 1)
        band = 1.65 * float(s.iloc[-1]) * vol * np.sqrt(steps)
        lower = yhat.values - band
        upper = yhat.values + band
        return pd.DataFrame({"yhat": yhat, "lower": lower, "upper": upper}, index=future_idx)

    if method == "Holt-Winters":
        try:
            model = ExponentialSmoothing(s, trend="add", seasonal=None, initialization_method="estimated")
            fit = model.fit(optimized=True)
            fc = fit.forecast(horizon_days)
            resid = s - fit.fittedvalues.reindex_like(s)
            sigma = float(resid.dropna().std())
            steps = np.arange(1, horizon_days + 1)
            band = 1.65 * sigma * np.sqrt(steps)
            lower = fc.values - band
            upper = fc.values + band
            return pd.DataFrame({"yhat": fc, "lower": lower, "upper": upper}, index=future_idx)
        except Exception:
            pass

    # ARIMA(1,1,1)
    try:
        model = ARIMA(s, order=(1, 1, 1))
        fit = model.fit()
        pred = fit.get_forecast(steps=horizon_days)
        mean = pred.predicted_mean
        conf = pred.conf_int(alpha=0.10)  # ~90%
        lower = conf.iloc[:, 0]
        upper = conf.iloc[:, 1]
        return pd.DataFrame({"yhat": mean, "lower": lower, "upper": upper}, index=future_idx)
    except Exception:
        yhat = pd.Series(last, index=future_idx)
        steps = np.arange(1, horizon_days + 1)
        band = 1.65 * last * vol * np.sqrt(steps)
        lower = yhat.values - band
        upper = yhat.values + band
        return pd.DataFrame({"yhat": yhat, "lower": lower, "upper": upper}, index=future_idx)

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
    """
    Lazy-load multilingual sentiment if transformers is available.
    Returns a pipeline or None.
    """
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
    """
    Returns a DataFrame with columns: published_date (date), sent in [-1,1].
    """
    if news_df is None or news_df.empty:
        return pd.DataFrame(columns=["published_date", "sent"])

    pipe = _load_sentiment_pipeline()
    if pipe is None:
        # Transformers not available; return zeros to keep pipeline flowing
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
    """
    Align daily mean sentiment to the price index dates; forward-fill a few days (weekend news).
    """
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
    """
    Train SARIMAX (ARIMA) with exogenous daily sentiment.
    Returns DataFrame with yhat, lower, upper indexed by future business days.
    """
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
    future_exog = pd.Series(0.0, index=future_idx)  # neutral future news assumption

    pred = fit.get_forecast(steps=horizon_days, exog=future_exog.values.reshape(-1, 1))
    mean = pred.predicted_mean
    conf = pred.conf_int(alpha=0.10)  # ~90%
    lower = conf.iloc[:, 0]
    upper = conf.iloc[:, 1]
    out = pd.DataFrame({"yhat": mean, "lower": lower, "upper": upper}, index=future_idx)
    return out


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
    user_input.strip().upper() if re.fullmatch(r"[A-Za-z0-9]{2,6}", user_input.strip()) else None  # <-- CHANGED
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

# News-aware Forecast
st.subheader("🧠 News-aware forecast (experimental)")
with st.expander("Use headlines sentiment as an exogenous feature"):
    st.caption("We convert recent Vietnamese headlines into a daily sentiment score and feed it into an ARIMAX model.")
    horizon_news = st.slider("Horizon (business days)", 5, 60, 20, step=5)
    run_news_forecast = st.checkbox("Enable news-aware forecast", value=False,
                                    help="Requires 'transformers' & 'torch' on first use (model download).")

    if run_news_forecast:
        # Prepare series aligned to business days
        s_close2 = hist["close"].dropna().copy()
        s_close2.index = pd.to_datetime(s_close2.index)
        s_close2 = s_close2.asfreq("B").ffill()

        sent_series = _build_daily_sentiment_series(news_df, s_close2.index)

        try:
            fc_news = make_arimax_with_sentiment(s_close2, sent_series, horizon_days=horizon_news)
            plot_forecast(s_close2, fc_news, resolved, label_suffix=" (news-aware)")
            # Transparency: show recent sentiment
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

st.caption("Disclaimer: Data is for research/education only. Not investment advice. Prices via vnstock; headlines via Google News RSS.")
