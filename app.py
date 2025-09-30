import sys, platform
import streamlit as st

st.set_page_config(page_title="Smoke Test", layout="centered")

st.title("Smoke Test")
st.write({
    "python": sys.version.split()[0],
    "platform": platform.platform(),
})

# Try importing vnstock and show its version / a quick call
try:
    import vnstock
    from vnstock import stock_historical_data
    st.success(f"vnstock imported: {vnstock.__version__}")
    df = stock_historical_data(symbol="HPG", start_date="2024-09-01", end_date="2024-10-01", resolution="1D")
    st.write(f"HPG rows: {len(df)}")
except Exception as e:
    st.error(f"vnstock import or call failed: {type(e).__name__}: {e}")
