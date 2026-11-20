import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="港股通均線帶掃描儀", layout="wide")
st.title("📊 港股通多週期均線帶 Dashboard")

# ==========================================
# 1. 讀取每日 16:10 自動更新的掃描結果
# ==========================================
if os.path.exists("daily_scan.csv"):
    res_df = pd.read_csv("daily_scan.csv")
    last_update = res_df["更新時間"].iloc[0] if "更新時間" in res_df.columns and not res_df.empty else "未知"
    trade_date = res_df["資料日期"].iloc[0] if "資料日期" in res_df.columns and not res_df.empty else "-"
    st.caption(f"🕒 排程自動更新時間：**{last_update}** ｜ 最新交易日：**{trade_date}**（每日 16:10 HKT 自動更新）")
else:
    st.warning("尚未產生 `daily_scan.csv`，請先至 GitHub Actions 手動執行一次排程，或等待 16:10 自動更新。")
    st.stop()

# ==========================================
# 2. 頂部指標卡 + 🔥 即日收市訊號專區
# ==========================================
today_df = res_df[res_df["即日收市訊號"] != "-"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("掃描標的數", len(res_df))
m2.metric("🔥 即日收市觸發", len(today_df))
m3.metric("近5日主升買進", int(res_df["近5日訊號"].str.contains("主升買進", na=False).sum()))
m4.metric("多頭排列家數", int((res_df["目前結構"] == "多頭排列").sum()))

st.subheader(f"🔥 即日收市觸發訊號 ({trade_date})")
if not today_df.empty:
    st.dataframe(
        today_df[["代碼", "名稱", "最新收盤", "單日漲跌%", "目前結構", "即日收市訊號"]],
        use_container_width=True, hide_index=True
    )
else:
    st.info("今日收市暫無個股觸發新買賣訊號。")

# ==========================================
# 3. 港股通完整掃描總表
# ==========================================
st.subheader("📋 港股通掃描總表")
filter_sig = st.multiselect(
    "篩選訊號或結構",
    ["🟢 主升買進", "🔵 底部試單", "🟠 轉弱減碼", "🔴 破位賣出", "多頭排列", "築底反彈"]
)

display_df = res_df.drop(columns=["更新時間"], errors="ignore").copy()
if filter_sig:
    mask = display_df["即日收市訊號"].apply(lambda x: any(s in str(x) for s in filter_sig)) | \
           display_df["近5日訊號"].apply(lambda x: any(s in str(x) for s in filter_sig)) | \
           display_df["目前結構"].isin(filter_sig)
    display_df = display_df[mask]

st.dataframe(display_df.sort_values("單日漲跌%", ascending=False), use_container_width=True, hide_index=True)

# ==========================================
# 4. 個股互動式均線帶 K 線圖
# ==========================================
@st.cache_data(ttl=1800)
def load_single_stock(ticker: str):
    df = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close"])

st.subheader("📈 個股均線帶與訊號圖表")
stock_options = dict(zip(res_df["代碼"], res_df["名稱"]))
selected_ticker = st.selectbox(
    "選擇要檢視的股票",
    options=list(stock_options.keys()),
    format_func=lambda x: f"{x} - {stock_options[x]}"
)

df_chart = load_single_stock(selected_ticker)
for n, col in [(20, "MA1"), (50, "MA2"), (100, "MA3"),
               (150, "C1A"), (200, "C1B"), (250, "C2A"), (300, "C2B")]:
    df_chart[col] = df_chart["Close"].ewm(span=n, adjust=False).mean()

chart_df = df_chart.tail(250)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.78, 0.22])
fig.add_trace(go.Candlestick(
    x=chart_df.index, open=chart_df["Open"], high=chart_df["High"],
    low=chart_df["Low"], close=chart_df["Close"], name="K線",
    increasing_line_color="#26A69A", decreasing_line_color="#EF5350"
), row=1, col=1)

# 長期均線帶
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["C1A"], line=dict(color="rgba(41,98,255,0.5)", width=1), name="EMA150"), row=1, col=1)
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["C1B"], line=dict(color="rgba(255,152,0,0.5)", width=1), fill="tonexty", fillcolor="rgba(41,98,255,0.12)", name="EMA200"), row=1, col=1)
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["C2A"], line=dict(color="rgba(76,175,80,0.5)", width=1), name="EMA250"), row=1, col=1)
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["C2B"], line=dict(color="rgba(244,67,54,0.5)", width=1), fill="tonexty", fillcolor="rgba(76,175,80,0.12)", name="EMA300"), row=1, col=1)

# 短中慢均線
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MA1"], line=dict(color="#00E676", width=2), name="EMA20"), row=1, col=1)
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MA2"], line=dict(color="#FF9800", width=2), name="EMA50"), row=1, col=1)
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MA3"], line=dict(color="#FF5252", width=2), name="EMA100"), row=1, col=1)

# 成交量
vol_colors = ["#26A69A" if c >= o else "#EF5350" for c, o in zip(chart_df["Close"], chart_df["Open"])]
fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["Volume"], marker_color=vol_colors, name="成交量"), row=2, col=1)

fig.update_layout(height=650, xaxis_rangeslider_visible=False, template="plotly_dark", hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)
