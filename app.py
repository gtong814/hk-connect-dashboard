import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="港股通均線帶掃描儀", layout="wide")

# ==========================================
# 1. 內建熱門港股通名單 (避免雲端 IP 遭限流時的備援)
# ==========================================
DEFAULT_GGT = {
    "0700.HK": "騰訊控股", "9988.HK": "阿里巴巴-W", "3690.HK": "美團-W",
    "1810.HK": "小米集團-W", "0941.HK": "中國移動", "0883.HK": "中國海洋石油",
    "0939.HK": "建設銀行", "1398.HK": "工商銀行", "3988.HK": "中國銀行",
    "2318.HK": "中國平安", "1211.HK": "比亞迪股份", "0981.HK": "中芯國際",
    "1024.HK": "快手-W", "9868.HK": "小鵬汽車-W", "2015.HK": "理想汽車-W",
    "0388.HK": "香港交易所", "1299.HK": "友邦保險", "0005.HK": "匯豐控股",
    "2800.HK": "盈富基金", "3033.HK": "南方恒生科技", "9660.HK": "地平線機器人-W",
    "1686.HK": "新意網集團", "1133.HK": "哈爾濱電氣", "2858.HK": "易鑫集團"
}

def format_hk_ticker(code: str) -> str:
    """將 5 碼港股代碼 (如 01810, 00700) 轉換為 Yahoo Finance 格式 (1810.HK, 0700.HK)"""
    clean = str(code).strip().upper().replace(".HK", "")
    if clean.isdigit():
        return clean.lstrip("0").zfill(4) + ".HK"
    return f"{clean}.HK"

@st.cache_data(ttl=86400)
def fetch_akshare_ggt_list():
    """透過 AkShare 抓取最新港股通成份股名單"""
    try:
        import akshare as ak
        df = ak.stock_hk_ggt_components_em()
        mapping = {}
        for _, row in df.iterrows():
            ticker = format_hk_ticker(row["代码"])
            mapping[ticker] = str(row["名称"])
        return mapping
    except Exception:
        return None

@st.cache_data(ttl=1800)
def download_stock_data(tickers: list, period: str = "2y"):
    """批次下載港股歷史日線資料"""
    if not tickers:
        return {}
    raw = yf.download(
        tickers,
        period=period,
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False
    )
    data_dict = {}
    if len(tickers) == 1:
        t = tickers[0]
        df = raw[t] if isinstance(raw.columns, pd.MultiIndex) and t in raw.columns.levels[0] else raw
        df = df.dropna(subset=["Close"])
        if not df.empty:
            data_dict[t] = df
    else:
        for t in tickers:
            if isinstance(raw.columns, pd.MultiIndex) and t in raw.columns.levels[0]:
                df = raw[t].dropna(subset=["Close"])
                if not df.empty:
                    data_dict[t] = df
    return data_dict

# ==========================================
# 2. 多週期均線帶與買賣訊號計算 (對齊 Pine Script 邏輯)
# ==========================================
def calc_indicators(df: pd.DataFrame, ma_type: str, l1, l2, l3, c1a, c1b, c2a, c2b):
    d = df.copy()
    def get_ma(s, n):
        return s.ewm(span=n, adjust=False).mean() if ma_type == "EMA" else s.rolling(n).mean()

    d["MA1"] = get_ma(d["Close"], l1)
    d["MA2"] = get_ma(d["Close"], l2)
    d["MA3"] = get_ma(d["Close"], l3)
    d["C1A"] = get_ma(d["Close"], c1a)
    d["C1B"] = get_ma(d["Close"], c1b)
    d["C2A"] = get_ma(d["Close"], c2a)
    d["C2B"] = get_ma(d["Close"], c2b)

    d["CloudTop"] = d[["C1A", "C1B", "C2A", "C2B"]].max(axis=1)
    d["CloudBot"] = d[["C1A", "C1B", "C2A", "C2B"]].min(axis=1)

    ma1_up = d["MA1"] > d["MA1"].shift(1)
    ma2_up = d["MA2"] > d["MA2"].shift(1)

    # 交叉判斷
    cross_up_12 = (d["MA1"] > d["MA2"]) & (d["MA1"].shift(1) <= d["MA2"].shift(1))
    cross_dn_12 = (d["MA1"] < d["MA2"]) & (d["MA1"].shift(1) >= d["MA2"].shift(1))
    cross_dn_c3 = (d["Close"] < d["MA3"]) & (d["Close"].shift(1) >= d["MA3"].shift(1))

    is_above_clouds = d["Close"] > d["CloudTop"]
    is_below_clouds = d["Close"] < d["CloudTop"]
    is_bull_align = (d["MA1"] > d["MA2"]) & (d["MA2"] > d["MA3"]) & ma1_up & ma2_up

    d["EarlyBuy"] = cross_up_12 & ma1_up & is_below_clouds
    d["TrimSell"] = cross_dn_12 & (~ma1_up)
    d["ExitSell"] = cross_dn_c3 & (~ma2_up)

    # 狀態機判斷主升買點 (避免連續觸發)
    trend_buy = np.zeros(len(d), dtype=bool)
    in_bull = False
    for i in range(len(d)):
        if is_above_clouds.iloc[i] and is_bull_align.iloc[i] and not in_bull:
            trend_buy[i] = True
            in_bull = True
        elif d["ExitSell"].iloc[i] or d["Close"].iloc[i] < d["CloudBot"].iloc[i]:
            in_bull = False
    d["TrendBuy"] = trend_buy

    return d

# ==========================================
# 3. 側邊欄參數設定
# ==========================================
st.title("📊 港股通多週期均線帶 Dashboard")

with st.sidebar:
    st.header("1. 股票池設定")
    pool_mode = st.radio(
        "選擇名單來源",
        ["熱門港股通精選 (快速)", "AkShare 港股通名單", "自訂輸入代碼"]
    )

    stock_map = DEFAULT_GGT.copy()
    if pool_mode == "AkShare 港股通名單":
        limit_n = st.slider("掃描前 N 檔 (避免請求超時)", 20, 200, 50, step=10)
        ak_map = fetch_akshare_ggt_list()
        if ak_map:
            stock_map = dict(list(ak_map.items())[:limit_n])
            st.success(f"已載入 {len(stock_map)} 檔港股通標的")
        else:
            st.warning("AkShare 連線失敗，已自動切換回內建精選名單。")
    elif pool_mode == "自訂輸入代碼":
        custom_input = st.text_area("輸入代碼 (逗號分隔)", "1810, 0700, 9988, 3690, 0981")
        tickers_raw = [format_hk_ticker(x) for x in custom_input.split(",") if x.strip()]
        stock_map = {t: DEFAULT_GGT.get(t, t) for t in tickers_raw}

    st.header("2. 均線參數")
    ma_type = st.selectbox("均線類型", ["EMA", "SMA"], index=0)
    col1, col2, col3 = st.columns(3)
    l1 = col1.number_input("快線", 5, 100, 20)
    l2 = col2.number_input("中線", 10, 200, 50)
    l3 = col3.number_input("慢線", 20, 300, 100)

    c1a, c1b = 150, 200
    c2a, c2b = 250, 300
    lookback_days = st.slider("訊號回溯天數 (顯示近 N 日觸發)", 1, 15, 5)

# ==========================================
# 4. 執行掃描與彙整表格
# ==========================================
with st.spinner("正在下載港股通最新日線資料..."):
    data_dict = download_stock_data(list(stock_map.keys()), period="2y")

rows = []
calc_dict = {}

for ticker, df in data_dict.items():
    if len(df) < 100:
        continue
    d = calc_indicators(df, ma_type, l1, l2, l3, c1a, c1b, c2a, c2b)
    calc_dict[ticker] = d

    last = d.iloc[-1]
    prev = d.iloc[-2]
    pct_chg = (last["Close"] - prev["Close"]) / prev["Close"] * 100
    dist_cloud = (last["Close"] - last["CloudTop"]) / last["CloudTop"] * 100

    # 檢查近 N 日訊號
    recent = d.tail(lookback_days)
    signals = []
    if recent["TrendBuy"].any(): signals.append("🟢 主升買進")
    if recent["EarlyBuy"].any(): signals.append("🔵 底部試單")
    if recent["TrimSell"].any(): signals.append("🟠 轉弱減碼")
    if recent["ExitSell"].any(): signals.append("🔴 破位賣出")

    trend_status = "多頭排列" if (last["Close"] > last["CloudTop"] and last["MA1"] > last["MA2"]) else \
                   "築底反彈" if (last["Close"] <= last["CloudTop"] and last["MA1"] > last["MA2"]) else "空頭/整理"

    rows.append({
        "代碼": ticker,
        "名稱": stock_map.get(ticker, ticker),
        "最新收盤": round(float(last["Close"]), 2),
        "單日漲跌%": round(float(pct_chg), 2),
        "距長期均線帶%": round(float(dist_cloud), 2),
        "目前結構": trend_status,
        f"近{lookback_days}日訊號": " / ".join(signals) if signals else "-"
    })

res_df = pd.DataFrame(rows)

if res_df.empty:
    st.error("查無資料，請確認股票代碼或稍後重試。")
    st.stop()

# 頂部指標卡
m1, m2, m3, m4 = st.columns(4)
m1.metric("掃描標的數", len(res_df))
m2.metric("主升買進訊號", int(res_df[f"近{lookback_days}日訊號"].str.contains("主升買進").sum()))
m3.metric("底部試單訊號", int(res_df[f"近{lookback_days}日訊號"].str.contains("底部試單").sum()))
m4.metric("多頭排列家數", int((res_df["目前結構"] == "多頭排列").sum()))

# 篩選器
filter_sig = st.multiselect(
    "篩選訊號或結構",
    ["🟢 主升買進", "🔵 底部試單", "🟠 轉弱減碼", "🔴 破位賣出", "多頭排列", "築底反彈"]
)
display_df = res_df.copy()
if filter_sig:
    mask = display_df[f"近{lookback_days}日訊號"].apply(lambda x: any(s in x for s in filter_sig)) | \
           display_df["目前結構"].isin(filter_sig)
    display_df = display_df[mask]

st.subheader("📋 港股通掃描總表")
st.dataframe(display_df.sort_values("單日漲跌%", ascending=False), use_container_width=True, hide_index=True)

# ==========================================
# 5. 個股互動式均線帶 K 線圖
# ==========================================
st.subheader("📈 個股均線帶與訊號圖表")
selected_ticker = st.selectbox(
    "選擇要檢視的股票",
    options=list(calc_dict.keys()),
    format_func=lambda x: f"{x} - {stock_map.get(x, '')}"
)

chart_df = calc_dict[selected_ticker].tail(250)  # 顯示近一年日線

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    vertical_spacing=0.04, row_heights=[0.78, 0.22]
)

# K 線
fig.add_trace(go.Candlestick(
    x=chart_df.index, open=chart_df["Open"], high=chart_df["High"],
    low=chart_df["Low"], close=chart_df["Close"], name="K線",
    increasing_line_color="#26A69A", decreasing_line_color="#EF5350"
), row=1, col=1)

# 藍橘均線帶 (150-200)
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["C1A"], line=dict(color="rgba(41,98,255,0.5)", width=1), name="EMA150"), row=1, col=1)
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["C1B"], line=dict(color="rgba(255,152,0,0.5)", width=1), fill="tonexty", fillcolor="rgba(41,98,255,0.12)", name="EMA200"), row=1, col=1)

# 紅綠均線帶 (250-300)
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["C2A"], line=dict(color="rgba(76,175,80,0.5)", width=1), name="EMA250"), row=1, col=1)
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["C2B"], line=dict(color="rgba(244,67,54,0.5)", width=1), fill="tonexty", fillcolor="rgba(76,175,80,0.12)", name="EMA300"), row=1, col=1)

# 短中慢均線
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MA1"], line=dict(color="#00E676", width=2), name=f"{ma_type}{l1}"), row=1, col=1)
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MA2"], line=dict(color="#FF9800", width=2), name=f"{ma_type}{l2}"), row=1, col=1)
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MA3"], line=dict(color="#FF5252", width=2), name=f"{ma_type}{l3}"), row=1, col=1)

# 買賣訊號標記
for col_name, label, color, symbol, y_col, mult in [
    ("EarlyBuy", "試單", "#00B0FF", "triangle-up", "Low", 0.97),
    ("TrendBuy", "買進", "#00C853", "triangle-up", "Low", 0.95),
    ("TrimSell", "減碼", "#FF9800", "triangle-down", "High", 1.03),
    ("ExitSell", "賣出", "#D50000", "triangle-down", "High", 1.05),
]:
    sig_pts = chart_df[chart_df[col_name]]
    if not sig_pts.empty:
        fig.add_trace(go.Scatter(
            x=sig_pts.index, y=sig_pts[y_col] * mult, mode="markers+text",
            text=[label]*len(sig_pts), textposition="bottom center" if mult < 1 else "top center",
            marker=dict(color=color, size=10, symbol=symbol), name=label
        ), row=1, col=1)

# 成交量
vol_colors = ["#26A69A" if c >= o else "#EF5350" for c, o in zip(chart_df["Close"], chart_df["Open"])]
fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["Volume"], marker_color=vol_colors, name="成交量"), row=2, col=1)

fig.update_layout(
    height=680, xaxis_rangeslider_visible=False,
    template="plotly_dark", hovermode="x unified",
    margin=dict(l=10, r=10, t=30, b=10)
)
st.plotly_chart(fig, use_container_width=True)
