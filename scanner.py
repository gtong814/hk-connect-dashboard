import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# 熱門港股通清單 (可自行增減，或由 AkShare 自動擴充)
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

def calc_signals(df: pd.DataFrame):
    d = df.copy()
    for n, col in [(20, "MA1"), (50, "MA2"), (100, "MA3"),
                   (150, "C1A"), (200, "C1B"), (250, "C2A"), (300, "C2B")]:
        d[col] = d["Close"].ewm(span=n, adjust=False).mean()

    d["CloudTop"] = d[["C1A", "C1B", "C2A", "C2B"]].max(axis=1)
    d["CloudBot"] = d[["C1A", "C1B", "C2A", "C2B"]].min(axis=1)

    ma1_up = d["MA1"] > d["MA1"].shift(1)
    ma2_up = d["MA2"] > d["MA2"].shift(1)

    cross_up_12 = (d["MA1"] > d["MA2"]) & (d["MA1"].shift(1) <= d["MA2"].shift(1))
    cross_dn_12 = (d["MA1"] < d["MA2"]) & (d["MA1"].shift(1) >= d["MA2"].shift(1))
    cross_dn_c3 = (d["Close"] < d["MA3"]) & (d["Close"].shift(1) >= d["MA3"].shift(1))

    is_above = d["Close"] > d["CloudTop"]
    is_below = d["Close"] < d["CloudTop"]
    is_bull = (d["MA1"] > d["MA2"]) & (d["MA2"] > d["MA3"]) & ma1_up & ma2_up

    d["EarlyBuy"] = cross_up_12 & ma1_up & is_below
    d["TrimSell"] = cross_dn_12 & (~ma1_up)
    d["ExitSell"] = cross_dn_c3 & (~ma2_up)

    trend_buy = np.zeros(len(d), dtype=bool)
    in_bull = False
    for i in range(len(d)):
        if is_above.iloc[i] and is_bull.iloc[i] and not in_bull:
            trend_buy[i] = True
            in_bull = True
        elif d["ExitSell"].iloc[i] or d["Close"].iloc[i] < d["CloudBot"].iloc[i]:
            in_bull = False
    d["TrendBuy"] = trend_buy
    return d

def main():
    tickers = list(DEFAULT_GGT.keys())
    raw = yf.download(tickers, period="2y", group_by="ticker", auto_adjust=True, threads=True, progress=False)

    hkt_now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M HKT")
    rows = []

    for t in tickers:
        try:
            df = raw[t].dropna(subset=["Close"]) if isinstance(raw.columns, pd.MultiIndex) else raw.dropna(subset=["Close"])
            if len(df) < 100:
                continue
            d = calc_signals(df)
            last = d.iloc[-1]
            prev = d.iloc[-2]

            # 1. 即日收市訊號 (僅檢查最後一個交易日)
            today_sigs = []
            if last["TrendBuy"]: today_sigs.append("🟢 主升買進")
            if last["EarlyBuy"]: today_sigs.append("🔵 底部試單")
            if last["TrimSell"]: today_sigs.append("🟠 轉弱減碼")
            if last["ExitSell"]: today_sigs.append("🔴 破位賣出")

            # 2. 近 5 日訊號
            recent = d.tail(5)
            recent_sigs = []
            if recent["TrendBuy"].any(): recent_sigs.append("🟢 主升買進")
            if recent["EarlyBuy"].any(): recent_sigs.append("🔵 底部試單")
            if recent["TrimSell"].any(): recent_sigs.append("🟠 轉弱減碼")
            if recent["ExitSell"].any(): recent_sigs.append("🔴 破位賣出")

            pct_chg = (last["Close"] - prev["Close"]) / prev["Close"] * 100
            dist_cloud = (last["Close"] - last["CloudTop"]) / last["CloudTop"] * 100
            status = "多頭排列" if (last["Close"] > last["CloudTop"] and last["MA1"] > last["MA2"]) else \
                     "築底反彈" if (last["Close"] <= last["CloudTop"] and last["MA1"] > last["MA2"]) else "空頭/整理"

            rows.append({
                "代碼": t,
                "名稱": DEFAULT_GGT.get(t, t),
                "資料日期": d.index[-1].strftime("%Y-%m-%d"),
                "最新收盤": round(float(last["Close"]), 2),
                "單日漲跌%": round(float(pct_chg), 2),
                "距長期均線帶%": round(float(dist_cloud), 2),
                "目前結構": status,
                "即日收市訊號": " / ".join(today_sigs) if today_sigs else "-",
                "近5日訊號": " / ".join(recent_sigs) if recent_sigs else "-",
                "更新時間": hkt_now
            })
        except Exception as e:
            print(f"Error {t}: {e}")

    pd.DataFrame(rows).to_csv("daily_scan.csv", index=False, encoding="utf-8-sig")
    print(f"掃描完成，共 {len(rows)} 檔，更新時間：{hkt_now}")

if __name__ == "__main__":
    main()
