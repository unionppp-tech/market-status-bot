import os
import datetime
import time
import requests
import numpy as np
import pandas as pd
import pyupbit

# =========================
# 환경 변수 / 설정
# =========================
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

COINS = ["KRW-BTC", "KRW-ETH", "KRW-XRP"]

# 미국 혁신 섹터 (섹터당 대표주 최대 2 + 섹터 ETF). 순서 유지.
US_SECTORS = {
    "지수·레버리지": ["QQQ", "QLD", "TQQQ"],
    "반도체": ["NVDA", "AVGO", "SMH"],
    "양자컴퓨팅": ["IONQ", "RGTI", "QTUM"],
    "양자통신·보안": ["ARQQ", "LAES"],
    "AI·데이터": ["PLTR", "AI", "BOTZ"],
    "사이버보안": ["CRWD", "PANW", "CIBR"],
    "우주·방산": ["RKLB", "LMT", "ARKX"],
    "전기차·자율주행": ["TSLA", "RIVN"],
    "의약·바이오": ["LLY", "NVO", "XBI"],
    "유전자·유전체": ["CRSP", "ARKG"],
}

# 리포트에서 ·ETF 로 표기할 티커(섹터 ETF)
ETF_TICKERS = {"SMH", "QTUM", "BOTZ", "CIBR", "ARKX", "XBI", "ARKG"}

# 지표 파라미터 (TradingView 설정과 동일)
LENGTH_BB = 20
LENGTH_KC = 20
MULT_KC = 1.5           # 원본 Pine: BB 편차도 이 값을 사용(mult 2.0은 미사용)
USE_TRUE_RANGE = True


# =========================
# 디스코드 메시지 전송
# =========================
def send_message(msg: str):
    now = datetime.datetime.now()
    payload = {"content": f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"}
    if not DISCORD_WEBHOOK_URL:
        print("[WARN] DISCORD_WEBHOOK_URL 미설정 — 콘솔 출력만 합니다.")
        print(payload["content"])
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
    except Exception as e:
        print(f"Discord send failed: {e}")
    print(payload["content"])


# =========================
# 선형회귀 (Pine ta.linreg(src, length, 0) 재현)
# =========================
def rolling_linreg(series: pd.Series, length: int) -> pd.Series:
    """각 봉에서 최근 length개 값에 최소자승 직선을 적합, 현재 봉의 회귀값 반환.
    x=0..length-1(0=과거), 결과 = intercept + slope*(length-1)."""
    values = series.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan)
    x = np.arange(length, dtype=float)
    for i in range(length - 1, n):
        window = values[i - length + 1 : i + 1]
        if np.isnan(window).any():
            continue
        slope, intercept = np.polyfit(x, window, 1)
        out[i] = intercept + slope * (length - 1)
    return pd.Series(out, index=series.index)


# =========================
# Squeeze Momentum 계산 (원본 Pine과 1:1 일치) + 포지션 시뮬레이션
# =========================
def squeeze_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    source = df["close"]

    # Bollinger Bands — ⚠️ 원본은 dev에 multKC(1.5) 사용
    basis = source.rolling(LENGTH_BB).mean()
    dev = MULT_KC * source.rolling(LENGTH_BB).std(ddof=0)  # 모집단 표준편차(ddof=0)
    df["upper_bb"] = basis + dev
    df["lower_bb"] = basis - dev

    # Keltner Channel
    ma = source.rolling(LENGTH_KC).mean()
    if USE_TRUE_RANGE:
        price_range = pd.concat(
            [
                (df["high"] - df["low"]).abs(),
                (df["high"] - df["close"].shift(1)).abs(),
                (df["low"] - df["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
    else:
        price_range = df["high"] - df["low"]
    range_ma = price_range.rolling(LENGTH_KC).mean()
    df["upper_kc"] = ma + range_ma * MULT_KC
    df["lower_kc"] = ma - range_ma * MULT_KC

    # Squeeze 상태
    df["sqz_on"] = (df["lower_bb"] > df["lower_kc"]) & (df["upper_bb"] < df["upper_kc"])
    df["sqz_off"] = (df["lower_bb"] < df["lower_kc"]) & (df["upper_bb"] > df["upper_kc"])
    df["no_sqz"] = (~df["sqz_on"]) & (~df["sqz_off"])

    # Momentum val (원본: linreg)
    highest = df["high"].rolling(LENGTH_KC).max()
    lowest = df["low"].rolling(LENGTH_KC).min()
    sma_c = source.rolling(LENGTH_KC).mean()
    src = source - ((highest + lowest) / 2 + sma_c) / 2
    df["val"] = rolling_linreg(src, LENGTH_KC)

    # 신호 판정 (원본 로직)
    cond1 = (~df["no_sqz"]) & (~df["sqz_on"])   # = sqz_off (스퀴즈 해제)
    cond2 = df["val"] > 0
    check1 = cond1.astype(int)
    check2 = cond2.astype(int)

    is_vola_start = check1.diff().fillna(0) == 1     # 스퀴즈 발화(십자가 불)
    is_mom_up = check2.diff().fillna(0) == 1         # val 음수→양수 전환
    is_mom_down = check2.diff().fillna(0) == -1      # val 양수→음수 전환

    # 매수: (발화 & val>0) 또는 (스퀴즈 해제 상태에서 val 양전) — 사용자 정의
    df["is_long"] = (is_vola_start & cond2) | (is_mom_up & cond1)
    # 매도(청산): val이 음수로 하향 전환되는 순간
    df["is_sell"] = is_mom_down

    # 포지션 시뮬레이션: 매수 진입, 매도 청산. pos=1 보유 / 0 현금
    pos = np.zeros(len(df), dtype=int)
    cur = 0
    il = df["is_long"].to_numpy()
    isl = df["is_sell"].to_numpy()
    for i in range(len(df)):
        if il[i]:
            cur = 1
        if isl[i]:
            cur = 0
        pos[i] = cur
    df["pos"] = pos
    return df


# =========================
# 한 자산의 현재 상태(판정봉) 추출
# =========================
def evaluate_asset(asset_name: str, df: pd.DataFrame, use_last_closed: bool):
    calc = squeeze_momentum(df)
    if len(calc) < LENGTH_KC + 3:
        return None
    idx = -2 if use_last_closed else -1
    row = calc.iloc[idx]
    prev = calc.iloc[idx - 1]
    if pd.isna(row["val"]) or pd.isna(prev["val"]):
        return None

    events = []
    if bool(row["is_long"]):
        events.append("진입")
    if bool(row["is_sell"]):
        events.append("청산")

    if int(row["pos"]) == 1:
        state = "buy"     # 전략상 보유(매수 유지)
    elif row["val"] < 0:
        state = "sell"    # 미보유 + 약세
    else:
        state = "wait"    # 미보유 + 중립

    sqz_state = "sqzOff" if row["sqz_off"] else ("sqzOn" if row["sqz_on"] else "noSqz")

    return {
        "asset": asset_name,
        "state": state,
        "events": events,
        "val": float(row["val"]),
        "dir": "▲" if row["val"] > prev["val"] else ("▼" if row["val"] < prev["val"] else "–"),
        "close": float(row["close"]),
        "sqz": sqz_state,
        "date": str(row["time"])[:10] if "time" in calc.columns else "",
    }


def format_state_line(info: dict) -> str:
    label = {"buy": "🟢 매수·보유", "sell": "🔴 매도·관망", "wait": "🟡 관망"}[info["state"]]
    val = info["val"]
    val_str = f"{val:+,.0f}" if abs(val) >= 1000 else f"{val:+.2f}"
    close = f"{info['close']:,.0f}" if info["close"] >= 1000 else f"{info['close']:,.2f}"
    ev = f"  ⚡오늘:{'·'.join(info['events'])}" if info["events"] else ""
    name = info["asset"] + ("·ETF" if info["asset"] in ETF_TICKERS else "")
    return (
        f"{label}  {name} | {info['date']} | "
        f"val {val_str}{info['dir']} | {info['sqz']} | 종가 {close}{ev}"
    )


# =========================
# 데이터 조회
# =========================
YAHOO_HOSTS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")


def _yahoo_chart_json(ticker: str):
    """야후 chart API에서 일봉 원시 JSON을 받는다(호스트 1회 폴백)."""
    now = datetime.datetime.now(datetime.timezone.utc)
    period1 = int((now - datetime.timedelta(days=400)).timestamp())
    period2 = int(now.timestamp())
    params = f"?period1={period1}&period2={period2}&interval=1d"
    headers = {"User-Agent": "Mozilla/5.0"}
    last_err = None
    for host in YAHOO_HOSTS:
        url = f"https://{host}/v8/finance/chart/{ticker}{params}"
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                return r.json()
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = str(e)
    raise RuntimeError(last_err or "요청 실패")


def get_us_stock_df(ticker: str):
    """야후 파이낸스 chart API로 배당·분할 조정 OHLC 일봉을 만든다.

    (기존 Stooq/FinanceDataReader가 anti-bot 차단으로 CSV 대신 검증 페이지를 반환 → 교체.)
    조정가 = raw × (adjclose/close), close = adjclose 로 TradingView ADJ(수정주가)와 정합.
    """
    j = _yahoo_chart_json(ticker)
    res = (j.get("chart", {}).get("result") or [None])[0]
    if not res:
        return None
    ts = res.get("timestamp") or []
    ind = res.get("indicators", {})
    quote = (ind.get("quote") or [{}])[0]
    adj = ((ind.get("adjclose") or [{}])[0]).get("adjclose") or []
    o_ = quote.get("open") or []
    h_ = quote.get("high") or []
    l_ = quote.get("low") or []
    c_ = quote.get("close") or []
    rows = []
    for i in range(len(ts)):
        if i >= len(o_) or i >= len(h_) or i >= len(l_) or i >= len(c_) or i >= len(adj):
            continue
        o, h, l, c, ac = o_[i], h_[i], l_[i], c_[i], adj[i]
        if None in (o, h, l, c, ac) or c == 0:
            continue
        f = ac / c  # 조정 계수
        rows.append((ts[i], o * f, h * f, l * f, ac))
    if len(rows) < LENGTH_KC + 3:
        return None
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])
    df = df.astype({"open": float, "high": float, "low": float, "close": float})
    # 미완성(당일) 봉 제외 — 21시 KST(=미국 프리마켓)엔 당일 정규장 미마감.
    # UTC 오늘 날짜 이상인 봉을 버려 '마지막 완성 봉'만 남긴다(process_orders_on_close 일치).
    today_utc = datetime.datetime.now(datetime.timezone.utc).date()
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[df["time"].dt.date < today_utc].reset_index(drop=True)
    return df


def get_coin_df(coin: str):
    df = pyupbit.get_ohlcv(coin, interval="day", count=200)
    if df is None or len(df) < LENGTH_KC + 3:
        return None
    df = df.rename(columns=str.lower).reset_index()
    df.rename(columns={"index": "time"}, inplace=True)
    return df


# =========================
# 전 종목 상태 수집
# =========================
def collect_report():
    """반환: blocks = [(섹터명, [상태라인, ...]), ...], errors = [..]"""
    blocks, errors = [], []

    # 코인: 마지막 완성 봉(iloc[-2]) 기준
    coin_lines = []
    for coin in COINS:
        try:
            df = get_coin_df(coin)
            if df is None:
                errors.append(f"❌ {coin} 데이터 수집 실패"); continue
            info = evaluate_asset(coin, df, use_last_closed=True)
            if info:
                coin_lines.append(format_state_line(info))
            time.sleep(0.2)
        except Exception as e:
            errors.append(f"❌ {coin} 처리 오류: {e}")
    if coin_lines:
        blocks.append(("코인", coin_lines))

    # 미국주식: 섹터별, 최신 완성 봉(iloc[-1]) 기준
    for sector, tickers in US_SECTORS.items():
        lines = []
        for t in tickers:
            try:
                time.sleep(0.5)
                df = get_us_stock_df(t)
                if df is None:
                    errors.append(f"❌ {t} 데이터 수집 실패"); continue
                info = evaluate_asset(t, df, use_last_closed=False)
                if info:
                    lines.append(format_state_line(info))
            except Exception as e:
                errors.append(f"❌ {t} 처리 오류: {e}")
        if lines:
            blocks.append((sector, lines))

    return blocks, errors


LEGEND = "범례 🟢보유 🔴약세 🟡관망 · ▲▼모멘텀 · sqzOn/Off · ⚡진입/청산"
MAX_LEN = 1800  # 디스코드 2000자 - 타임스탬프/여유


def send_report(blocks, errors):
    title = "📊 **스퀴즈 모멘텀 상태 리포트**\n" + LEGEND
    segments = [f"\n\n**[{s}]**\n" + "\n".join(lines) for s, lines in blocks]
    if errors:
        segments.append("\n\n" + "\n".join(errors))

    chunks, cur = [], title
    for seg in segments:
        if len(cur) + len(seg) > MAX_LEN and cur:
            chunks.append(cur)
            cur = seg.lstrip("\n")
        else:
            cur += seg
    if cur:
        chunks.append(cur)

    for i, msg in enumerate(chunks):
        send_message(msg)
        if i < len(chunks) - 1:
            time.sleep(0.5)


def main():
    blocks, errors = collect_report()
    send_report(blocks, errors)


if __name__ == "__main__":
    main()
