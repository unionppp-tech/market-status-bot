import os
import json
import datetime
import time
import requests
import numpy as np
import pandas as pd
import pyupbit
import FinanceDataReader as fdr

# =========================
# 환경 변수 / 설정
# =========================
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

# 자산 리스트
COINS = ["KRW-BTC", "KRW-ETH", "KRW-XRP"]
US_STOCKS = ["QQQ", "QLD", "TQQQ", "TSLA", "NVDA"]

# 지표 파라미터 (TradingView 설정과 동일)
LENGTH_BB = 20
MULT_BB = 2.0     # 원본 코드는 실제로 이 값을 쓰지 않음(아래 dev 계산 참고)
LENGTH_KC = 20
MULT_KC = 1.5
USE_TRUE_RANGE = True

# 같은 봉을 반복 알림하지 않기 위한 상태 파일 (레포 루트/state.json)
STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "state.json"
)
# 신호가 없어도 "오늘 신호 없음" 하트비트를 보낼지 여부
SEND_HEARTBEAT = os.environ.get("SEND_HEARTBEAT", "0") == "1"


# =========================
# 디스코드 메시지 전송
# =========================
def send_message(msg: str):
    now = datetime.datetime.now()
    payload = {"content": f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"}
    if not DISCORD_WEBHOOK_URL:
        print("[WARN] DISCORD_WEBHOOK_URL 미설정 — 콘솔 출력만 합니다.")
        print(payload)
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
    except Exception as e:
        print(f"Discord send failed: {e}")
    print(payload)


# =========================
# 상태 파일 (중복 알림 방지)
# =========================
def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"state 저장 실패: {e}")


# =========================
# 선형회귀 (Pine ta.linreg(src, length, 0) 재현)
# =========================
def rolling_linreg(series: pd.Series, length: int) -> pd.Series:
    """각 봉에서 최근 `length`개 값에 최소자승 직선을 적합해 현재 봉의 회귀값을 반환.
    x = 0..length-1 (0=가장 과거, length-1=현재), 결과 = intercept + slope*(length-1).
    """
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
# Squeeze Momentum 계산 (원본 Pine과 1:1 일치)
# =========================
def squeeze_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    source = df["close"]

    # --- Bollinger Bands ---
    # ⚠️ 원본 Pine: dev = multKC * ta.stdev(...)  (mult(2.0)이 아니라 1.5를 사용)
    basis = source.rolling(LENGTH_BB).mean()
    dev = MULT_KC * source.rolling(LENGTH_BB).std(ddof=0)  # 모집단 표준편차(ddof=0)
    df["upper_bb"] = basis + dev
    df["lower_bb"] = basis - dev

    # --- Keltner Channel ---
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

    # --- Squeeze 상태 ---
    df["sqz_on"] = (df["lower_bb"] > df["lower_kc"]) & (df["upper_bb"] < df["upper_kc"])
    df["sqz_off"] = (df["lower_bb"] < df["lower_kc"]) & (df["upper_bb"] > df["upper_kc"])
    df["no_sqz"] = (~df["sqz_on"]) & (~df["sqz_off"])

    # --- Momentum val (원본: linreg) ---
    highest = df["high"].rolling(LENGTH_KC).max()
    lowest = df["low"].rolling(LENGTH_KC).min()
    sma_c = source.rolling(LENGTH_KC).mean()
    src = source - ((highest + lowest) / 2 + sma_c) / 2
    df["val"] = rolling_linreg(src, LENGTH_KC)

    # --- 신호 판정 (원본 로직) ---
    cond1 = (~df["no_sqz"]) & (~df["sqz_on"])   # = sqz_off (스퀴즈 해제 상태)
    cond2 = df["val"] > 0                        # 모멘텀 양수
    check1 = cond1.astype(int)
    check2 = cond2.astype(int)

    is_vola_start = check1.diff().fillna(0) == 1          # 스퀴즈 방금 풀림(0→1)
    is_mom_change = check2.diff().fillna(0) != 0          # 모멘텀 부호 전환
    is_mom_change2 = check2.diff().fillna(0) == 1         # 모멘텀 방금 양전(0→1)

    df["cond1"] = cond1
    df["cond2"] = cond2
    # isLong or isLong2
    df["is_long"] = (is_vola_start & cond2) | (is_mom_change2 & cond1)
    df["is_close"] = is_mom_change
    df["is_short"] = is_vola_start & (~cond2)
    return df


# =========================
# 한 자산의 마지막 완성 봉에서 신호 추출
# =========================
def evaluate_asset(asset_name: str, df: pd.DataFrame, use_last_closed: bool):
    """use_last_closed=True(코인): 마지막 행은 진행중 봉이므로 iloc[-2]로 판정.
       False(미국주식): 21시 KST엔 장 마감 상태라 iloc[-1]이 완성봉.
    반환: 신호 dict 또는 None
    """
    calc = squeeze_momentum(df)
    if len(calc) < LENGTH_KC + 2:
        return None

    idx = -2 if use_last_closed else -1
    row = calc.iloc[idx]
    if pd.isna(row["val"]):
        return None

    signals = []
    if bool(row["is_long"]):
        signals.append("buy")
    if bool(row["is_close"]):
        signals.append("close")
    if bool(row["is_short"]):
        signals.append("short")

    bar_date = str(row["time"])[:10] if "time" in calc.columns else ""

    if row["sqz_off"]:
        sqz_state = "sqzOff"
    elif row["sqz_on"]:
        sqz_state = "sqzOn"
    else:
        sqz_state = "noSqz"

    return {
        "asset": asset_name,
        "signals": signals,
        "close": float(row["close"]),
        "val": float(row["val"]),
        "sqz": sqz_state,
        "date": bar_date,
    }


def format_signal_line(info: dict) -> str:
    emoji = {"buy": "🟢 매수", "close": "🔴 청산", "short": "🔻 숏(참고)"}
    tags = " · ".join(emoji[s] for s in info["signals"])
    return (
        f"📅 {info['date']} | {info['asset']}  {tags}\n"
        f"   종가 {info['close']:.2f} · val {info['val']:+.2f} · 상태 {info['sqz']}"
    )


# =========================
# 데이터 조회
# =========================
def get_us_stock_df(ticker: str):
    start = (datetime.date.today() - datetime.timedelta(days=400)).strftime("%Y-%m-%d")
    df = fdr.DataReader(ticker, start)
    if df is None or df.empty:
        return None
    df = df[["Open", "Close", "High", "Low"]].reset_index()
    date_col = "Date" if "Date" in df.columns else "index"
    df.rename(
        columns={date_col: "time", "Open": "open", "Close": "close",
                 "High": "high", "Low": "low"},
        inplace=True,
    )
    df = df.astype({"open": float, "close": float, "high": float, "low": float})
    # 미완성(당일) 봉 제외 — 21시 KST(=미국 프리마켓)엔 당일 미국 정규장 미마감.
    # UTC 오늘 날짜 이상인 봉을 버려 '마지막 완성 봉'만 남긴다(process_orders_on_close 일치).
    today_utc = datetime.datetime.now(datetime.timezone.utc).date()
    df["time"] = pd.to_datetime(df["time"])
    df = df[df["time"].dt.date < today_utc].reset_index(drop=True)
    return df


def get_coin_df(coin: str):
    df = pyupbit.get_ohlcv(coin, interval="day", count=200)
    if df is None or len(df) < LENGTH_KC + 2:
        return None
    df = df.rename(columns=str.lower).reset_index()
    df.rename(columns={"index": "time"}, inplace=True)
    return df


# =========================
# 자산군별 처리
# =========================
def collect_signals():
    state = load_state()
    lines = []
    errors = []

    # 코인: 마지막 완성 봉(iloc[-2]) 기준
    for coin in COINS:
        try:
            df = get_coin_df(coin)
            if df is None:
                errors.append(f"❌ {coin} 데이터 수집 실패")
                continue
            info = evaluate_asset(coin, df, use_last_closed=True)
            _maybe_add(info, state, lines)
            time.sleep(0.2)
        except Exception as e:
            errors.append(f"❌ {coin} 처리 오류: {e}")

    # 미국주식: 최신 완성 봉(iloc[-1]) 기준
    for stock in US_STOCKS:
        try:
            time.sleep(1)
            df = get_us_stock_df(stock)
            if df is None:
                errors.append(f"❌ {stock} 데이터 수집 실패")
                continue
            info = evaluate_asset(stock, df, use_last_closed=False)
            _maybe_add(info, state, lines)
        except Exception as e:
            errors.append(f"❌ {stock} 처리 오류: {e}")

    save_state(state)
    return lines, errors


def _maybe_add(info, state, lines):
    if info is None:
        return
    asset = info["asset"]
    bar_date = info["date"]
    # 이미 알림한(=처리한) 봉이면 스킵. 새 봉일 때만 진행.
    if state.get(asset) == bar_date:
        return
    # 봉을 처리했다고 기록(신호가 없어도 기록해 중복 방지)
    state[asset] = bar_date
    if info["signals"]:
        lines.append(format_signal_line(info))


# =========================
# 메인 실행
# =========================
def main():
    lines, errors = collect_signals()

    if lines:
        header = "📊 스퀴즈 모멘텀 신호"
        send_message(header + "\n\n" + "\n\n".join(lines))
    elif SEND_HEARTBEAT:
        send_message("📊 스퀴즈 모멘텀 — 오늘 새 신호 없음")
    else:
        print("새 신호 없음 — 디스코드 전송 생략")

    for err in errors:
        send_message(err)


if __name__ == "__main__":
    main()
