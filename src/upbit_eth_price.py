import os
import datetime
import requests
import pyupbit

# =========================
# 환경 변수
# =========================
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
TICKERS = ["KRW-BTC", "KRW-ETH", "KRW-XRP"]  # 멀티 코인


# =========================
# 디스코드 메시지 전송
# =========================
def send_message(msg: str):
    now = datetime.datetime.now()
    payload = {
        "content": f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    }
    requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)


# =========================
# 업비트 데이터 조회
# =========================
def get_ohlcv(ticker: str, count: int = 200):
    return pyupbit.get_ohlcv(ticker, interval="day", count=count)


# =========================
# Squeeze Momentum 계산
# =========================
def squeeze_momentum(df, length_bb=20, length_kc=20, mult_bb=2.0, mult_kc=1.5):
    close = df["close"]

    # Bollinger Bands
    basis = close.rolling(length_bb).mean()
    dev = close.rolling(length_bb).std(ddof=0) * mult_bb
    upper_bb = basis + dev
    lower_bb = basis - dev

    # Keltner Channel
    ma_kc = close.rolling(length_kc).mean()
    tr = (df["high"] - df["low"]).rolling(length_kc).mean()
    upper_kc = ma_kc + tr * mult_kc
    lower_kc = ma_kc - tr * mult_kc

    # Squeeze 상태
    sqz_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)

    # Momentum
    avg_high = df["high"].rolling(length_kc).max()
    avg_low = df["low"].rolling(length_kc).min()
    avg_close = close.rolling(length_kc).mean()
    center = (avg_high + avg_low + avg_close) / 3
    momentum = close - center

    df["sqz_off"] = sqz_off
    df["momentum"] = momentum

    return df


# =========================
# 현재 상태 판단
# =========================
def check_current_state(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Squeeze OFF 전환 + 모멘텀 양수
    is_buy_signal = (
        not prev["sqz_off"]
        and last["sqz_off"]
        and last["momentum"] > 0
    )

    if is_buy_signal:
        return "🟢 매수 상태 (Squeeze OFF + Positive Momentum)"
    else:
        return "🔴 관망 상태 (No Buy Signal)"


# =========================
# 실행부 (cron 전용)
# =========================
if __name__ == "__main__":
    for ticker in TICKERS:
        df = get_ohlcv(ticker)

        if df is None or len(df) < 30:
            send_message(f"❌ {ticker} 데이터 수집 실패")
            continue

        df = squeeze_momentum(df)
        state = check_current_state(df)

        send_message(f"{ticker} Squeeze Momentum 상태 → {state}")
