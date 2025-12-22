import os
import datetime
import time
import requests
import pyupbit
import pandas as pd
import FinanceDataReader as fdr

# =========================
# 환경 변수
# =========================
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

# =========================
# 자산 리스트
# =========================
COINS = ["KRW-BTC", "KRW-ETH", "KRW-XRP"]
US_STOCKS = ["QQQ", "QLD", "TQQQ", "TSLA", "NVDA"]

# =========================
# 디스코드 메시지 전송
# =========================
def send_message(msg: str):
    now = datetime.datetime.now()
    payload = {"content": f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"}
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
    except Exception as e:
        print(f"Discord send failed: {e}")
    print(payload)

# =========================
# 미국 주식 데이터 조회
# =========================
def get_us_stock_data(stock_list, start="2024-01-01"):
    data = {}
    for ticker in stock_list:
        time.sleep(1)  # API 부담 최소화
        df = fdr.DataReader(ticker, start)
        if df.empty:
            continue
        df = df[['Open','Close','High','Low']].reset_index()
        # Date 컬럼을 time으로 통일
        if 'Date' in df.columns:
            df.rename(columns={'Date':'time','Open':'open','Close':'close','High':'high','Low':'low'}, inplace=True)
        else:
            df.rename(columns={'index':'time','Open':'open','Close':'close','High':'high','Low':'low'}, inplace=True)
        df = df.astype({'open':float,'close':float,'high':float,'low':float})
        data[ticker] = df
    return data

# =========================
# Squeeze Momentum 계산 (공통)
# =========================
def squeeze_momentum(df, length_bb=20, mult_bb=2.0, length_kc=20, mult_kc=1.5):
    source = df['close']

    # Bollinger Bands
    df['basis'] = source.rolling(length_bb).mean()
    df['dev'] = source.rolling(length_bb).std(ddof=0) * mult_kc
    df['upper_bb'] = df['basis'] + df['dev']
    df['lower_bb'] = df['basis'] - df['dev']

    # Keltner Channel
    df['ma_kc'] = source.rolling(length_kc).mean()
    df['tr1'] = abs(df['high'] - df['low'])
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['tr1','tr2','tr3']].max(axis=1)
    df['range_ma'] = df['true_range'].rolling(length_kc).mean()
    df['upper_kc'] = df['ma_kc'] + df['range_ma'] * mult_kc
    df['lower_kc'] = df['ma_kc'] - df['range_ma'] * mult_kc

    # Squeeze Off 상태
    df['sqz_off'] = (df['lower_bb'] < df['lower_kc']) & (df['upper_bb'] > df['upper_kc'])

    # Momentum
    avg_high = df['high'].rolling(length_kc).max()
    avg_low = df['low'].rolling(length_kc).min()
    avg_close_ma = df['close'].rolling(length_kc).mean()
    df['val'] = df['close'] - (avg_high + avg_low + avg_close_ma)/3

    return df[['time','close','high','low','upper_bb','lower_bb','upper_kc','lower_kc','sqz_off','val']]

# =========================
# 현재 매수 상태 판단 및 디스코드 전송
# =========================
def check_current_state(asset_name, df):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    is_buy_signal = (not prev['sqz_off'] and last['sqz_off'] and last['val'] > 0)

    if is_buy_signal:
        send_message(f"{asset_name} 🟢 매수 상태 | Close: {last['close']:.2f}")
    else:
        send_message(f"{asset_name} 🔴 관망 상태 | Close: {last['close']:.2f}")

# =========================
# 코인 상태 체크
# =========================
def check_coins():
    for coin in COINS:
        df = pyupbit.get_ohlcv(coin)
        if df is None or len(df) < 30:
            send_message(f"❌ {coin} 데이터 수집 실패")
            continue
        df = df.rename(columns=str.lower).reset_index()      # 인덱스 초기화
        df.rename(columns={'index':'time'}, inplace=True)   # 인덱스 → time 컬럼
        df_calc = squeeze_momentum(df)
        check_current_state(coin, df_calc)

# =========================
# 미국 주식 상태 체크
# =========================
def check_us_stocks():
    data = get_us_stock_data(US_STOCKS)
    for stock in US_STOCKS:
        if stock not in data:
            send_message(f"❌ {stock} 데이터 수집 실패")
            continue
        df = data[stock]
        df_calc = squeeze_momentum(df)
        check_current_state(stock, df_calc)

# =========================
# 메인 실행
# =========================
if __name__ == "__main__":
    check_coins()
    check_us_stocks()
