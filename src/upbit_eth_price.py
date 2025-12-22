import requests
import os
import json

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

def send_to_discord(message: str):
    if not DISCORD_WEBHOOK_URL:
        print("No webhook url")
        return

    payload = {"content": message}
    r = requests.post(
        DISCORD_WEBHOOK_URL,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=10
    )
    print("Discord status:", r.status_code)

def get_eth_price():
    url = "https://api.upbit.com/v1/ticker?markets=KRW-ETH"
    r = requests.get(url, timeout=10)
    data = r.json()[0]
    price = data["trade_price"]
    return price

if __name__ == "__main__":
    price = get_eth_price()
    message = f"📈 업비트 ETH 현재가: {price:,} KRW"
    send_to_discord(message)
