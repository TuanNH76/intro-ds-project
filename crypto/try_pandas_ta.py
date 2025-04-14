import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# Load your CSV
df = pd.read_csv("BTC_price_volume_4y.csv", parse_dates=["Date"])

# Rename if needed
df.rename(columns={"Close Price (USD)": "close", "Volume To (USD)": "volume"}, inplace=True)

# Create indicators
df["sma_20"] = SMAIndicator(close=df["close"], window=20).sma_indicator()
df["macd"] = MACD(close=df["close"]).macd()
df["rsi"] = RSIIndicator(close=df["close"]).rsi()
df["bb_bbm"] = BollingerBands(close=df["close"]).bollinger_mavg()

# Save to CSV
df.to_csv("BTC_with_indicators.csv", index=False)
print("✅ Indicators saved to BTC_with_indicators.csv")
