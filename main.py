import os
import logging
import asyncio
import httpx
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO

from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes,
    JobQueue
)
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
import google.generativeai as genai

# خواندن توکن‌ها از محیط
BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# تنظیم Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# متغیر تنظیم قیمت فروش
sell_price = None

async def fetch_crypto_data():
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "5m", "limit": 100}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "time", "open", "high", "low", "close", "volume",
        "_", "__", "___", "____", "_____", "______"
    ])
    df["close"] = pd.to_numeric(df["close"])
    df["high"] = pd.to_numeric(df["high"])
    df["low"] = pd.to_numeric(df["low"])
    return df

def compute_indicators(df):
    df["SMA_14"] = SMAIndicator(df["close"], window=14).sma_indicator()
    df["SMA_50"] = SMAIndicator(df["close"], window=50).sma_indicator()
    df["RSI"] = RSIIndicator(df["close"], window=14).rsi()
    macd = MACD(df["close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_diff"] = macd.macd_diff()
    bb = BollingerBands(df["close"])
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()
    df["BB_mid"] = bb.bollinger_mavg()
    # فیبوناچی دستی
    max_price = df["high"].max()
    min_price = df["low"].min()
    diff = max_price - min_price
    for level, name in zip([0, 0.236, 0.382, 0.5, 0.618, 1.0],
                           ["Fib_0", "Fib_23.6", "Fib_38.2", "Fib_50", "Fib_61.8", "Fib_100"]):
        df[name] = max_price - level * diff
    return df

def create_plot(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["close"], label="Close", color="black")
    ax.plot(df["SMA_14"], label="SMA 14", color="blue")
    ax.plot(df["SMA_50"], label="SMA 50", color="red")
    ax.fill_between(df.index, df["BB_low"], df["BB_high"], color="gray", alpha=0.2, label="Bollinger Bands")
    ax.set_title("نمودار قیمت و اندیکاتورها")
    ax.legend()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf

async def generate_analysis(df):
    last = df.iloc[-1]
    prompt = (
        f"قیمت فعلی بیت‌کوین: {last['close']:.2f} دلار\n"
        f"RSI: {last['RSI']:.2f}\n"
        f"MACD: {last['MACD']:.2f}, Signal: {last['MACD_signal']:.2f}, Diff: {last['MACD_diff']:.2f}\n"
        f"SMA 14: {last['SMA_14']:.2f}, SMA 50: {last['SMA_50']:.2f}\n"
        f"Bollinger High: {last['BB_high']:.2f}, Low: {last['BB_low']:.2f}\n"
        f"سطوح فیبوناچی: 0: {last['Fib_0']:.2f}, 23.6%: {last['Fib_23.6']:.2f}, 38.2%: {last['Fib_38.2']:.2f}, "
        f"50%: {last['Fib_50']:.2f}, 61.8%: {last['Fib_61.8']:.2f}, 100%: {last['Fib_100']:.2f}\n"
        f"لطفا براساس این داده‌ها به زبان فارسی سیگنال خرید یا فروش بده."
    )
    response = model.generate_content(prompt)
    return response.text

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = await fetch_crypto_data()
    df = compute_indicators(df)
    plot = create_plot(df)
    analysis = await generate_analysis(df)
    await update.message.reply_photo(photo=plot, caption=analysis)

async def set_sell_price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global sell_price
    try:
        price = float(context.args[0])
        sell_price = price
        await update.message.reply_text(f"✅ قیمت فروش روی {sell_price} دلار تنظیم شد.")
    except:
        await update.message.reply_text("❌ لطفا عدد معتبر وارد کن. مثال:\n/set_sell_price 60200")

async def auto_alert(context: ContextTypes.DEFAULT_TYPE):
    global sell_price
    if not sell_price:
        return
    df = await fetch_crypto_data()
    current_price = df["close"].iloc[-1]
    if current_price < sell_price:
        await context.bot.send_message(
            chat_id=context.job.chat_id,
            text=f"📉 هشدار! قیمت بیت‌کوین ({current_price}$) از قیمت فروش تنظیم‌شده ({sell_price}$) پایین‌تر رفته است!"
        )

async def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("set_sell_price", set_sell_price))

    app.job_queue.run_repeating(auto_alert, interval=300, first=10)

    print("ربات فعال شد.")
    await app.run_polling()

if __name__ == "__main__":
    asyncio.run(main())

