


import asyncio
import logging
import httpx
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO

from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator

from telegram import InputFile, Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from telegram.constants import ParseMode

import google.generativeai as genai

# --- Configurations ---
BOT_TOKEN = "8325739398:AAGH9BNR8_kElZe_Dv5fLyOzLa9a9LAwYFQ"
USER_CHAT_ID = 7693531146
COIN_ID = "notcoin"
VS_CURRENCY = "usd"
GEMINI_API_KEY = "AIzaSyAJmhm6zZf4TP177opJF--zjQ8IwTTzuCE"
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "gemini-2.0-flash"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def escape_markdown_v2(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    escape_chars = r'\_*[]()~`>#+-=|{}.!'
    return ''.join(['\\' + c if c in escape_chars else c for c in text])

async def fetch_price_and_history():
    logging.info(f"Fetching price and history for {COIN_ID}...")
    async with httpx.AsyncClient() as client:
        price_url = f"https://api.coingecko.com/api/v3/simple/price?ids={COIN_ID}&vs_currencies={VS_CURRENCY}"
        price_response = await client.get(price_url, timeout=20)
        price_response.raise_for_status()
        price = price_response.json().get(COIN_ID, {}).get(VS_CURRENCY)
        if price is None:
            raise ValueError(f"Could not fetch current price for {COIN_ID}.")

        chart_url = f"https://api.coingecko.com/api/v3/coins/{COIN_ID}/market_chart?vs_currency={VS_CURRENCY}&days=7"
        chart_response = await client.get(chart_url, timeout=20)
        chart_response.raise_for_status()
        data = chart_response.json()

        if "prices" not in data or not data["prices"]:
            raise KeyError("No price data found.")

        df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)
        logging.info(f"Fetched {len(df)} price points.")
        return price, df

def calculate_indicators(df):
    if df.empty or len(df) < 50:
        for col in ["SMA_14", "SMA_50", "RSI", "MACD", "MACD_signal", "MACD_diff", "BB_high", "BB_low", "BB_mid",
                    "Fib_0", "Fib_23.6", "Fib_38.2", "Fib_50", "Fib_61.8", "Fib_100"]:
            df[col] = pd.NA
        return df

    df["SMA_14"] = SMAIndicator(df["price"], window=14).sma_indicator()
    df["SMA_50"] = SMAIndicator(df["price"], window=50).sma_indicator()
    df["RSI"] = RSIIndicator(df["price"], window=14).rsi()
    macd = MACD(df["price"], window_fast=12, window_slow=26, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_diff"] = macd.macd_diff()
    bb = BollingerBands(df["price"], window=20, window_dev=2)
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()
    df["BB_mid"] = bb.bollinger_mavg()

    max_price = df["price"].max()
    min_price = df["price"].min()
    price_range = max_price - min_price
    if price_range > 0:
        df["Fib_0"] = min_price
        df["Fib_23.6"] = min_price + 0.236 * price_range
        df["Fib_38.2"] = min_price + 0.382 * price_range
        df["Fib_50"] = min_price + 0.5 * price_range
        df["Fib_61.8"] = min_price + 0.618 * price_range
        df["Fib_100"] = max_price
    else:
        df["Fib_0"] = df["Fib_23.6"] = df["Fib_38.2"] = df["Fib_50"] = df["Fib_61.8"] = df["Fib_100"] = min_price

    return df

def generate_signal(df):
    if df.empty or len(df) < 2:
        return "⚠️ داده ناکافی", []

    latest = df.iloc[-1]
    if any(pd.isna(latest.get(ind)) for ind in ["RSI", "MACD", "MACD_signal", "SMA_14", "BB_low", "BB_high"]):
        return "⚠️ اندیکاتورها موجود نیستند", []

    buy, sell = [], []

    if latest["RSI"] < 30: buy.append("RSI < 30 (اشباع فروش)")
    if latest["RSI"] > 70: sell.append("RSI > 70 (اشباع خرید)")
    if latest["MACD"] > latest["MACD_signal"] and df["MACD"].iloc[-2] < df["MACD_signal"].iloc[-2]:
        buy.append("تقاطع طلایی MACD")
    if latest["MACD"] < latest["MACD_signal"] and df["MACD"].iloc[-2] > df["MACD_signal"].iloc[-2]:
        sell.append("تقاطع مرگ MACD")
    if latest["price"] <= latest["BB_low"] * 1.01: buy.append("قیمت نزدیک باند پایین بولینگر")
    if latest["price"] >= latest["BB_high"] * 0.99: sell.append("قیمت نزدیک باند بالا بولینگر")

    if len(buy) >= 2: return "📈 **خرید قوی**", buy
    if len(sell) >= 2: return "📉 **فروش قوی**", sell
    if len(buy) >= 1: return "⬆️ احتمالا خرید", buy
    if len(sell) >= 1: return "⬇️ احتمالا فروش", sell
    return "⚪️ سیگنال واضحی نیست", []

def plot_chart(df, current_price):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 18), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})

    ax1.plot(df.index, df["price"], label="قیمت", color="blue", linewidth=1.5)
    ax1.plot(df.index, df["SMA_14"], label="SMA 14", color="orange", linestyle="--", linewidth=1)
    ax1.plot(df.index, df["SMA_50"], label="SMA 50", color="purple", linestyle="-.", linewidth=1)

    if "BB_high" in df.columns and pd.notna(df["BB_high"].iloc[-1]):
        ax1.fill_between(df.index, df["BB_low"], df["BB_high"], color='gray', alpha=0.2, label='باند بولینگر')
    if "Fib_0" in df.columns and pd.notna(df["Fib_0"].iloc[-1]):
        fib_levels = {"23.6%": "Fib_23.6", "38.2%": "Fib_38.2", "50%": "Fib_50", "61.8%": "Fib_61.8"}
        for name, col in fib_levels.items():
            ax1.axhline(y=df[col].iloc[-1], color='red', linestyle='--', linewidth=0.7, alpha=0.7)
            ax1.text(df.index[0], df[col].iloc[-1], f' {name}', color='red', va='center', ha='left', fontsize=9)

    ax1.set_title(f"{COIN_ID.upper()} - نمودار 7 روزه (قیمت فعلی: ${current_price:,.4f})")
    ax1.set_ylabel("قیمت (دلار)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(df.index, df["RSI"], label="RSI", color="purple")
    ax2.axhline(y=70, color='red', linestyle=':', label="اشباع خرید (70)")
    ax2.axhline(y=30, color='green', linestyle=':', label="اشباع فروش (30)")
    ax2.set_ylabel("RSI")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    ax3.plot(df.index, df["MACD"], label="MACD", color="blue")
    ax3.plot(df.index, df["MACD_signal"], label="سیگنال", color="red", linestyle="--")
    ax3.bar(df.index, df["MACD_diff"], label="هیستوگرام", color=np.where(df['MACD_diff'] > 0, 'g', 'r'), alpha=0.5)
    ax3.set_xlabel("تاریخ")
    ax3.set_ylabel("MACD")
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png', dpi=150)
    image_stream.seek(0)
    plt.close(fig)
    return image_stream

async def generate_farsi_prompt_and_get_gemini_analysis(coin_id: str, current_price: float, df: pd.DataFrame, model_name: str = GEMINI_MODEL_NAME):
    try:
        last_data = df[['price', 'SMA_14', 'SMA_50', 'RSI', 'MACD', 'MACD_signal', 'BB_high', 'BB_low']].tail(20)
        data_str = last_data.to_string(index=True, float_format="{:,.4f}".format)
        prompt = (
            f"شما یک تحلیلگر حرفه‌ای بازار ارزهای دیجیتال هستید.\n\n"
            f"لطفاً بر اساس داده‌های زیر برای ارز {coin_id.upper()} که شامل قیمت‌های 7 روز گذشته و اندیکاتورهای تکنیکال است، تحلیل کوتاه، دقیق و واضح بازار ارائه دهید.\n\n"
            f"قیمت فعلی: {current_price:,.4f} دلار\n\n"
            f"داده‌های قیمت و اندیکاتورها:\n{data_str}\n\n"
            f"لطفاً یک سیگنال مشخص (خرید، فروش، نگهداری) همراه با دلیل آن به زبان ساده ارائه دهید.\n"
            f"تحلیل شما باید حداکثر 150 کلمه باشد و فقط متن ساده بدون هیچ فرمت یا علامت خاصی باشد."
        )
        model = genai.GenerativeModel(model_name)
        response = await model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ خطا در دریافت تحلیل از Gemini: {e}"

# برای نگه‌داشتن قیمت فروش (با مقدار اولیه None)
SELL_PRICE = None

async def auto_alert(context: ContextTypes.DEFAULT_TYPE):
    global SELL_PRICE
    try:
        current_price, df = await fetch_price_and_history()
        if df.shape[0] < 50:
            msg = f"⚠️ داده ناکافی برای تحلیل کامل ({df.shape[0]} داده). حداقل 50 داده نیاز است."
            await context.bot.send_message(chat_id=USER_CHAT_ID, text=escape_markdown_v2(msg), parse_mode=ParseMode.MARKDOWN_V2)
            return

        df = calculate_indicators(df)
        local_signal, reasons = generate_signal(df)
        chart_img = plot_chart(df, current_price)
        gemini_analysis = await generate_farsi_prompt_and_get_gemini_analysis(COIN_ID, current_price, df)

        caption_parts = [
            f"🪙 *تحلیل {escape_markdown_v2(COIN_ID.upper())}*",
            f"💵 قیمت فعلی: `${escape_markdown_v2(f'{current_price:,.4f}')}`",
            f"🔔 سیگنال: *{escape_markdown_v2(local_signal)}*",
        ]
        if reasons:
            reasons_text = "\n".join([f"\\- {escape_markdown_v2(r)}" for r in reasons])
            caption_parts.append(f"\n*دلایل سیگنال:*\n{reasons_text}")

        caption_parts.append(f"\n🤖 *تحلیل Gemini AI:*\n{escape_markdown_v2(gemini_analysis)}")

        if SELL_PRICE is not None and current_price <= SELL_PRICE:
            caption_parts.append(f"\n⚠️ قیمت فعلی پایین‌تر یا برابر با قیمت فروش تنظیم شده است: {SELL_PRICE}")
        
        final_caption = "\n".join(caption_parts)

        await context.bot.send_photo(
            chat_id=USER_CHAT_ID,
            photo=InputFile(chart_img),
            caption=final_caption,
            parse_mode=ParseMode.MARKDOWN_V2
        )
    except Exception as e:
        logging.error(f"Error in auto_alert: {e}", exc_info=True)
        try:
            await context.bot.send_message(
                chat_id=USER_CHAT_ID,
                text=escape_markdown_v2(f"⚠️ خطا در اجرای برنامه: {e}"),
                parse_mode=ParseMode.MARKDOWN_V2
            )
        except Exception as se:
            logging.error(f"Failed to send error message: {se}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=escape_markdown_v2(f"🤖 ربات تحلیل ارز دیجیتال فعال شد.\n\n"
                                f"کوین مورد نظر: {COIN_ID.upper()}\n"
                                f"قیمت فروش را می‌توانید با دستور زیر تنظیم کنید:\n"
                                f"/set_sell_price مقدار_قیمت\n\n"
                                f"مثال:\n"
                                f"/set_sell_price 28000"),
        parse_mode=ParseMode.MARKDOWN_V2
    )

async def set_sell_price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SELL_PRICE
    try:
        if len(context.args) != 1:
            raise ValueError("لطفا فقط یک مقدار عددی وارد کنید.")
        price = float(context.args[0])
        SELL_PRICE = price
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                       text=f"✅ قیمت فروش با موفقیت روی {price} تنظیم شد.")
    except Exception:
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                       text="❌ مقدار وارد شده معتبر نیست. لطفا یک عدد صحیح یا اعشاری وارد کنید.")

if __name__ == "__main__":
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("set_sell_price", set_sell_price))

    # هر 5 دقیقه یکبار اجرا (300 ثانیه)
    app.job_queue.run_repeating(auto_alert, interval=300, first=10)

    print("ربات فعال است...")
    app.run_polling()
