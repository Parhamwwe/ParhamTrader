"""
Toobit Multi-Timeframe Signal Bot
----------------------------------
همون منطق استراتژی Trend + RSI + MACD + ADX + MTF + Volume که توی
Pine Script نوشته شده بود رو اینجا با پایتون بازسازی می‌کنه، مستقیم
از Toobit دیتا می‌گیره (از طریق کتابخونه ccxt) و وقتی سیگنال BUY یا
SELL پیدا بشه، به تلگرام پیام می‌فرسته.

این اسکریپت قراره توسط GitHub Actions هر ۱۵ دقیقه یک‌بار اجرا بشه
(فایل .github/workflows/signal-bot.yml رو ببین).
"""

import os
import json
import time
import requests
import pandas as pd
import ccxt

# ===================== تنظیمات قابل تغییر =====================

# به‌جای لیست ثابت، هر بار که اسکریپت اجرا می‌شه (هر ۱۵ دقیقه)، خودش از Toobit
# همه‌ی جفت‌ارزهای USDT رو می‌گیره، بر اساس حجم معامله (quoteVolume) مرتب می‌کنه
# و همین تعداد از پرحجم‌ترین‌ها رو انتخاب می‌کنه. یعنی لیست همیشه به‌روزه، نه
# فقط هر ۱ ساعت.
TOP_N_BY_VOLUME = 20
QUOTE_CURRENCY = "USDT"

# تایم‌فریم‌هایی که چک می‌شن، و تایم‌فریم بالاتر (HTF) متناظر هرکدوم
# برای فیلتر Multi-Timeframe
TIMEFRAMES = {
    "15m": "1h",
    "1h": "4h",
    "4h": "1d",
}

# پارامترهای استراتژی (دقیقاً هم‌ارز نسخه Pine Script)
EMA_FAST_LEN = 50
EMA_SLOW_LEN = 200
RSI_LEN = 14
RSI_OVERSOLD = 45
RSI_OVERBOUGHT = 55
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ADX_LEN = 14
ADX_THRESHOLD = 20
VOLUME_MA_LEN = 20
ATR_LEN = 14
ATR_SL_MULT = 1.5
ATR_TP_MULT = 3.0

USE_ADX_FILTER = True
USE_MTF_FILTER = True
USE_VOLUME_FILTER = True

STATE_FILE = os.path.join(os.path.dirname(__file__), "state.json")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# ===================== توابع کمکی اندیکاتور =====================


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int, slow: int, signal: int):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def adx(df: pd.DataFrame, length: int) -> pd.Series:
    high, low = df["high"], df["low"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    tr_atr = atr(df, length).replace(0, 1e-10)

    plus_di = 100 * plus_dm.ewm(alpha=1 / length, adjust=False).mean() / tr_atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / length, adjust=False).mean() / tr_atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
    return dx.ewm(alpha=1 / length, adjust=False).mean()


# ===================== دیتا و سیگنال =====================


def get_top_symbols_by_volume(exchange, quote: str = QUOTE_CURRENCY, top_n: int = TOP_N_BY_VOLUME) -> list:
    """
    همه‌ی جفت‌ارزهای بازار اسپات Toobit با ارز پایه‌ی quote (مثلاً USDT) رو
    می‌گیره، بر اساس حجم معامله‌ی ۲۴ ساعته (quoteVolume) نزولی مرتب می‌کنه،
    و اسم N تای اول رو برمی‌گردونه.
    """
    tickers = exchange.fetch_tickers()
    candidates = []
    for symbol, ticker in tickers.items():
        if not symbol.endswith(f"/{quote}"):
            continue
        volume = ticker.get("quoteVolume")
        if volume is None:
            # بعضی صرافی‌ها quoteVolume نمی‌دن، از baseVolume * آخرین قیمت تخمین می‌زنیم
            base_volume = ticker.get("baseVolume")
            last_price = ticker.get("last")
            if base_volume is not None and last_price is not None:
                volume = base_volume * last_price
        if volume is None:
            continue
        candidates.append((symbol, volume))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top_symbols = [symbol for symbol, _ in candidates[:top_n]]
    print(f"لیست {len(top_symbols)} نماد پرحجم فعلی: {top_symbols}")
    return top_symbols


def fetch_ohlcv_df(exchange, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    # آخرین کندل معمولاً هنوز کامل نشده، پس حذفش می‌کنیم و فقط کندل‌های بسته‌شده رو نگه می‌داریم
    df = df.iloc[:-1].reset_index(drop=True)
    return df


def get_htf_trend(exchange, symbol: str, htf_timeframe: str) -> str:
    """جهت روند تایم‌فریم بالاتر رو برمی‌گردونه: 'up' یا 'down'"""
    df = fetch_ohlcv_df(exchange, symbol, htf_timeframe, limit=EMA_SLOW_LEN + 10)
    if len(df) < 5:
        return "unknown"
    htf_ema = ema(df["close"], min(EMA_FAST_LEN, len(df) - 1))
    last_close = df["close"].iloc[-1]
    return "up" if last_close > htf_ema.iloc[-1] else "down"


def evaluate_signal(exchange, symbol: str, timeframe: str, htf_timeframe: str):
    df = fetch_ohlcv_df(exchange, symbol, timeframe, limit=max(EMA_SLOW_LEN + 50, 260))
    if len(df) < EMA_SLOW_LEN + 5:
        return None  # دیتای کافی نیست

    close = df["close"]
    ema_fast = ema(close, EMA_FAST_LEN)
    ema_slow = ema(close, EMA_SLOW_LEN)
    rsi_val = rsi(close, RSI_LEN)
    macd_line, signal_line = macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    adx_val = adx(df, ADX_LEN)
    vol_ma = df["volume"].rolling(VOLUME_MA_LEN).mean()
    atr_val = atr(df, ATR_LEN)

    i = -1  # آخرین کندل بسته‌شده
    prev = -2

    uptrend = ema_fast.iloc[i] > ema_slow.iloc[i]
    downtrend = ema_fast.iloc[i] < ema_slow.iloc[i]

    rsi_bullish = (rsi_val.iloc[i] < RSI_OVERSOLD) and (rsi_val.iloc[i] > rsi_val.iloc[prev])
    rsi_bearish = (rsi_val.iloc[i] > RSI_OVERBOUGHT) and (rsi_val.iloc[i] < rsi_val.iloc[prev])

    macd_bullish = (macd_line.iloc[prev] <= signal_line.iloc[prev]) and (
        macd_line.iloc[i] > signal_line.iloc[i]
    )
    macd_bearish = (macd_line.iloc[prev] >= signal_line.iloc[prev]) and (
        macd_line.iloc[i] < signal_line.iloc[i]
    )

    strong_trend = (not USE_ADX_FILTER) or (adx_val.iloc[i] > ADX_THRESHOLD)
    volume_confirm = (not USE_VOLUME_FILTER) or (df["volume"].iloc[i] > vol_ma.iloc[i])

    if USE_MTF_FILTER:
        htf_trend = get_htf_trend(exchange, symbol, htf_timeframe)
        htf_up = htf_trend == "up"
        htf_down = htf_trend == "down"
    else:
        htf_up = htf_down = True

    buy_signal = uptrend and rsi_bullish and macd_bullish and strong_trend and htf_up and volume_confirm
    sell_signal = downtrend and rsi_bearish and macd_bearish and strong_trend and htf_down and volume_confirm

    if not (buy_signal or sell_signal):
        return None

    last_close = close.iloc[i]
    if buy_signal:
        sl = last_close - atr_val.iloc[i] * ATR_SL_MULT
        tp = last_close + atr_val.iloc[i] * ATR_TP_MULT
        side = "BUY"
    else:
        sl = last_close + atr_val.iloc[i] * ATR_SL_MULT
        tp = last_close - atr_val.iloc[i] * ATR_TP_MULT
        side = "SELL"

    candle_ts = int(df["ts"].iloc[i])

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "side": side,
        "entry": round(float(last_close), 6),
        "sl": round(float(sl), 6),
        "tp": round(float(tp), 6),
        "candle_ts": candle_ts,
    }


# ===================== تلگرام =====================


def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("توکن یا چت آیدی تلگرام تنظیم نشده - پیام ارسال نشد.")
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
    if resp.status_code != 200:
        print(f"خطا در ارسال تلگرام: {resp.status_code} {resp.text}")


def format_message(sig: dict) -> str:
    emoji = "🟢" if sig["side"] == "BUY" else "🔴"
    return (
        f"{emoji} {sig['side']} سیگنال\n"
        f"نماد: {sig['symbol']}\n"
        f"تایم‌فریم: {sig['timeframe']}\n"
        f"قیمت ورود: {sig['entry']}\n"
        f"SL: {sig['sl']}\n"
        f"TP: {sig['tp']}"
    )


# ===================== مدیریت وضعیت (جلوگیری از پیام تکراری) =====================


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ===================== اجرای اصلی =====================


def main():
    exchange = ccxt.toobit({"enableRateLimit": True})
    state = load_state()
    new_alerts = 0

    try:
        symbols = get_top_symbols_by_volume(exchange)
    except Exception as e:
        print(f"خطا در گرفتن لیست نمادهای پرحجم: {e}")
        return

    if not symbols:
        print("هیچ نمادی پیدا نشد، اجرا متوقف شد.")
        return

    for symbol in symbols:
        for timeframe, htf_timeframe in TIMEFRAMES.items():
            state_key = f"{symbol}_{timeframe}"
            try:
                sig = evaluate_signal(exchange, symbol, timeframe, htf_timeframe)
            except Exception as e:
                print(f"خطا روی {symbol} ({timeframe}): {e}")
                continue

            if sig is None:
                continue

            last_alerted_ts = state.get(state_key)
            if last_alerted_ts == sig["candle_ts"]:
                continue  # قبلاً برای همین کندل پیام فرستاده شده

            send_telegram_message(format_message(sig))
            state[state_key] = sig["candle_ts"]
            new_alerts += 1
            time.sleep(0.5)  # جلوگیری از rate limit تلگرام و صرافی

    save_state(state)
    print(f"اجرا تمام شد. تعداد سیگنال جدید: {new_alerts}")


if __name__ == "__main__":
    main()
