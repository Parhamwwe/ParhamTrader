"""
Supply/Demand + VWAP + Trend Structure Signal Bot
----------------------------------------------------
همون منطقی که توی Pine Script (Trend Structure + Supply/Demand + VWAP)
پیاده کردیم رو اینجا با پایتون بازسازی می‌کنه:

۱. تشخیص روند با ساختار سوئینگ (HH/HL برای صعودی، LH/LL برای نزولی)
۲. شناسایی نواحی Supply/Demand بر اساس ۳ کندل هم‌جهت با حرکت قوی (>= 1.2*ATR)
۳. سیگنال وقتی «برخورد به ناحیه» و «آخرین لمس/قطع VWAP» حداکثر ۱۰ کندل از
   هم فاصله داشته باشن، به شرطی که جهت روند هم با جهت سیگنال هم‌خوانی داشته باشه

چون این استراتژی به تاریخچه‌ی نواحی نیاز داره (نه فقط آخرین کندل)، هر بار
که اجرا می‌شه یه بازه‌ی ۴۰۰ کندلی رو از اول شبیه‌سازی می‌کنه - دقیقاً مثل
خود TradingView که هر بار چارت باز می‌شه، کل تاریخچه رو دوباره پردازش می‌کنه.

این اسکریپت باید کنار signal_bot.py، توسط همون GitHub Action اجرا بشه.
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd
import ccxt

# ===================== تنظیمات قابل تغییر =====================

TOP_N_BY_VOLUME = 20
QUOTE_CURRENCY = "USDT"
TIMEFRAMES = ["15m", "1h", "4h"]
LOOKBACK_BARS = 400

PIVOT_LEN = 3
IMPULSE_CANDLES = 3
IMPULSE_ATR_MULT = 1.2
ATR_LEN = 14
MAX_ZONE_AGE = 150
VWAP_PROXIMITY_BARS = 10

STATE_FILE = os.path.join(os.path.dirname(__file__), "supply_demand_state.json")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# ===================== توابع کمکی اندیکاتور =====================


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def compute_pivots(highs: np.ndarray, lows: np.ndarray, pivot_len: int):
    """پیدا کردن سقف/کف‌های سوئینگ - دقیقاً هم‌ارز ta.pivothigh/ta.pivotlow"""
    n = len(highs)
    pivot_high = [None] * n
    pivot_low = [None] * n
    for i in range(pivot_len, n - pivot_len):
        window_h = highs[i - pivot_len: i + pivot_len + 1]
        if highs[i] == window_h.max():
            pivot_high[i] = highs[i]
        window_l = lows[i - pivot_len: i + pivot_len + 1]
        if lows[i] == window_l.min():
            pivot_low[i] = lows[i]
    return pivot_high, pivot_low


def compute_daily_vwap(df: pd.DataFrame) -> np.ndarray:
    """VWAP با ریست روزانه (هم‌ارز ta.vwap با anchor پیش‌فرض)"""
    dates = pd.to_datetime(df["ts"], unit="ms").dt.date
    typical = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical * df["volume"]
    cum_tp_vol = tp_vol.groupby(dates).cumsum()
    cum_vol = df["volume"].groupby(dates).cumsum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap.values


# ===================== شبیه‌سازی کامل استراتژی =====================


def run_supply_demand_strategy(df: pd.DataFrame):
    """
    کل تاریخچه‌ی df رو کندل‌به‌کندل شبیه‌سازی می‌کنه (دقیقاً مثل اجرای
    Pine Script روی چارت) و برای هر کندل مشخص می‌کنه سیگنال BUY/SELL
    صادر شده یا نه. خروجی: (strong_buy, strong_sell) - دو تا آرایه‌ی
    bool هم‌طول df.
    """
    n = len(df)
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values

    pivot_high, pivot_low = compute_pivots(highs, lows, PIVOT_LEN)
    vwap = compute_daily_vwap(df)
    atr_series = atr(df, ATR_LEN).values

    swing_high1 = swing_high2 = None
    swing_low1 = swing_low2 = None
    trend_state = "unknown"

    dir_count = 0
    last_dir = 0

    demand_zones = []  # هر عضو: dict با top/bottom/origin_bar/touched/touch_bar/triggered
    supply_zones = []

    last_vwap_cross_bar = None

    strong_buy = [False] * n
    strong_sell = [False] * n

    for i in range(n):
        # ---------- ردیابی لمس/قطع VWAP ----------
        vwap_val = vwap[i]
        if not np.isnan(vwap_val):
            touched_now = lows[i] <= vwap_val <= highs[i]
            crossed_now = False
            if i > 0 and not np.isnan(vwap[i - 1]):
                crossed_now = (closes[i - 1] - vwap[i - 1]) * (closes[i] - vwap_val) < 0
            if touched_now or crossed_now:
                last_vwap_cross_bar = i

        # ---------- تشخیص روند با سوئینگ (تاییدیه با تاخیر PIVOT_LEN کندل) ----------
        confirm_idx = i - PIVOT_LEN
        if confirm_idx >= 0:
            if pivot_high[confirm_idx] is not None:
                swing_high2 = swing_high1
                swing_high1 = pivot_high[confirm_idx]
            if pivot_low[confirm_idx] is not None:
                swing_low2 = swing_low1
                swing_low1 = pivot_low[confirm_idx]

        if None not in (swing_high1, swing_high2, swing_low1, swing_low2):
            if swing_high1 > swing_high2 and swing_low1 > swing_low2:
                trend_state = "up"
            elif swing_high1 < swing_high2 and swing_low1 < swing_low2:
                trend_state = "down"
            # وگرنه روند قبلی معتبر می‌مونه

        is_up = trend_state == "up"
        is_down = trend_state == "down"

        # ---------- تشخیص حرکت شدید (Impulsive Move) ----------
        d = 1 if closes[i] > opens[i] else (-1 if closes[i] < opens[i] else 0)
        if d != 0 and d == last_dir:
            dir_count += 1
        else:
            dir_count = 1 if d != 0 else 0
        last_dir = d

        atr_val = atr_series[i] if i < len(atr_series) else np.nan
        move_size = abs(closes[i] - closes[i - IMPULSE_CANDLES]) if i >= IMPULSE_CANDLES else 0.0

        is_impulsive_up = (
            d == 1 and dir_count == IMPULSE_CANDLES and not np.isnan(atr_val)
            and move_size >= IMPULSE_ATR_MULT * atr_val
        )
        is_impulsive_down = (
            d == -1 and dir_count == IMPULSE_CANDLES and not np.isnan(atr_val)
            and move_size >= IMPULSE_ATR_MULT * atr_val
        )

        if is_impulsive_up and i - IMPULSE_CANDLES >= 0:
            base = i - IMPULSE_CANDLES
            demand_zones.append({
                "top": highs[base], "bottom": lows[base], "origin_bar": base,
                "touched": False, "touch_bar": None, "triggered": False,
            })
        if is_impulsive_down and i - IMPULSE_CANDLES >= 0:
            base = i - IMPULSE_CANDLES
            supply_zones.append({
                "top": highs[base], "bottom": lows[base], "origin_bar": base,
                "touched": False, "touch_bar": None, "triggered": False,
            })

        # ---------- پردازش نواحی Demand ----------
        still_alive = []
        for z in demand_zones:
            too_old = (i - z["origin_bar"]) > MAX_ZONE_AGE
            if closes[i] < z["bottom"] or too_old:
                continue  # ناحیه باطل شد، حذفش می‌کنیم
            if not z["touched"] and lows[i] <= z["top"] and highs[i] >= z["bottom"]:
                z["touched"] = True
                z["touch_bar"] = i
            if (
                z["touched"] and not z["triggered"] and is_up
                and last_vwap_cross_bar is not None
                and abs(z["touch_bar"] - last_vwap_cross_bar) <= VWAP_PROXIMITY_BARS
            ):
                z["triggered"] = True
                strong_buy[i] = True
            still_alive.append(z)
        demand_zones = still_alive

        # ---------- پردازش نواحی Supply ----------
        still_alive = []
        for z in supply_zones:
            too_old = (i - z["origin_bar"]) > MAX_ZONE_AGE
            if closes[i] > z["top"] or too_old:
                continue
            if not z["touched"] and lows[i] <= z["top"] and highs[i] >= z["bottom"]:
                z["touched"] = True
                z["touch_bar"] = i
            if (
                z["touched"] and not z["triggered"] and is_down
                and last_vwap_cross_bar is not None
                and abs(z["touch_bar"] - last_vwap_cross_bar) <= VWAP_PROXIMITY_BARS
            ):
                z["triggered"] = True
                strong_sell[i] = True
            still_alive.append(z)
        supply_zones = still_alive

    return strong_buy, strong_sell


# ===================== دیتا =====================


def get_top_symbols_by_volume(exchange, quote: str = QUOTE_CURRENCY, top_n: int = TOP_N_BY_VOLUME) -> list:
    tickers = exchange.fetch_tickers()
    candidates = []
    for symbol, ticker in tickers.items():
        if not symbol.endswith(f"/{quote}"):
            continue
        volume = ticker.get("quoteVolume")
        if volume is None:
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


def fetch_ohlcv_df(exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.iloc[:-1].reset_index(drop=True)  # آخرین کندل هنوز کامل نشده
    return df


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


def format_message(symbol: str, timeframe: str, side: str, price: float) -> str:
    emoji = "🟢" if side == "BUY" else "🔴"
    return (
        f"{emoji} {side} - Supply/Demand + VWAP\n"
        f"نماد: {symbol}\n"
        f"تایم‌فریم: {timeframe}\n"
        f"قیمت: {price}"
    )


# ===================== مدیریت وضعیت =====================


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

    for symbol in symbols:
        for timeframe in TIMEFRAMES:
            state_key = f"{symbol}_{timeframe}"
            try:
                df = fetch_ohlcv_df(exchange, symbol, timeframe, limit=LOOKBACK_BARS)
                if len(df) < ATR_LEN + PIVOT_LEN + IMPULSE_CANDLES + 5:
                    continue

                strong_buy, strong_sell = run_supply_demand_strategy(df)
                last_idx = len(df) - 1
                candle_ts = int(df["ts"].iloc[last_idx])

                if state.get(state_key) == candle_ts:
                    continue  # قبلاً برای همین کندل چک شده

                if strong_buy[last_idx]:
                    price = float(df["close"].iloc[last_idx])
                    send_telegram_message(format_message(symbol, timeframe, "BUY", price))
                    new_alerts += 1
                elif strong_sell[last_idx]:
                    price = float(df["close"].iloc[last_idx])
                    send_telegram_message(format_message(symbol, timeframe, "SELL", price))
                    new_alerts += 1

                state[state_key] = candle_ts
            except Exception as e:
                print(f"خطا روی {symbol} ({timeframe}): {e}")
                continue
            time.sleep(0.3)

    save_state(state)
    print(f"اجرا تمام شد. تعداد سیگنال جدید: {new_alerts}")


if __name__ == "__main__":
    main()
