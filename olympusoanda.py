"""
╔══════════════════════════════════════════════════════════════╗
║         GBP/USD OLYMPUS SIGNAL BOT v4                       ║
║         Fully Independent — No TradingView Required         ║
║         Signal Engine: Lorentzian KNN Classifier            ║
║         Data Source: OANDA API                              ║
║         Alerts: Telegram                                     ║
║         Scan: Every 1H candle close                         ║
╚══════════════════════════════════════════════════════════════╝

SETUP INSTRUCTIONS:
──────────────────────────────────────
STEP 1 — GET OANDA API KEY
  1. Go to https://www.oanda.com
  2. Create a free practice account
  3. Go to: My Account → Manage API Access
  4. Generate an API key — copy it
  5. Also copy your Account ID from the dashboard
  6. Practice account uses: https://api-fxtrade.oanda.com
     Live account uses:     https://api-fxtrade.oanda.com
     (keep OANDA_ENV = "practice" until you go live)

STEP 2 — TELEGRAM BOT (if not already done)
  1. Open Telegram → search @BotFather
  2. Send: /newbot
  3. Copy the token it gives you
  4. Search @userinfobot → send any message → copy your Chat ID

STEP 3 — SET ENVIRONMENT VARIABLES ON RENDER
  BOT_TOKEN        → Telegram bot token
  CHAT_ID          → Telegram chat ID
  OANDA_API_KEY    → OANDA API key
  OANDA_ACCOUNT_ID → OANDA account ID
  OANDA_ENV        → practice  (or live)

STEP 4 — GITHUB REPO FILES
  gbpusd_signal_bot.py   ← this file
  requirements.txt
  render.yaml

──────────────────────────────────────
HOW THE SIGNAL ENGINE WORKS:
  Rebuilt from the Lorentzian Classification indicator logic:

  1. FEATURES — same as your Olympus settings
     RSI(14), WT(10,11), CCI(20), ADX(20), RSI(9)

  2. LORENTZIAN KNN CLASSIFIER
     Compares current bar features to last 2000 bars
     Finds 6 nearest neighbors using Lorentzian distance
     Votes bullish or bearish based on future price direction

  3. KERNEL REGRESSION FILTER
     Lookback: 18, Relative Weighting: 14
     Confirms trend direction before signaling

  4. FILTERS
     Volatility filter — avoids choppy markets
     Regime filter    — threshold -0.1

  5. SIGNAL OUTPUT
     BUY  — when KNN + Kernel both bullish + filters pass
     SELL — when KNN + Kernel both bearish + filters pass

  6. DYNAMIC SL (same engine as before)
     ATR + Session Volume + TF Base combined
──────────────────────────────────────
"""

import os
import time
import math
import requests
import numpy as np
from datetime import datetime, timezone
from flask import Flask, jsonify

app = Flask(__name__)

# ══════════════════════════════════════
# CREDENTIALS — set in Render env vars
# ══════════════════════════════════════
BOT_TOKEN        = os.environ.get("BOT_TOKEN")
CHAT_ID          = os.environ.get("CHAT_ID")
OANDA_API_KEY    = os.environ.get("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID")
OANDA_ENV        = os.environ.get("OANDA_ENV", "practice")
# ══════════════════════════════════════

INSTRUMENT  = "GBP_USD"
GRANULARITY = "H1"
CANDLE_COUNT = 2100  # enough for 2000 lookback + buffer
PIP_SIZE    = 0.0001

# TP Ratios from Olympus settings
TP1_RATIO = 1.0
TP2_RATIO = 1.75
TP3_RATIO = 4.5

# KNN settings from Olympus
NEIGHBORS_COUNT  = 6
MAX_BARS_BACK    = 2000
FEATURE_COUNT    = 5

# Kernel settings
LOOKBACK_WINDOW    = 18
RELATIVE_WEIGHTING = 14
REGRESSION_LEVEL   = 35

# Regime filter threshold
REGIME_THRESHOLD = -0.1


# ══════════════════════════════════════
# OANDA DATA FETCHER
# ══════════════════════════════════════

def get_oanda_url() -> str:
    if OANDA_ENV == "live":
        return "https://api-fxtrade.oanda.com"
    return "https://api-fxtrade.oanda.com"


def fetch_candles(count: int = CANDLE_COUNT) -> list:
    """Fetch GBP/USD H1 candles from OANDA."""
    url = f"{get_oanda_url()}/v3/instruments/{INSTRUMENT}/candles"
    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Content-Type": "application/json"
    }
    params = {
        "count"      : count,
        "granularity": GRANULARITY,
        "price"      : "M"  # midpoint
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    candles = response.json().get("candles", [])

    # Only use complete candles
    return [c for c in candles if c["complete"]]


def parse_candles(candles: list) -> dict:
    """Parse OANDA candles into numpy arrays."""
    opens   = np.array([float(c["mid"]["o"]) for c in candles])
    highs   = np.array([float(c["mid"]["h"]) for c in candles])
    lows    = np.array([float(c["mid"]["l"]) for c in candles])
    closes  = np.array([float(c["mid"]["c"]) for c in candles])
    volumes = np.array([float(c["volume"])   for c in candles])
    times   = [c["time"] for c in candles]

    return {
        "open"  : opens,
        "high"  : highs,
        "low"   : lows,
        "close" : closes,
        "volume": volumes,
        "time"  : times
    }


# ══════════════════════════════════════
# TECHNICAL INDICATORS
# ══════════════════════════════════════

def calc_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI calculation."""
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.zeros(len(closes))
    avg_loss = np.zeros(len(closes))

    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    for i in range(period + 1, len(closes)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period

    rs  = np.where(avg_loss == 0, 100.0, avg_gain / avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = 50.0
    return rsi


def calc_wt(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
            period_a: int = 10, period_b: int = 11) -> np.ndarray:
    """Wave Trend oscillator."""
    hlc3   = (highs + lows + closes) / 3.0
    esa    = _ema(hlc3, period_a)
    d      = _ema(np.abs(hlc3 - esa), period_a)
    ci     = np.where(d * 0.015 == 0, 0, (hlc3 - esa) / (0.015 * d))
    wt1    = _ema(ci, period_b)
    return np.nan_to_num(wt1, nan=0.0)


def calc_cci(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
             period: int = 20) -> np.ndarray:
    """CCI calculation."""
    tp  = (highs + lows + closes) / 3.0
    cci = np.zeros(len(tp))
    for i in range(period - 1, len(tp)):
        window   = tp[i - period + 1:i + 1]
        mean_tp  = np.mean(window)
        mean_dev = np.mean(np.abs(window - mean_tp))
        cci[i]   = 0 if mean_dev == 0 else (tp[i] - mean_tp) / (0.015 * mean_dev)
    return cci


def calc_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
             period: int = 20) -> np.ndarray:
    """ADX calculation."""
    n      = len(closes)
    tr     = np.zeros(n)
    pdm    = np.zeros(n)
    ndm    = np.zeros(n)

    for i in range(1, n):
        hl  = highs[i]  - lows[i]
        hpc = abs(highs[i]  - closes[i-1])
        lpc = abs(lows[i]   - closes[i-1])
        tr[i]  = max(hl, hpc, lpc)
        up     = highs[i] - highs[i-1]
        down   = lows[i-1] - lows[i]
        pdm[i] = up   if (up > down and up > 0)   else 0
        ndm[i] = down if (down > up and down > 0) else 0

    atr_  = _rma(tr,  period)
    pdi   = 100 * _rma(pdm, period) / np.where(atr_ == 0, 1, atr_)
    ndi   = 100 * _rma(ndm, period) / np.where(atr_ == 0, 1, atr_)
    dx    = 100 * np.abs(pdi - ndi) / np.where((pdi + ndi) == 0, 1, (pdi + ndi))
    adx   = _rma(dx, period)
    return np.nan_to_num(adx, nan=0.0)


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    alpha  = 2.0 / (period + 1)
    result = np.zeros(len(data))
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result


def _rma(data: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing (RMA)."""
    alpha  = 1.0 / period
    result = np.zeros(len(data))
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result


def calc_atr(highs: np.ndarray, lows: np.ndarray,
             closes: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR calculation."""
    n  = len(closes)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i]  - closes[i-1]),
            abs(lows[i]   - closes[i-1])
        )
    return _rma(tr, period)


# ══════════════════════════════════════
# LORENTZIAN KNN CLASSIFIER
# ══════════════════════════════════════

def lorentzian_distance(f1: np.ndarray, f2: np.ndarray) -> float:
    """Lorentzian distance metric — same as jdehorty's indicator."""
    return sum(math.log(1 + abs(f1[i] - f2[i])) for i in range(len(f1)))


def normalize(series: np.ndarray, lookback: int = 500) -> np.ndarray:
    """Normalize series to 0-1 range over rolling window."""
    result = np.zeros(len(series))
    for i in range(len(series)):
        start = max(0, i - lookback)
        window = series[start:i+1]
        mn, mx = window.min(), window.max()
        result[i] = 0.5 if mx == mn else (series[i] - mn) / (mx - mn)
    return result


def build_features(data: dict) -> np.ndarray:
    """
    Build 5 normalized features matching Olympus/Lorentzian settings:
    RSI(14), WT(10,11), CCI(20), ADX(20), RSI(9)
    """
    closes = data["close"]
    highs  = data["high"]
    lows   = data["low"]

    f1 = normalize(calc_rsi(closes, 14))
    f2 = normalize(calc_wt(highs, lows, closes, 10, 11))
    f3 = normalize(calc_cci(highs, lows, closes, 20))
    f4 = normalize(calc_adx(highs, lows, closes, 20))
    f5 = normalize(calc_rsi(closes, 9))

    return np.column_stack([f1, f2, f3, f4, f5])


def knn_classify(features: np.ndarray, closes: np.ndarray,
                 bar_index: int) -> int:
    """
    KNN Lorentzian classifier.
    Returns: 1 = bullish, -1 = bearish, 0 = no signal
    """
    if bar_index < NEIGHBORS_COUNT + 4:
        return 0

    current_features = features[bar_index]
    distances = []

    start = max(0, bar_index - MAX_BARS_BACK)
    # Sample every 4 bars (matching Pine Script behavior)
    for i in range(start, bar_index - 4, 4):
        if i + 4 >= len(closes):
            continue
        dist  = lorentzian_distance(current_features, features[i])
        label = 1 if closes[i + 4] > closes[i] else -1
        distances.append((dist, label))

    if not distances:
        return 0

    # Get k nearest neighbors
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:NEIGHBORS_COUNT]

    vote = sum(label for _, label in neighbors)
    if vote > 0:
        return 1
    elif vote < 0:
        return -1
    return 0


# ══════════════════════════════════════
# KERNEL REGRESSION FILTER
# ══════════════════════════════════════

def rational_quadratic_kernel(closes: np.ndarray, bar_index: int,
                               lookback: int = LOOKBACK_WINDOW,
                               weight: float = RELATIVE_WEIGHTING,
                               level: int = REGRESSION_LEVEL) -> float:
    """Rational quadratic kernel regression estimate."""
    if bar_index < lookback:
        return closes[bar_index]

    weights     = np.zeros(lookback)
    sum_weights = 0.0
    estimate    = 0.0

    for i in range(lookback):
        idx = bar_index - i
        if idx < 0:
            break
        w            = (1 + (i ** 2) / (2 * weight * level ** 2)) ** (-weight)
        weights[i]   = w
        sum_weights += w
        estimate    += closes[idx] * w

    return estimate / sum_weights if sum_weights > 0 else closes[bar_index]


def kernel_direction(closes: np.ndarray, bar_index: int) -> int:
    """Returns 1 if kernel trending up, -1 if down."""
    if bar_index < 2:
        return 0
    k_now  = rational_quadratic_kernel(closes, bar_index)
    k_prev = rational_quadratic_kernel(closes, bar_index - 1)
    if k_now > k_prev:
        return 1
    elif k_now < k_prev:
        return -1
    return 0


# ══════════════════════════════════════
# FILTERS
# ══════════════════════════════════════

def volatility_filter(closes: np.ndarray, bar_index: int,
                      period: int = 1) -> bool:
    """Pass if recent volatility is normal (not extreme spike)."""
    if bar_index < 20:
        return True
    recent_range = np.std(closes[bar_index-20:bar_index])
    return recent_range > 0


def regime_filter(closes: np.ndarray, bar_index: int,
                  threshold: float = REGIME_THRESHOLD) -> bool:
    """
    Regime filter — avoids trading in choppy/ranging markets.
    Uses normalized slope of kernel estimate.
    """
    if bar_index < LOOKBACK_WINDOW + 2:
        return True

    k1 = rational_quadratic_kernel(closes, bar_index)
    k2 = rational_quadratic_kernel(closes, bar_index - 2)
    slope = (k1 - k2) / 2.0

    # Normalize by recent volatility
    vol = np.std(closes[max(0, bar_index-20):bar_index])
    if vol == 0:
        return True

    normalized_slope = slope / vol
    return normalized_slope > threshold or normalized_slope < -threshold


# ══════════════════════════════════════
# DYNAMIC SL ENGINE
# ══════════════════════════════════════

def get_session_factor() -> tuple:
    hour = datetime.now(timezone.utc).hour
    if 12 <= hour < 16:
        return 0.85, "London/NY Overlap 🔥 (Peak Volume)"
    elif 7 <= hour < 10:
        return 0.90, "London Open 🇬🇧 (High Volume)"
    elif 13 <= hour < 17:
        return 0.90, "New York Open 🇺🇸 (High Volume)"
    elif 0 <= hour < 7:
        return 1.20, "Asian Session 🌏 (Low Volume)"
    else:
        return 1.00, "Off-Peak Hours"


def calculate_dynamic_sl(atr_value: float) -> tuple:
    """Dynamic SL using ATR + session volume for 1H timeframe."""
    atr_pips     = atr_value / PIP_SIZE
    tf_base      = 20  # 1H base
    sess_mult, sess_name = get_session_factor()

    raw_sl   = (tf_base * 0.40) + (atr_pips * 0.60)
    final_sl = round(raw_sl * sess_mult)
    final_sl = max(10, min(final_sl, 40))  # 1H clamp

    breakdown = {
        "tf_base_pips" : tf_base,
        "atr_pips"     : round(atr_pips, 1),
        "session_mult" : sess_mult,
        "session_name" : sess_name,
        "final_sl_pips": final_sl,
    }
    return final_sl, breakdown


def calculate_levels(signal: str, entry: float, sl_pips: float) -> tuple:
    tp1_pips = round(sl_pips * TP1_RATIO)
    tp2_pips = round(sl_pips * TP2_RATIO)
    tp3_pips = round(sl_pips * TP3_RATIO)
    p = PIP_SIZE

    if signal == "BUY":
        sl  = round(entry - sl_pips  * p, 5)
        tp1 = round(entry + tp1_pips * p, 5)
        tp2 = round(entry + tp2_pips * p, 5)
        tp3 = round(entry + tp3_pips * p, 5)
    else:
        sl  = round(entry + sl_pips  * p, 5)
        tp1 = round(entry - tp1_pips * p, 5)
        tp2 = round(entry - tp2_pips * p, 5)
        tp3 = round(entry - tp3_pips * p, 5)

    return sl, tp1, tp2, tp3, tp1_pips, tp2_pips, tp3_pips


# ══════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════

def format_message(signal, entry, sl, tp1, tp2, tp3,
                   sl_pips, tp1_pips, tp2_pips, tp3_pips,
                   breakdown, time_str) -> str:

    emoji  = "🟢" if signal == "BUY" else "🔴"
    direct = "BUY  📈" if signal == "BUY" else "SELL 📉"

    rr1 = round(tp1_pips / sl_pips, 1)
    rr2 = round(tp2_pips / sl_pips, 1)
    rr3 = round(tp3_pips / sl_pips, 1)

    return (
        f"{emoji} *OLYMPUS SIGNAL — GBP/USD*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *Direction:* {direct}\n"
        f"⏱ *Timeframe:* ⏱ 1 Hour\n"
        f"🕐 *Time:* {time_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 *Entry:* `{entry}`\n"
        f"🔴 *Stop Loss:* `{sl}`  (-{sl_pips} pips)\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ *TP1:* `{tp1}`  (+{tp1_pips} pips)  RR {rr1}:1\n"
        f"✅ *TP2:* `{tp2}`  (+{tp2_pips} pips)  RR {rr2}:1\n"
        f"✅ *TP3:* `{tp3}`  (+{tp3_pips} pips)  RR {rr3}:1\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📐 *Dynamic SL Breakdown:*\n"
        f"  · TF Base: {breakdown['tf_base_pips']} pips\n"
        f"  · ATR: {breakdown['atr_pips']} pips\n"
        f"  · {breakdown['session_name']}\n"
        f"  · Volume Mult: {breakdown['session_mult']}×\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⚙️ _Lorentzian KNN · Kernel ML · 1H_"
    )


def send_telegram(message: str) -> dict:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    return requests.post(url, json={
        "chat_id"   : CHAT_ID,
        "text"      : message,
        "parse_mode": "Markdown"
    }).json()


# ══════════════════════════════════════
# MAIN SIGNAL ENGINE
# ══════════════════════════════════════

# Track last signal to avoid duplicates
last_signal = {"direction": None, "time": None}


def run_signal_scan() -> dict:
    """
    Full signal scan pipeline:
    1. Fetch OANDA candles
    2. Calculate features
    3. Run KNN classifier
    4. Apply kernel + filters
    5. Send Telegram if signal found
    """
    global last_signal

    print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] Running signal scan...")

    # 1. Fetch data
    candles = fetch_candles(CANDLE_COUNT)
    if len(candles) < 100:
        return {"status": "error", "reason": "Not enough candle data"}

    data   = parse_candles(candles)
    closes = data["close"]
    highs  = data["high"]
    lows   = data["low"]
    bar    = len(closes) - 1  # latest complete bar

    # 2. Build features
    features = build_features(data)

    # 3. KNN signal
    knn_signal = knn_classify(features, closes, bar)

    # 4. Kernel direction
    kernel_dir = kernel_direction(closes, bar)

    # 5. Filters
    vol_ok    = volatility_filter(closes, bar)
    regime_ok = regime_filter(closes, bar)

    # 6. Combine — all must agree
    final_signal = None
    if knn_signal == 1 and kernel_dir == 1 and vol_ok and regime_ok:
        final_signal = "BUY"
    elif knn_signal == -1 and kernel_dir == -1 and vol_ok and regime_ok:
        final_signal = "SELL"

    bar_time = data["time"][bar][:16].replace("T", " ") + " UTC"

    result = {
        "bar_time"    : bar_time,
        "close"       : closes[bar],
        "knn_signal"  : knn_signal,
        "kernel_dir"  : kernel_dir,
        "vol_filter"  : vol_ok,
        "regime_filter": regime_ok,
        "final_signal": final_signal,
    }

    # 7. Send alert if new signal
    if final_signal:
        # Avoid duplicate alerts on same bar
        if last_signal["time"] == bar_time and last_signal["direction"] == final_signal:
            result["status"] = "duplicate — skipped"
            return result

        entry  = closes[bar]
        atr    = calc_atr(highs, lows, closes)
        sl_pips, breakdown = calculate_dynamic_sl(atr[bar])
        sl, tp1, tp2, tp3, tp1_pips, tp2_pips, tp3_pips = calculate_levels(
            final_signal, entry, sl_pips
        )

        message = format_message(
            final_signal, entry,
            sl, tp1, tp2, tp3,
            sl_pips, tp1_pips, tp2_pips, tp3_pips,
            breakdown, bar_time
        )
        send_telegram(message)

        last_signal = {"direction": final_signal, "time": bar_time}
        result["status"]  = "signal sent ✅"
        result["sl_pips"] = sl_pips
        print(f"  ✅ {final_signal} signal sent @ {entry} | SL: {sl_pips}p")
    else:
        result["status"] = "no signal"
        print(f"  ➖ No signal | KNN: {knn_signal} | Kernel: {kernel_dir} | Vol: {vol_ok} | Regime: {regime_ok}")

    return result


# ══════════════════════════════════════
# SCHEDULER — runs at top of every hour
# ══════════════════════════════════════

def start_scheduler():
    """Run signal scan at the close of every 1H candle."""
    import threading

    def scheduler_loop():
        while True:
            now     = datetime.now(timezone.utc)
            # Wait until 5 seconds after the next hour
            # (gives OANDA time to close the candle)
            minutes_remaining = 59 - now.minute
            seconds_remaining = 60 - now.second + (minutes_remaining * 60) + 5
            print(f"  ⏳ Next scan in {minutes_remaining}m {60 - now.second}s")
            time.sleep(seconds_remaining)
            try:
                run_signal_scan()
            except Exception as e:
                print(f"  ❌ Scan error: {e}")

    thread = threading.Thread(target=scheduler_loop, daemon=True)
    thread.start()
    print("  📡 Scheduler started — scanning every 1H candle close")


# ══════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════

@app.route("/scan", methods=["GET"])
def manual_scan():
    """Manually trigger a signal scan."""
    try:
        result = run_signal_scan()
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/status", methods=["GET"])
def status():
    """Check bot status and last signal."""
    return jsonify({
        "status"      : "running ✅",
        "instrument"  : "GBP/USD",
        "timeframe"   : "1H",
        "last_signal" : last_signal,
        "scan_time"   : "Every 1H candle close",
        "engine"      : "Lorentzian KNN + Kernel Regression",
        "features"    : ["RSI(14)", "WT(10,11)", "CCI(20)", "ADX(20)", "RSI(9)"],
        "filters"     : ["Volatility", "Regime(-0.1)"],
    }), 200


@app.route("/", methods=["GET"])
def home():
    return (
        "<h2>GBP/USD Olympus Signal Bot v4 ✅</h2>"
        "<p>Fully independent — no TradingView required</p>"
        "<p>Engine: Lorentzian KNN + Kernel Regression</p>"
        "<p>Scanning: Every 1H candle close (OANDA)</p>"
        "<hr>"
        "<a href='/scan'>/scan</a> — trigger manual scan<br>"
        "<a href='/status'>/status</a> — bot status + last signal"
    ), 200


# ══════════════════════════════════════
# STARTUP
# ══════════════════════════════════════

# Start the hourly scheduler when app loads
start_scheduler()

if __name__ == "__main__":
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  GBP/USD Olympus Signal Bot v4")
    print("  Engine      : Lorentzian KNN Classifier")
    print("  Data        : OANDA H1 candles")
    print("  Scan        : Every 1H candle close")
    print("  Alerts      : Telegram")
    print("  Manual scan : http://localhost:5000/scan")
    print("  Status      : http://localhost:5000/status")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    app.run(host="0.0.0.0", port=5000, debug=False)
