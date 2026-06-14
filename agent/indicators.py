import math


def _ema_series(values, period):
    """Full EMA series; last element aligns with values[-1]."""
    if len(values) < period:
        return []
    k = 2 / (period + 1)
    result = [sum(values[:period]) / period]
    for v in values[period:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def _macd(closes):
    """MACD(12,26,9). histogram.rising=True signals a bullish momentum shift."""
    ema12 = _ema_series(closes, 12)
    ema26 = _ema_series(closes, 26)
    if len(ema12) < 9 or not ema26:
        return None
    offset = len(ema12) - len(ema26)
    macd_line = [ema12[offset + i] - ema26[i] for i in range(len(ema26))]
    signal_line = _ema_series(macd_line, 9)
    if not signal_line:
        return None
    hist = [macd_line[len(macd_line) - len(signal_line) + i] - signal_line[i]
            for i in range(len(signal_line))]
    return {
        "histogram": round(hist[-1], 4),
        "rising":    hist[-1] > hist[-2] if len(hist) >= 2 else None,
    }


def _bollinger(closes, period=20):
    """Bollinger Bands(20,2). pct_b: 0=at lower, 1=at upper, >1=breakout above."""
    if len(closes) < period:
        return None
    window = closes[-period:]
    mid    = sum(window) / period
    std    = math.sqrt(sum((x - mid) ** 2 for x in window) / period)
    upper  = mid + 2 * std
    lower  = mid - 2 * std
    bw     = upper - lower
    return {
        "pct_b":     round((closes[-1] - lower) / bw, 3) if bw > 0 else 0.5,
        "bandwidth": round(bw / mid, 4),
    }


def _ema_cross(closes, fast=9, slow=21):
    """EMA 9/21 crossover. just_crossed=True means it crossed on the last candle."""
    ema_f = _ema_series(closes, fast)
    ema_s = _ema_series(closes, slow)
    if len(ema_f) < 2 or len(ema_s) < 2:
        return None
    curr = ema_f[-1] - ema_s[-1]
    prev = ema_f[-2] - ema_s[-2]
    cross = "bullish" if prev <= 0 < curr else "bearish" if prev >= 0 > curr else None
    return {
        "bullish":      curr > 0,
        "just_crossed": cross is not None,
        "direction":    cross,
    }


def _rsi14(closes):
    """RSI-14 with Wilder's smoothing."""
    if len(closes) < 15:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [max(d, 0) for d in deltas]
    losses = [max(-d, 0) for d in deltas]
    avg_gain = sum(gains[:14]) / 14
    avg_loss = sum(losses[:14]) / 14
    for i in range(14, len(deltas)):
        avg_gain = (avg_gain * 13 + gains[i]) / 14
        avg_loss = (avg_loss * 13 + losses[i]) / 14
    return round(100 - 100 / (1 + avg_gain / avg_loss), 2) if avg_loss else 100.0


def _atr(highs, lows, closes, period=14):
    """ATR(14) with Wilder smoothing. atr_pct normalises volatility to price."""
    if len(closes) < period + 1:
        return None
    tr_list = [
        max(highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]))
        for i in range(1, len(closes))
    ]
    atr = sum(tr_list[:period]) / period
    for tr in tr_list[period:]:
        atr = (atr * (period - 1) + tr) / period
    return round(atr / closes[-1] * 100, 3)
