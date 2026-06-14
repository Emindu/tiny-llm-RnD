#!/usr/bin/env python3
"""
Binance Symbol Scout Agent
Powered by gemma4:e4b via Ollama — finds the best active trading symbols.
"""

import argparse
import json
import math
import ssl
import time
import urllib.request
import urllib.parse
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

OLLAMA_URL          = "http://localhost:11434/api/chat"
BINANCE_URL         = "https://api.binance.com"
BINANCE_FUTURES_URL = "https://fapi.binance.com"
MODEL               = "gemma4:e4b"


# ── API helpers ───────────────────────────────────────────────────────────────

# macOS Python often lacks the CA bundle needed for Binance's CDN cert chain;
# fall back to an unverified context only when SSL verification fails.
_NOVERIFY_CTX = ssl._create_unverified_context()


def _http_get(url, timeout=15):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.URLError as exc:
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):
            with urllib.request.urlopen(url, timeout=timeout, context=_NOVERIFY_CTX) as r:
                return json.loads(r.read())
        raise


def _binance_get(path, params=None):
    url = f"{BINANCE_URL}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    return _http_get(url)


def _futures_get(path, params=None):
    url = f"{BINANCE_FUTURES_URL}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    return _http_get(url)


def _ollama_post(payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


# ── Technical indicator helpers ───────────────────────────────────────────────

def _ema_series(values, period):
    """Full EMA series; last element aligns with values[-1]."""
    if len(values) < period:
        return []
    k = 2 / (period + 1)
    result = [sum(values[:period]) / period]
    for v in values[period:]:
        result.append(v * k + result[-1] * (1 - k))
    return result  # length = len(values) - period + 1


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


# ── Tool implementations ──────────────────────────────────────────────────────

STABLECOINS = {
    "USDC", "BUSD", "TUSD", "FDUSD", "USDP", "DAI", "FRAX", "LUSD", "SUSD",
    "USDD", "GUSD", "PYUSD", "USD1", "CUSD", "CEUR", "AGEUR", "PAXG", "XAUT",
    "EURT", "BKRW", "IDRT", "BIDR", "BVND", "VAI", "USTC", "USDN",
}


def _is_stable_pair(symbol: str, quote_asset: str) -> bool:
    base = symbol[: -len(quote_asset)]
    return base in STABLECOINS


_SORT_KEYS = {
    "volume":  lambda x: float(x["quoteVolume"]),
    "gainers": lambda x: float(x["priceChangePercent"]),
    "losers":  lambda x: -float(x["priceChangePercent"]),
    "change":  lambda x: abs(float(x["priceChangePercent"])),
    "trades":  lambda x: int(x["count"]),
}


def _filter_tickers(tickers, quote_asset="USDT", top_n=20, sort_by="volume",
                    min_volume_M=None, max_volume_M=None):
    """Filter and rank pre-fetched tickers for one tier — no HTTP calls."""
    pairs = []
    for t in tickers:
        if not t["symbol"].endswith(quote_asset):
            continue
        if _is_stable_pair(t["symbol"], quote_asset):
            continue
        vol_M = float(t["quoteVolume"]) / 1_000_000
        if vol_M < 0.1:
            continue
        if min_volume_M is not None and vol_M < min_volume_M:
            continue
        if max_volume_M is not None and vol_M > max_volume_M:
            continue
        pairs.append(t)

    pairs.sort(key=_SORT_KEYS.get(sort_by, _SORT_KEYS["volume"]), reverse=True)

    results = [
        {
            "symbol":         t["symbol"],
            "price":          float(t["lastPrice"]),
            "change_24h_pct": round(float(t["priceChangePercent"]), 2),
            "volume_M_usdt":  round(float(t["quoteVolume"]) / 1_000_000, 2),
            "trades_24h":     int(t["count"]),
        }
        for t in pairs[:top_n]
    ]
    tier = (
        "large-cap" if (min_volume_M or 0) >= 50 else
        "mid-cap"   if (max_volume_M or 999) <= 50 else
        "low-cap"   if (max_volume_M or 999) <= 2  else
        "all"
    )
    return {"symbols": results, "total": len(results), "sorted_by": sort_by, "tier": tier}


def fetch_top_symbols(quote_asset="USDT", top_n=20, sort_by="volume",
                      min_volume_M=None, max_volume_M=None):
    """Fetch and rank top Binance symbols by the given criteria.

    Volume tiers (quote_volume in millions USD/day):
      large-cap : min_volume_M=50
      mid-cap   : min_volume_M=2,  max_volume_M=50
      low-cap   : min_volume_M=0.1, max_volume_M=2
    """
    tickers = _binance_get("/api/v3/ticker/24hr")
    return _filter_tickers(tickers, quote_asset, top_n, sort_by, min_volume_M, max_volume_M)


_HTF_STACK = {
    "15m": ["1h", "4h"],
    "1h":  ["4h", "1d"],
    "4h":  ["1d"],
    "1d":  [],
}


def get_symbol_klines(symbol, interval="1h", limit=48, confirm_higher_tf=True):
    """Fetch candles and compute RSI-14, MACD, Bollinger Bands, EMA 9/21, ATR, momentum, volume trend.

    When confirm_higher_tf=True (default), also fetches the full TF stack above the primary interval
    (15m→[1h,4h], 1h→[4h,1d], 4h→[1d]) in parallel and appends:
      tf_stack      — list of {interval, ema_bullish, rsi_14, macd_rising} per confirmation TF
      stack_aligned — True only when every TF in the stack agrees with the primary EMA direction
    Only take HIGH-confidence signals when stack_aligned=true.
    """
    limit = max(limit, 48)  # RSI needs 15, MACD needs 35 — 48 guarantees all indicators compute
    htf_intervals = _HTF_STACK.get(interval, []) if confirm_higher_tf else []

    # fire primary + all HTF fetches in parallel
    with ThreadPoolExecutor(max_workers=1 + len(htf_intervals)) as ex:
        fut_primary = ex.submit(_binance_get, "/api/v3/klines",
                                {"symbol": symbol, "interval": interval, "limit": limit})
        fut_htfs = [
            ex.submit(_binance_get, "/api/v3/klines",
                      {"symbol": symbol, "interval": tf, "limit": limit})
            for tf in htf_intervals
        ]
        raw = fut_primary.result()
        htf_fetched = []
        for tf, fut in zip(htf_intervals, fut_htfs):
            try:
                htf_fetched.append((tf, fut.result(), None))
            except Exception as e:
                htf_fetched.append((tf, None, str(e)))

    closes  = [float(k[4]) for k in raw]
    highs   = [float(k[2]) for k in raw]
    lows    = [float(k[3]) for k in raw]
    volumes = [float(k[5]) for k in raw]

    mid = len(volumes) // 2
    vol_early = sum(volumes[:mid]) / mid if mid else 1
    vol_late  = sum(volumes[mid:]) / (len(volumes) - mid) if (len(volumes) - mid) else 1

    result = {
        "symbol":       symbol,
        "price":        closes[-1],
        "momentum_pct": round((closes[-1] - closes[0]) / closes[0] * 100, 2),
        "rsi_14":       _rsi14(closes),
        "vol_trend":    round(vol_late / vol_early, 2),
    }

    macd = _macd(closes)
    if macd:
        result["macd"] = macd

    bb = _bollinger(closes)
    if bb:
        result["bb"] = bb

    cross = _ema_cross(closes)
    if cross:
        result["ema_9_21"] = cross

    atr_pct = _atr(highs, lows, closes)
    if atr_pct is not None:
        result["atr_pct"] = atr_pct

    if htf_fetched:
        ltf_bullish  = bool(cross and cross["bullish"])
        stack        = []
        stack_aligned = True
        for tf, htf_raw, htf_err in htf_fetched:
            if htf_err:
                stack.append({"interval": tf, "error": htf_err})
                stack_aligned = False
            else:
                try:
                    htf_closes  = [float(k[4]) for k in htf_raw]
                    htf_cross   = _ema_cross(htf_closes)
                    htf_macd    = _macd(htf_closes)
                    htf_bullish = bool(htf_cross and htf_cross["bullish"])
                    if htf_bullish != ltf_bullish:
                        stack_aligned = False
                    stack.append({
                        "interval":    tf,
                        "ema_bullish": htf_bullish,
                        "rsi_14":      _rsi14(htf_closes),
                        "macd_rising": htf_macd["rising"] if htf_macd else None,
                    })
                except Exception as e:
                    stack.append({"interval": tf, "error": str(e)})
                    stack_aligned = False
        result["tf_stack"]      = stack
        result["stack_aligned"] = stack_aligned

    return result


def get_order_book_depth(symbol, limit=10):
    """Return bid/ask spread and liquidity depth for a symbol."""
    book = _binance_get("/api/v3/depth", {"symbol": symbol, "limit": limit})

    best_bid = float(book["bids"][0][0])
    best_ask = float(book["asks"][0][0])
    spread   = round((best_ask - best_bid) / best_bid * 100, 4)

    bid_liq = sum(float(b[0]) * float(b[1]) for b in book["bids"])
    ask_liq = sum(float(a[0]) * float(a[1]) for a in book["asks"])

    return {
        "symbol":            symbol,
        "best_bid":          best_bid,
        "best_ask":          best_ask,
        "spread_pct":        spread,
        "bid_liquidity_usd": round(bid_liq, 2),
        "ask_liquidity_usd": round(ask_liq, 2),
        "buy_sell_ratio":    round(bid_liq / ask_liq, 2) if ask_liq else 1.0,
    }


def get_symbol_detail(symbol):
    """Detailed 24-hr statistics for one symbol."""
    t = _binance_get("/api/v3/ticker/24hr", {"symbol": symbol})
    return {
        "symbol":         t["symbol"],
        "price":          float(t["lastPrice"]),
        "change_24h_pct": round(float(t["priceChangePercent"]), 2),
        "change_24h_abs": float(t["priceChange"]),
        "volume_M_usdt":  round(float(t["quoteVolume"]) / 1_000_000, 2),
        "vwap":           float(t["weightedAvgPrice"]),
        "high_24h":       float(t["highPrice"]),
        "low_24h":        float(t["lowPrice"]),
        "trades_count":   int(t["count"]),
    }


def scan_futures_sentiment(sort_by="funding_asc", top_n=15):
    """Bulk-scan all USDT perpetual futures and rank by funding rate.

    One API call returns sentiment across the entire futures market.
    Use this first to discover crowded-short or crowded-long candidates,
    then call get_futures_data on the top results for OI and L/S ratio detail.
    """
    all_premium = _futures_get("/fapi/v1/premiumIndex")

    results = []
    for p in all_premium:
        symbol = p.get("symbol", "")
        if not symbol.endswith("USDT") or "_" in symbol:
            continue
        funding = float(p.get("lastFundingRate", 0))
        mark    = float(p.get("markPrice", 0))
        if mark < 0.000001:
            continue
        results.append({
            "symbol":           symbol,
            "funding_rate_pct": round(funding * 100, 4),
        })

    sort_keys = {
        "funding_asc":  lambda x:  x["funding_rate_pct"],   # most negative = crowded shorts
        "funding_desc": lambda x: -x["funding_rate_pct"],   # most positive = crowded longs
        "funding_abs":  lambda x: -abs(x["funding_rate_pct"]),  # most extreme either way
    }
    results.sort(key=sort_keys.get(sort_by, sort_keys["funding_asc"]))

    return {"symbols": results[:top_n], "total_scanned": len(results), "sorted_by": sort_by}


def get_futures_data(symbol):
    """Fetch funding rate, open interest trend, and long/short ratio for a perpetual futures symbol.

    Only works for symbols with active USDT perpetual contracts on Binance Futures.
    Each sub-field has its own error key so partial failures don't drop the whole result.
    """
    result = {"symbol": symbol}

    with ThreadPoolExecutor(max_workers=3) as ex:
        fut_premium = ex.submit(_futures_get, "/fapi/v1/premiumIndex", {"symbol": symbol})
        fut_oi      = ex.submit(_futures_get, "/futures/data/openInterestHist",
                                {"symbol": symbol, "period": "1h", "limit": 8})
        fut_ls      = ex.submit(_futures_get, "/futures/data/globalLongShortAccountRatio",
                                {"symbol": symbol, "period": "1h", "limit": 1})

        try:
            premium = fut_premium.result()
            result["funding_rate_pct"] = round(float(premium["lastFundingRate"]) * 100, 4)
        except Exception as e:
            result["funding_rate_error"] = str(e)

        try:
            oi_hist = fut_oi.result()
            if len(oi_hist) >= 2:
                oi_vals = [float(x["sumOpenInterest"]) for x in oi_hist]
                result["oi_change_8h_pct"] = round(
                    (oi_vals[-1] - oi_vals[0]) / oi_vals[0] * 100, 2
                )
        except Exception as e:
            result["oi_hist_error"] = str(e)

        try:
            ls = fut_ls.result()
            if ls:
                result["long_short_ratio"] = round(float(ls[0]["longShortRatio"]), 2)
        except Exception as e:
            result["long_short_error"] = str(e)

    return result


# ── Tool registry ─────────────────────────────────────────────────────────────

TOOL_FN_MAP = {
    "fetch_top_symbols":      fetch_top_symbols,
    "get_symbol_klines":      get_symbol_klines,
    "get_order_book_depth":   get_order_book_depth,
    "get_symbol_detail":      get_symbol_detail,
    "scan_futures_sentiment": scan_futures_sentiment,
    "get_futures_data":       get_futures_data,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_top_symbols",
            "description": (
                "Fetch top Binance trading pairs ranked by volume, gainers, losers, "
                "change magnitude, or trade count. Use min_volume_M / max_volume_M to target "
                "a market-cap tier: large-cap (>=50M), mid-cap (2-50M), low-cap (0.1-2M). "
                "Call this multiple times with different tiers to catch movers across all caps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "quote_asset": {
                        "type": "string",
                        "description": "Quote currency: USDT, BTC, ETH, BNB",
                        "default": "USDT",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "How many symbols to return (max 50)",
                        "default": 20,
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["volume", "gainers", "losers", "change", "trades"],
                        "description": "Ranking criterion",
                    },
                    "min_volume_M": {
                        "type": "number",
                        "description": "Min 24h volume in millions USD. Use 50 for large-cap, 2 for mid-cap, 0.1 for low-cap.",
                    },
                    "max_volume_M": {
                        "type": "number",
                        "description": "Max 24h volume in millions USD. Use 50 to exclude large-caps, 2 to exclude mid-caps.",
                    },
                },
                "required": ["sort_by"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_symbol_klines",
            "description": (
                "Get OHLCV candles + computed indicators: RSI-14, MACD(12,26,9), "
                "Bollinger Bands(20,2), EMA 9/21 crossover, ATR(14), momentum %, volume trend. "
                "By default fetches the full TF confirmation stack in parallel: "
                "15m→[1h,4h], 1h→[4h,1d], 4h→[1d]. "
                "Returns tf_stack (list of {interval, ema_bullish, rsi_14, macd_rising} per TF) "
                "and stack_aligned (true only when ALL confirmation TFs agree with primary EMA direction). "
                "Require stack_aligned=true for HIGH-confidence signals. "
                "Set confirm_higher_tf=false to skip the stack when speed matters more than confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol":   {"type": "string", "description": "e.g. BTCUSDT"},
                    "interval": {
                        "type": "string",
                        "enum": ["15m", "1h", "4h", "1d"],
                        "default": "1h",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of candles (35–100)",
                        "default": 48,
                    },
                    "confirm_higher_tf": {
                        "type": "boolean",
                        "description": "Fetch next higher TF for trend alignment check (default true)",
                        "default": True,
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_book_depth",
            "description": (
                "Get order book spread and liquidity for a symbol. "
                "Tight spread and deep book = easier to enter/exit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "e.g. ETHUSDT"},
                    "limit":  {"type": "integer", "default": 10},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_symbol_detail",
            "description": "Get detailed 24-hr price stats for one specific symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "e.g. SOLUSDT"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_futures_sentiment",
            "description": (
                "Bulk-scan ALL Binance USDT perpetual futures and rank by funding rate. "
                "One call covers the entire market — use this first for any query about "
                "crowded shorts, crowded longs, or extreme funding rates. "
                "sort_by='funding_asc' = most negative funding first (crowded shorts, squeeze candidates). "
                "sort_by='funding_desc' = most positive funding first (crowded longs). "
                "sort_by='funding_abs' = most extreme positioning regardless of direction. "
                "Follow up with get_futures_data on the top results for OI trend and L/S ratio."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sort_by": {
                        "type": "string",
                        "enum": ["funding_asc", "funding_desc", "funding_abs"],
                        "description": "funding_asc=crowded shorts, funding_desc=crowded longs, funding_abs=most extreme",
                        "default": "funding_asc",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of results to return (default 15)",
                        "default": 15,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_futures_data",
            "description": (
                "Fetch perpetual futures market structure for a symbol: "
                "funding rate, open interest (current + 8h trend), and global long/short ratio. "
                "funding_rate_pct < -0.01 = shorts paying longs (bearish crowding / squeeze risk). "
                "oi_change_8h_pct rising + price rising = confirmed trend strength. "
                "oi_change_8h_pct rising + price falling = distribution / bearish divergence. "
                "long_short_ratio > 2.5 = crowded longs (contrarian caution). "
                "Only available for symbols with active USDT perpetuals on Binance Futures."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Futures symbol, e.g. BTCUSDT, ETHUSDT, SOLUSDT",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
]


# ── Agent loop ────────────────────────────────────────────────────────────────

def _prune_messages(messages, max_tool_pairs=6):
    """Drop old tool-call/result pairs beyond max_tool_pairs to cap context growth.

    Always keeps messages[0] (system) and messages[1] (user query).
    Each pair = one assistant message (tool call) + one tool message (result).
    """
    fixed   = messages[:2]                                          # system + user
    history = messages[2:]                                          # tool exchanges
    if len(history) > max_tool_pairs * 2:
        history = history[-(max_tool_pairs * 2):]
    return fixed + history

SYSTEM_PROMPT = """\
You are a crypto market analysis agent with live access to Binance spot and futures data.

Workflow:
1. Discover      — for price/volume queries: fetch_top_symbols across tiers.
                   for sentiment queries (crowded shorts/longs, funding rate): scan_futures_sentiment FIRST.
2. Drill down    — call get_symbol_klines for RSI-14, MACD, Bollinger Bands, EMA 9/21, ATR.
3. Market struct — call get_futures_data on candidates for OI trend and long/short ratio detail.
4. Liquidity     — call get_order_book_depth to confirm spread and depth.
5. Report        — ranked shortlist with specific metric values and clear signal reasoning.

Indicator guide:
- RSI < 30 = oversold; > 70 = overbought
- MACD histogram rising=true = bullish momentum shift; negative histogram = bearish
- Bollinger pct_b > 1.0 = breakout above upper band; < 0 = below lower; bandwidth < 0.05 = squeeze
- EMA 9/21 bullish cross (just_crossed=true, bullish=true) + rising volume_trend = strong entry
- ATR% high = wide stops needed; use for position sizing
- funding_rate_pct < -0.01 = crowded shorts, potential reversal; > 0.03 = crowded longs
- OI rising + price rising = trend confirmed; OI rising + price falling = distribution
- long_short_ratio > 2.5 = crowded long (fade carefully); < 0.5 = crowded short
- stack_aligned=true (all confirmation TFs agree with primary EMA) is required for HIGH confidence; downgrade to MEDIUM if one TF misaligns, LOW if two or more disagree
- tf_stack entries: ema_bullish=true + macd_rising=true on every level = strongest trend confirmation
- partial alignment (e.g. 4h agrees but 1d does not) = valid MEDIUM signal if primary setup is strong

Keep answers concise and data-driven. Always cite the actual metric values.\
"""


def run_agent(user_query: str, quiet: bool = False, tools: list = None,
              history: list = None) -> tuple[str, list]:
    """Run one agent turn. Returns (answer, updated_history).

    history: prior assistant/tool/user messages from previous turns (excludes system prompt).
    Pass the returned history back on the next call to maintain multi-turn context.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_query})

    if not quiet:
        print(f"\n{'─'*60}")
        print(f"  {user_query[:120]}{'…' if len(user_query) > 120 else ''}")
        print(f"{'─'*60}")

    final_answer = "(no response)"
    active_tools = tools if tools is not None else TOOLS

    for step in range(1, 11):
        payload = {
            "model":    MODEL,
            "messages": _prune_messages(messages),
            "tools":    active_tools,
            "stream":   False,
            "options": {
                "num_ctx":     16384,
                "num_predict": 2048,
            },
        }

        try:
            data = _ollama_post(payload)
        except Exception as e:
            print(f"[error] Ollama request failed: {e}")
            break

        msg = data.get("message", {})

        if not quiet and msg.get("thinking"):
            snippet = msg["thinking"][:600]
            print(f"\n[thinking] {snippet}{'…' if len(msg['thinking']) > 600 else ''}")

        tool_calls = msg.get("tool_calls", [])

        if not tool_calls:
            final_answer = msg.get("content", "(no response)")
            if not quiet:
                print(f"\n{final_answer}\n")
            break

        messages.append({
            "role":       "assistant",
            "content":    msg.get("content", ""),
            "tool_calls": tool_calls,
        })

        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args = tc["function"]["arguments"]
            if isinstance(fn_args, str):
                fn_args = json.loads(fn_args)

            if not quiet:
                print(f"\n  → {fn_name}({json.dumps(fn_args)})")

            fn = TOOL_FN_MAP.get(fn_name)
            if fn:
                try:
                    result = fn(**fn_args)
                except Exception as e:
                    result = {"error": str(e)}
            else:
                result = {"error": f"unknown tool: {fn_name}"}

            result_str = json.dumps(result)
            if not quiet:
                preview = result_str[:500] + "…" if len(result_str) > 500 else result_str
                print(f"  ← {preview}")

            messages.append({
                "role":    "tool",
                "name":    fn_name,
                "content": result_str,
            })
    else:
        print("[agent] reached max iterations without a final answer.")

    if not quiet:
        print(f"{'─'*60}\n")

    # Return everything after the system prompt as history for the next turn
    return final_answer, messages[1:]


# ── Continuous signal scanner ─────────────────────────────────────────────────

# fetch_top_symbols excluded so the model drills down on pre-fetched candidates only
DRILLDOWN_TOOLS = [t for t in TOOLS if t["function"]["name"] != "fetch_top_symbols"]


def build_signal_query() -> str:
    """Pre-fetch all three tiers, return a query with candidates embedded."""
    # one HTTP call for all tickers; three in-memory filter passes instead of three fetches
    tickers = _binance_get("/api/v3/ticker/24hr")
    large = _filter_tickers(tickers, sort_by="gainers", min_volume_M=50,             top_n=10)
    mid   = _filter_tickers(tickers, sort_by="gainers", min_volume_M=2,   max_volume_M=50, top_n=15)
    low   = _filter_tickers(tickers, sort_by="gainers", min_volume_M=0.1, max_volume_M=2,  top_n=15)

    def fmt(symbols):
        return "\n".join(
            f"  {s['symbol']:18} {s['change_24h_pct']:+6.1f}%  vol={s['volume_M_usdt']}M"
            for s in symbols[:10]
        )

    ts = datetime.now().strftime("%H:%M:%S")
    return (
        f"[{ts}] Live Binance candidates — pre-fetched, do NOT call fetch_top_symbols.\n\n"
        f"LARGE-CAP (>50M vol/day):\n{fmt(large['symbols'])}\n\n"
        f"MID-CAP (2–50M vol/day):\n{fmt(mid['symbols'])}\n\n"
        f"LOW-CAP (0.1–2M vol/day):\n{fmt(low['symbols'])}\n\n"
        f"Task: pick the 3 best signals across tiers. "
        f"Call get_symbol_klines on your chosen candidates (RSI, MACD, Bollinger, EMA 9/21, ATR). "
        f"Call get_futures_data on large/mid-cap picks to check funding rate, OI trend, L/S ratio. "
        f"Call get_order_book_depth on your final picks. "
        f"Output per signal: symbol | tier (LARGE/MID/LOW) | type (OVERSOLD/MOMENTUM/BREAKOUT/SQUEEZE) "
        f"| RSI | MACD_hist | BB_pct_b | EMA_cross | funding_rate | OI_8h | confidence (LOW/MEDIUM/HIGH). "
        f"No filler."
    )


LOG_FILE = "signals.log"


def _write_log(text: str):
    with open(LOG_FILE, "a") as f:
        f.write(text + "\n")


def run_continuous(interval_minutes: int = 5):
    print(f"\nContinuous Signal Scanner — gemma4:e4b")
    print(f"Interval : every {interval_minutes} min")
    print(f"Log file : {LOG_FILE}")
    print("Press Ctrl+C to stop\n")

    scan = 0
    seen_signals: dict[str, str] = {}  # symbol → last signal type, for dedup display

    while True:
        scan += 1
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n{'═'*60}")
        print(f"  SCAN #{scan}  —  {ts}")
        print(f"{'═'*60}")

        try:
            query = build_signal_query()
        except Exception as e:
            print(f"[error] pre-fetch failed: {e}")
            time.sleep(30)
            continue

        answer, _ = run_agent(query, quiet=False, tools=DRILLDOWN_TOOLS)

        _write_log(f"\n{'='*60}")
        _write_log(f"SCAN #{scan}  {ts}")
        _write_log(f"{'='*60}")
        _write_log(answer)

        next_ts = datetime.fromtimestamp(
            time.time() + interval_minutes * 60
        ).strftime("%H:%M:%S")
        print(f"\n  Logged to {LOG_FILE}")
        print(f"  Next scan at {next_ts}  (Ctrl+C to stop)")

        try:
            time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            print("\n\nScanner stopped.")
            break


# ── CLI entry point ───────────────────────────────────────────────────────────

EXAMPLES = [
    "What are the top 5 momentum plays right now?",
    "Find the biggest gainers with high volume today",
    "Which USDT pairs have the tightest spreads and best liquidity?",
    "Show me coins that look oversold (low RSI) with rising volume",
    "Compare SOL and AVAX — which has better momentum right now?",
    "Find coins with Bollinger squeeze setups (low bandwidth) ready to break out",
    "Which symbols have bullish MACD crossovers and negative funding rates?",
]


def main():
    parser = argparse.ArgumentParser(description="Binance Symbol Scout Agent")
    parser.add_argument(
        "--watch", action="store_true",
        help="Run continuous signal scanner"
    )
    parser.add_argument(
        "--interval", type=int, default=5,
        help="Scan interval in minutes (default: 5)"
    )
    args = parser.parse_args()

    if args.watch:
        run_continuous(args.interval)
        return

    print("\nBinance Symbol Scout Agent")
    print("Powered by gemma4:e4b via Ollama\n")
    print("Example queries:")
    for i, ex in enumerate(EXAMPLES, 1):
        print(f"  {i}. {ex}")
    print("\nType a number to run an example, ask anything, /reset to start fresh, or 'exit' to quit.\n")

    history: list = []

    while True:
        try:
            raw = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not raw:
            continue
        if raw.lower() in ("exit", "quit", "q"):
            break
        if raw.lower() == "/reset":
            history = []
            print("  [conversation reset]\n")
            continue

        if raw.isdigit() and 1 <= int(raw) <= len(EXAMPLES):
            query = EXAMPLES[int(raw) - 1]
            print(f"You: {query}")
        else:
            query = raw

        _, history = run_agent(query, history=history)


if __name__ == "__main__":
    main()
