from concurrent.futures import ThreadPoolExecutor

from .api import _binance_get, _futures_get
from .indicators import _rsi14, _macd, _bollinger, _ema_cross, _atr

STABLECOINS = {
    "USDC", "BUSD", "TUSD", "FDUSD", "USDP", "DAI", "FRAX", "LUSD", "SUSD",
    "USDD", "GUSD", "PYUSD", "USD1", "CUSD", "CEUR", "AGEUR", "PAXG", "XAUT",
    "EURT", "BKRW", "IDRT", "BIDR", "BVND", "VAI", "USTC", "USDN",
}

_SORT_KEYS = {
    "volume":  lambda x: float(x["quoteVolume"]),
    "gainers": lambda x: float(x["priceChangePercent"]),
    "losers":  lambda x: -float(x["priceChangePercent"]),
    "change":  lambda x: abs(float(x["priceChangePercent"])),
    "trades":  lambda x: int(x["count"]),
}

_HTF_STACK = {
    "15m": ["1h", "4h"],
    "1h":  ["4h", "1d"],
    "4h":  ["1d"],
    "1d":  [],
}


def _is_stable_pair(symbol: str, quote_asset: str) -> bool:
    base = symbol[: -len(quote_asset)]
    return base in STABLECOINS


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


def get_symbol_klines(symbol, interval="1h", limit=48, confirm_higher_tf=True):
    """Fetch candles and compute RSI-14, MACD, Bollinger Bands, EMA 9/21, ATR, momentum, volume trend."""
    limit = max(limit, 48)
    htf_intervals = _HTF_STACK.get(interval, []) if confirm_higher_tf else []

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
        ltf_bullish   = bool(cross and cross["bullish"])
        stack         = []
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
    """Bulk-scan all USDT perpetual futures and rank by funding rate."""
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
        "funding_asc":  lambda x:  x["funding_rate_pct"],
        "funding_desc": lambda x: -x["funding_rate_pct"],
        "funding_abs":  lambda x: -abs(x["funding_rate_pct"]),
    }
    results.sort(key=sort_keys.get(sort_by, sort_keys["funding_asc"]))

    return {"symbols": results[:top_n], "total_scanned": len(results), "sorted_by": sort_by}


def get_futures_data(symbol):
    """Fetch funding rate, open interest trend, and long/short ratio for a perpetual futures symbol."""
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
