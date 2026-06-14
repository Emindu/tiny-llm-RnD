import time
from datetime import datetime

from .api import _binance_get
from .config import LOG_FILE
from .tools import _filter_tickers
from .core import run_agent, DRILLDOWN_TOOLS


def build_signal_query() -> str:
    """Pre-fetch all three tiers, return a query with candidates embedded."""
    tickers = _binance_get("/api/v3/ticker/24hr")
    large = _filter_tickers(tickers, sort_by="gainers", min_volume_M=50,              top_n=10)
    mid   = _filter_tickers(tickers, sort_by="gainers", min_volume_M=2,  max_volume_M=50, top_n=15)
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


def _write_log(text: str):
    with open(LOG_FILE, "a") as f:
        f.write(text + "\n")


def run_continuous(interval_minutes: int = 5):
    print(f"\nContinuous Signal Scanner — gemma4:e4b")
    print(f"Interval : every {interval_minutes} min")
    print(f"Log file : {LOG_FILE}")
    print("Press Ctrl+C to stop\n")

    scan = 0

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
