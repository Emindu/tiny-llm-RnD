import json

from .config import MODEL
from .api import _ollama_post
from .tools import TOOL_FN_MAP, TOOLS

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

# fetch_top_symbols excluded so the model drills down on pre-fetched candidates only
DRILLDOWN_TOOLS = [t for t in TOOLS if t["function"]["name"] != "fetch_top_symbols"]


def _prune_messages(messages, max_tool_pairs=6):
    """Drop old tool-call/result pairs beyond max_tool_pairs to cap context growth."""
    fixed   = messages[:2]
    history = messages[2:]
    if len(history) > max_tool_pairs * 2:
        history = history[-(max_tool_pairs * 2):]
    return fixed + history


def run_agent(user_query: str, quiet: bool = False, tools: list = None,
              history: list = None) -> tuple[str, list]:
    """Run one agent turn. Returns (answer, updated_history).

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

    return final_answer, messages[1:]
