#!/usr/bin/env python3
"""
Binance Symbol Scout Agent
Powered by gemma4:e4b via Ollama — finds the best active trading symbols.
"""
import argparse

from agent.core import run_agent
from agent.scanner import run_continuous

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
