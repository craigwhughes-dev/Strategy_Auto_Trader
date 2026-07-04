"""Run full S&P500 backtest with optimised and trend_follow strategies independently."""

from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def run_strategy(strategy_name: str, tickers: list[str]) -> None:
    """Run batch for a single strategy."""
    watchlist = {
        "defaults": {
            "strategy": strategy_name,
            "initial_cash": 10000,
            "transaction_cost": 10,
            "capital_pot": 10000,
            "max_positions": 50,
            "daily_buy_limit": None,
            "daily_sell_limit": None,
        },
        "tickers": [{"ticker": t} for t in tickers]
    }

    watchlist_path = ROOT / f"config/watchlist_sp500_{strategy_name}.json"
    with open(watchlist_path, 'w') as f:
        json.dump(watchlist, f, indent=2)

    print(f"\n{'='*64}")
    print(f" Running {strategy_name.upper()} strategy on {len(tickers)} S&P 500 tickers")
    print(f"{'='*64}\n")

    cmd = [
        sys.executable, "-m", "Strategy_Auto_Trader.markov_cli.batch",
        "--watchlist", str(watchlist_path),
        "--no-email"
    ]

    subprocess.run(cmd, check=False)
    print(f"\n✓ {strategy_name} complete")

if __name__ == "__main__":
    # Load SP500 tickers
    with open(ROOT / "config/watchlist_sp500.json") as f:
        sp500_config = json.load(f)

    tickers = [t["ticker"] if isinstance(t, dict) else t for t in sp500_config["tickers"]]
    print(f"Loaded {len(tickers)} S&P 500 tickers\n")

    # Run optimised
    run_strategy("optimised", tickers)

    # Run trend_follow
    run_strategy("trend", tickers)

    print("\n" + "="*64)
    print(" Both strategies complete")
    print("="*64)
    print(f"\nResults in: data/journals/live.csv")
