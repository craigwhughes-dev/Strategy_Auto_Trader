#!/usr/bin/env python3
"""Diagnose the 2023 miss (-5.8% vs S&P +24.2%) from the 26yr synthetic journal.

Reads data_synthetic/journals/synth_26yr.csv. Reports:
- Which tickers were active (opened or closed) in 2023
- P&L breakdown per ticker
- Exit reason distribution for 2023 trades
- Whether large AI/momentum names (NVDA, META, MSFT, TSLA, AAPL) were in top-70
- Score distribution for 2023 entries vs 2022 baseline
"""

from pathlib import Path
import pandas as pd

JOURNAL = Path("data_synthetic/journals/synth_26yr.csv")
TOP_70_TICKERS = {
    # From BACKTEST_LOG.md 2026-09-12 entry
    "HSBA.L", "FLEX", "CLX", "WY", "FCX", "ALW.L", "GD", "PEG", "SNPS", "COST",
    "III.L", "LLY", "MRK", "FE", "LAND.L", "EOG", "SVT.L", "CNP", "BAC", "CSX",
    "BATS.L", "HAL", "COR", "MU", "WEC", "CNA.L", "RTO.L", "A", "EIX", "IP",
    "PPL", "HSIC", "GRMN", "MTB", "SMIN.L", "HSY", "MDT", "GPC", "KO", "BLND.L",
    "NKE", "QCOM", "MKC", "T", "MSI", "BG", "CDNS", "IRM", "DLR", "ROL",
    "AME", "IT", "L", "BBY", "CAH", "TDG", "PSA", "SDR.L", "LMT", "GLW",
    "UPS", "ETR", "BA", "BR", "SBAC", "MCK", "SBUX", "FITB", "CRH", "PM",
}
AI_NAMES = {"NVDA", "META", "MSFT", "TSLA", "AAPL", "AMZN", "GOOGL", "GOOG"}


def main():
    if not JOURNAL.exists():
        print(f"Journal not found: {JOURNAL}")
        return

    df = pd.read_csv(JOURNAL)
    print(f"Total trades in journal: {len(df)}\n")

    # Normalize date cols
    for col in ["entry_date", "exit_date", "date_opened", "date_closed"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Use whichever date columns exist
    open_col = "entry_date" if "entry_date" in df.columns else "date_opened"
    close_col = "exit_date" if "exit_date" in df.columns else "date_closed"
    pnl_col = next((c for c in ["pnl_usd", "pnl", "net_pnl"] if c in df.columns), None)
    score_col = next((c for c in ["entry_score", "score"] if c in df.columns), None)
    exit_col = next((c for c in ["exit_reason", "exit_type"] if c in df.columns), None)
    ticker_col = "ticker" if "ticker" in df.columns else None

    if ticker_col is None:
        print("No ticker column found — check journal schema")
        return

    # 2023 trades: opened or closed during 2023
    in_2023 = df[(df[open_col].dt.year == 2023) | (df[close_col].dt.year == 2023)].copy()
    in_2022 = df[(df[open_col].dt.year == 2022) | (df[close_col].dt.year == 2022)].copy()

    print(f"Trades active in 2023: {len(in_2023)}")
    print(f"Trades active in 2022: {len(in_2022)} (baseline)\n")

    # --- AI/momentum names check ---
    print("=== AI/Momentum Names in Top-70 ===")
    for name in sorted(AI_NAMES):
        in_top70 = name in TOP_70_TICKERS
        in_journal = name in df[ticker_col].values
        print(f"  {name:<8} top-70: {str(in_top70):<6} in journal: {in_journal}")
    print()

    # --- P&L by ticker for 2023 ---
    if pnl_col:
        print("=== 2023 P&L by Ticker (worst to best) ===")
        by_ticker = in_2023.groupby(ticker_col)[pnl_col].agg(["sum", "count", "mean"])
        by_ticker.columns = ["total_pnl", "trades", "avg_pnl"]
        by_ticker = by_ticker.sort_values("total_pnl")
        print(by_ticker.to_string())
        print(f"\n2023 total realized P&L: £{in_2023[pnl_col].sum():,.0f}")
        print(f"2022 total realized P&L: £{in_2022[pnl_col].sum():,.0f}")
        print()

    # --- Exit reason breakdown ---
    if exit_col:
        print("=== 2023 Exit Reasons ===")
        print(in_2023[exit_col].value_counts().to_string())
        print("\n=== 2022 Exit Reasons (baseline) ===")
        print(in_2022[exit_col].value_counts().to_string())
        print()

    # --- Score distribution ---
    if score_col:
        print("=== Entry Score Distribution ===")
        print(f"2023 scores: {in_2023[score_col].describe().to_dict()}")
        print(f"2022 scores: {in_2022[score_col].describe().to_dict()}")
        print()

    # --- VIX-blocked check: were there any 2023 VIX rejections? ---
    vix_col = next((c for c in ["vix_rejected", "n_rejected_vix"] if c in df.columns), None)
    if vix_col:
        print(f"=== 2023 VIX rejections (column={vix_col}) ===")
        print(in_2023[vix_col].sum())
    else:
        print("No VIX rejection column in journal — check if summary row exists")


if __name__ == "__main__":
    main()
