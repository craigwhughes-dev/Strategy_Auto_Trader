"""Charting: backtest equity curves vs actual share price."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates


def plot_backtest(
    close: pd.Series,
    bt: dict,
    ticker: str,
    out_path: Path,
    initial_cash: float = 20_000,
) -> None:
    """Four-panel chart saved to out_path.

    Panel 1: price + SMA overlays + signal background + trailing stop markers.
    Panel 2: strategy equity vs buy-and-hold (rebased to 1.0).
    Panel 3: position — initial_cash when in the market, 0 when flat.
    Panel 4: composite score bar chart.
    """
    detail = bt["detail"]
    if detail.empty:
        print("  No backtest data to chart.")
        return

    start = detail.index[0]
    price_in_window = close.loc[start:]

    # Binary in/out expressed as cash amount
    in_market = (detail["position"] > 0).astype(float) * initial_cash

    fig = plt.figure(figsize=(14, 12))
    fig.patch.set_facecolor("#0f1117")

    gs = gridspec.GridSpec(
        4, 1, figure=fig,
        height_ratios=[3, 2, 1, 1],
        hspace=0.08,
    )

    ax_price  = fig.add_subplot(gs[0])
    ax_equity = fig.add_subplot(gs[1], sharex=ax_price)
    ax_pos    = fig.add_subplot(gs[2], sharex=ax_price)
    ax_score  = fig.add_subplot(gs[3], sharex=ax_price)

    for ax in (ax_price, ax_equity, ax_pos, ax_score):
        _style_ax(ax)

    # ── Panel 1: price + MAs + signal background ──────────────────────────────
    _shade_signals(ax_price, detail)
    ax_price.plot(price_in_window.index, price_in_window.values,
                  color="#e0e0e0", linewidth=1.2, zorder=3, label="Price")

    mom_cols = {
        "sma20":  ("#4fc3f7", "SMA20"),
        "sma50":  ("#ffb74d", "SMA50"),
        "sma200": ("#ef9a9a", "SMA200"),
    }
    if "sma20" in detail.columns:
        for col, (colour, lbl) in mom_cols.items():
            if col in detail.columns:
                ax_price.plot(detail.index, detail[col],
                              color=colour, linewidth=0.9, alpha=0.8, zorder=2, label=lbl)

    ts_exits = detail[detail["sell_reason"].str.startswith("trailing", na=False)].index
    for ts_date in ts_exits:
        ax_price.axvline(ts_date, color="#ff9800", linewidth=1.0, alpha=0.8, zorder=4)
    if len(ts_exits):
        ax_price.axvline(ts_exits[0], color="#ff9800", linewidth=1.0,
                         alpha=0.8, zorder=4, label=f"Trailing stop ({len(ts_exits)}x)")

    ax_price.set_ylabel("Price ($)", color="#cccccc")
    ax_price.set_title(f"{ticker}  —  composite signal backtest",
                       color="#ffffff", fontsize=13, pad=10)
    ax_price.legend(loc="upper left", fontsize=8,
                    facecolor="#1e2130", edgecolor="#444", labelcolor="#cccccc")
    ax_price.tick_params(labelbottom=False)

    # ── Panel 2: equity curves ────────────────────────────────────────────────
    _shade_signals(ax_equity, detail)
    ax_equity.plot(detail.index, detail["strategy_equity"],
                   color="#69f0ae", linewidth=1.4, label="Strategy", zorder=3)
    ax_equity.plot(detail.index, detail["bh_equity"],
                   color="#82b1ff", linewidth=1.4, label="Buy & Hold", zorder=3, alpha=0.85)
    ax_equity.axhline(1.0, color="#555", linewidth=0.6, linestyle="--")

    cfg = bt.get("config", {})
    vm = cfg.get("vol_stop_mult", 0)
    ts = cfg.get("trailing_stop", 0)
    ts_str = (f"vol×{vm}(w={cfg.get('vol_stop_window','?')})" if vm
              else (f"fixed {ts*100:.0f}%" if ts else "off"))
    ps = cfg.get("profit_stop_scale", 0)
    ps_str = (f"  profit_scale={ps}(floor={cfg.get('min_stop_pct',0.05)*100:.0f}%)" if ps else "")
    cfg_label = (
        f"mode={cfg.get('position_mode','?')}  "
        f"sell_thr={cfg.get('in_sell_threshold','?')}(in)/{cfg.get('sell_threshold','?')}(out)  "
        f"trail={ts_str}{ps_str}"
    )
    ax_equity.set_ylabel("Equity (rebased)", color="#cccccc")
    ax_equity.legend(
        title=(
            f"Strategy: {bt['total_return_strategy']*100:+.1f}%  "
            f"Sharpe {bt['sharpe_strategy']:.2f}  "
            f"Sortino {bt.get('sortino_strategy', float('nan')):.2f}\n"
            f"B&H:      {bt['total_return_bh']*100:+.1f}%  "
            f"Sharpe {bt['sharpe_bh']:.2f}  "
            f"Sortino {bt.get('sortino_bh', float('nan')):.2f}\n"
            f"{cfg_label}"
        ),
        title_fontsize=7.5, loc="upper left", fontsize=8,
        facecolor="#1e2130", edgecolor="#444", labelcolor="#cccccc",
    )
    ax_equity.tick_params(labelbottom=False)

    # ── Panel 3: in-market step (£20k or £0) ─────────────────────────────────
    _shade_signals(ax_pos, detail)
    ax_pos.step(in_market.index, in_market.values, where="post",
                color="#ffd54f", linewidth=1.5, zorder=3)
    ax_pos.fill_between(in_market.index, in_market.values, step="post",
                        color="#ffd54f", alpha=0.15, zorder=2)
    ax_pos.set_ylabel(f"In market", color="#cccccc")
    ax_pos.set_ylim(-initial_cash * 0.05, initial_cash * 1.2)
    ax_pos.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"£{int(x):,}" if x > 0 else "£0")
    )
    status = "IN" if in_market.iloc[-1] > 0 else "OUT"
    status_colour = "#69f0ae" if status == "IN" else "#ef9a9a"
    ax_pos.text(0.99, 0.85, f"Currently: {status}",
                transform=ax_pos.transAxes, ha="right", va="top",
                fontsize=8, color=status_colour, fontweight="bold")
    ax_pos.tick_params(labelbottom=False)

    # ── Panel 4: composite score bars ─────────────────────────────────────────
    scores = detail["signal_score"].to_numpy()
    colours = np.where(scores > 0, "#69f0ae", np.where(scores < 0, "#ef9a9a", "#888888"))
    ax_score.bar(detail.index, scores, color=colours, width=1.5, zorder=3)
    ax_score.axhline(0, color="#555", linewidth=0.6)
    ax_score.set_ylabel("Score", color="#cccccc")
    ax_score.set_ylim(min(-1, scores.min()) - 0.5, max(1, scores.max()) + 0.5)

    # ── shared x-axis ─────────────────────────────────────────────────────────
    ax_score.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax_score.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax_score.xaxis.get_majorticklabels(), rotation=30, ha="right", color="#aaaaaa")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Chart saved: {out_path}")


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("#1e2130")
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    ax.yaxis.label.set_color("#cccccc")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.grid(axis="y", color="#2a2d3e", linewidth=0.5, zorder=0)


def _shade_signals(ax: plt.Axes, detail: pd.DataFrame) -> None:
    """Shade background green/red/grey by BUY/SELL/HOLD flag."""
    colour_map = {"BUY": "#1b4d2e", "SELL": "#4d1b1b", "HOLD": "#2a2a2a"}
    flags = detail["gate_flag"]
    prev_flag = None
    seg_start = None

    for date, flag in flags.items():
        if flag != prev_flag:
            if prev_flag is not None:
                ax.axvspan(seg_start, date,
                           facecolor=colour_map.get(prev_flag, "#2a2a2a"),
                           alpha=0.35, zorder=1)
            seg_start = date
            prev_flag = flag

    if prev_flag is not None and seg_start is not None:
        ax.axvspan(seg_start, flags.index[-1],
                   facecolor=colour_map.get(prev_flag, "#2a2a2a"),
                   alpha=0.35, zorder=1)
