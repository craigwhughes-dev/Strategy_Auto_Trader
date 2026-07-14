"""Daily HTML summary report written to the run output directory."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


_FLAG_COLOUR = {"BUY": "#1a7a3f", "SELL": "#a02020", "HOLD": "#5a5a5a"}
_FLAG_TEXT   = {"BUY": "#b9f6ca", "SELL": "#ffcdd2", "HOLD": "#e0e0e0"}
_VOTE_COLOUR = {1: "#1a5c30", 0: "#3a3a3a", -1: "#7a1a1a"}
_VOTE_TEXT   = {1: "#b9f6ca", 0: "#cccccc", -1: "#ffcdd2"}
_VOTE_LABEL  = {1: "&#x25B2; Bull", 0: "&#x25A0; Neutral", -1: "&#x25BC; Bear"}

_HMM_REGIME_COLOURS = {"Bear": "#ef9a9a", "Sideways": "#ffe082", "Bull": "#b9f6ca"}
_STATUS_COLOURS = {"green": "#69f0ae", "amber": "#ffe082", "red": "#ef9a9a", "grey": "#888"}


def _pct(v: float, sign: bool = False) -> str:
    if not np.isfinite(v):
        return "—"
    s = "+" if (sign and v >= 0) else ""
    return f"{s}{v*100:.1f}%"


def _gbp(v: float, sign: bool = False) -> str:
    s = "+" if (sign and v >= 0) else ("-" if v < 0 else "")
    return f"{s}£{abs(v):,.0f}"


def _f2(v: float) -> str:
    return f"{v:.2f}" if np.isfinite(v) else "—"


# ── small HTML-building primitives, shared by every section builder ──────────

def _badge(text: str, bg: str, fg: str, size: str = "2em") -> str:
    return (f'<span style="background:{bg};color:{fg};padding:6px 18px;'
            f'border-radius:6px;font-size:{size};font-weight:bold">{text}</span>')


def _stat_row(label: str, val: str, sub: str = "") -> str:
    sub_html = f'<br><small style="color:#888">{sub}</small>' if sub else ""
    return (f'<tr><td style="padding:5px 12px;color:#aaa">{label}</td>'
            f'<td style="padding:5px 12px;color:#eee;text-align:right">'
            f'{val}{sub_html}</td></tr>')


def _section(title: str, content: str) -> str:
    return (f'<div style="margin:18px 0">'
            f'<h3 style="color:#82b1ff;margin:0 0 8px;font-size:1em;'
            f'text-transform:uppercase;letter-spacing:1px">{title}</h3>'
            f'{content}</div>')


def _table(*rows: str, width: str = "100%") -> str:
    return (f'<table style="width:{width};border-collapse:collapse;'
            f'background:#1e2130;border-radius:8px;overflow:hidden">'
            + "".join(rows) + "</table>")


# ── section builders ──────────────────────────────────────────────────────

def _build_vote_rows_html(votes: dict, current_state_name: str) -> str:
    vote_names = {"markov": f"Markov ({current_state_name})", "rsi": "RSI(14)",
                  "sma200": "SMA 200", "trend": "Trend", "volume": "Volume",
                  "hmm": "HiddenMarkovModel"}
    vote_rows = ""
    for key, v in votes.items():
        bg = _VOTE_COLOUR[v]; fg = _VOTE_TEXT[v]
        vote_rows += (
            f'<tr><td style="padding:5px 12px;color:#ccc">{vote_names.get(key, key)}</td>'
            f'<td style="padding:5px 12px;text-align:right">'
            f'<span style="background:{bg};color:{fg};padding:2px 10px;border-radius:4px">'
            f'{_VOTE_LABEL[v]}</span></td></tr>'
        )
    return vote_rows


def _build_trade_history_rows_html(trades_df: pd.DataFrame) -> str:
    if trades_df.empty:
        return '<tr><td colspan="7" style="padding:10px;color:#666;text-align:center">No trades yet</td></tr>'

    trade_rows = ""
    for idx, row in trades_df.iterrows():
        te   = row["trade_event"]
        te_c = "#69f0ae" if te == "BUY" else "#ef9a9a"
        sr   = row.get("sell_reason", "") or ""
        es   = row.get("effective_stop", None)
        es_s = f"{float(es)*100:.1f}%" if es and str(es) != "" else "—"
        pv   = row.get("portfolio_value", "")
        pv_s = f"£{float(pv):,.0f}" if pv and str(pv) != "" else "—"
        trade_rows += (
            f'<tr style="border-top:1px solid #2a2d3e">'
            f'<td style="padding:5px 10px;color:#aaa">{str(idx)[:10]}</td>'
            f'<td style="padding:5px 10px;color:#ddd">${float(row["close"]):.2f}</td>'
            f'<td style="padding:5px 10px;color:{te_c};font-weight:bold">{te}</td>'
            f'<td style="padding:5px 10px;color:#ddd;text-align:center">{float(row["score"]):+.1f}</td>'
            f'<td style="padding:5px 10px;color:#aaa;font-size:0.85em">{sr[:40] if sr else "—"}</td>'
            f'<td style="padding:5px 10px;color:#aaa;text-align:right">{es_s}</td>'
            f'<td style="padding:5px 10px;color:#ddd;text-align:right">{pv_s}</td>'
            f'</tr>'
        )
    return trade_rows


def _build_stop_description(eff_stop_today: float, vol_stop_mult: float,
                             vol_stop_window: int, trailing_stop: float,
                             bt: dict) -> str:
    if vol_stop_mult > 0 and eff_stop_today > 0:
        stop_desc = f"Vol-scaled: {eff_stop_today*100:.1f}% ({vol_stop_mult}&times; daily vol &times; &radic;{vol_stop_window})"
    elif trailing_stop > 0:
        stop_desc = f"Fixed: {trailing_stop*100:.0f}%"
    else:
        stop_desc = "Off"

    ps = bt.get("config", {}).get("profit_stop_scale", 0)
    mn = bt.get("config", {}).get("min_stop_pct", 0.05)
    if ps:
        stop_desc += f" &nbsp;|&nbsp; Profit scale: {ps} (floor {mn*100:.0f}%)"
    return stop_desc


def _build_stationary_dist_rows_html(stationary: dict) -> str:
    stat_dist_rows = ""
    for sn, sp in stationary.items():
        stat_dist_rows += _stat_row(sn, f"{sp*100:.1f}%")
    return stat_dist_rows


def _build_transition_matrix_html(transition_matrix, state_names: list[str]) -> str:
    tm_html = (
        f'<table style="border-collapse:collapse;font-size:0.85em;color:#ccc">'
        f'<tr><th style="padding:4px 8px"></th>'
    )
    for sn in state_names:
        tm_html += f'<th style="padding:4px 8px;color:#82b1ff">&rarr; {sn}</th>'
    tm_html += "</tr>"
    for i, from_s in enumerate(state_names):
        tm_html += f'<tr><td style="padding:4px 8px;color:#82b1ff">{from_s}</td>'
        for j in range(len(state_names)):
            v = transition_matrix[i, j]
            bold = "font-weight:bold;" if i == j else ""
            tm_html += f'<td style="padding:4px 8px;{bold}text-align:right">{v*100:.1f}%</td>'
        tm_html += "</tr>"
    tm_html += "</table>"
    return tm_html


def _compute_portfolio_pl(bt: dict) -> dict:
    """Strategy vs buy-and-hold P&L summary, with colour-coded display strings."""
    ic = bt.get("initial_cash", 20000)
    fp = bt.get("final_portfolio", ic)
    pl = bt.get("total_pl", 0.0)
    pl_colour = "#69f0ae" if pl >= 0 else "#ef9a9a"

    bh_tr = bt.get("total_return_bh", float("nan"))
    bh_fp = ic * (1 + bh_tr) if np.isfinite(bh_tr) else float("nan")
    bh_pl = bh_fp - ic if np.isfinite(bh_fp) else float("nan")
    bh_pl_colour = "#69f0ae" if np.isfinite(bh_pl) and bh_pl >= 0 else "#ef9a9a"

    return {
        "ic": ic, "fp": fp, "pl": pl, "pl_colour": pl_colour,
        "bh_tr": bh_tr, "bh_fp": bh_fp, "bh_pl": bh_pl, "bh_pl_colour": bh_pl_colour,
    }


def _build_hmm_section_html(hmm: dict | None) -> str:
    if hmm is None:
        return ""

    hr = hmm
    hmm_current = hr["current_regime"]
    hmm_c = _HMM_REGIME_COLOURS.get(hmm_current, "#ccc")

    hmm_regime_rows = ""
    for i, name in enumerate(hr["regime_names"]):
        c = _HMM_REGIME_COLOURS.get(name, "#ccc")
        hmm_regime_rows += (
            f'<tr>'
            f'<td style="padding:5px 12px;color:{c};font-weight:bold">{name}</td>'
            f'<td style="padding:5px 12px;color:#ddd;text-align:right;white-space:nowrap">'
            f'{hr["regime_means"][i]*100:+.3f}%</td>'
            f'<td style="padding:5px 12px;color:#ddd;text-align:right;white-space:nowrap">'
            f'{hr["regime_vols"][i]*100:.3f}%</td>'
            f'<td style="padding:5px 12px;color:#ddd;text-align:right;white-space:nowrap">'
            f'{hr["state_counts"][name]}</td>'
            f'<td style="padding:5px 12px;color:#ddd;text-align:right;white-space:nowrap">'
            f'{hr["stationary_distribution"][i]*100:.1f}%</td>'
            f'</tr>'
        )

    hmm_tm_html = (
        f'<table style="border-collapse:collapse;font-size:0.85em;color:#ccc">'
        f'<tr><th style="padding:4px 8px"></th>'
    )
    for name in hr["regime_names"]:
        hmm_tm_html += f'<th style="padding:4px 8px;color:{_HMM_REGIME_COLOURS.get(name,"#ccc")}">&rarr; {name}</th>'
    hmm_tm_html += "</tr>"
    for i, from_r in enumerate(hr["regime_names"]):
        c = _HMM_REGIME_COLOURS.get(from_r, "#ccc")
        hmm_tm_html += f'<tr><td style="padding:4px 8px;color:{c}">{from_r}</td>'
        for j in range(len(hr["regime_names"])):
            v = hr["transition_matrix"][i, j]
            bold = "font-weight:bold;" if i == j else ""
            hmm_tm_html += f'<td style="padding:4px 8px;{bold}text-align:right">{v*100:.1f}%</td>'
        hmm_tm_html += "</tr>"
    hmm_tm_html += "</table>"

    return f"""
  <!-- HMM analysis -->
  {_section(f"Hidden Markov Model &nbsp;<small style='color:#888;font-weight:normal'>best of {hr['n_seeds']} seeds &middot; {hr['n_converged']} converged</small>",
  f'''
  <div style="text-align:center;margin:10px 0 16px">
    Current HMM state: &nbsp;
    <span style="background:#1e2130;color:{hmm_c};padding:4px 14px;border-radius:5px;
                 font-weight:bold;font-size:1.1em;border:1px solid {hmm_c}">{hmm_current}</span>
  </div>
  <table style="width:100%;border-collapse:collapse;background:#1e2130;border-radius:8px;overflow:hidden">
    <tr>
      <th style="padding:6px 12px;color:#888;text-align:left">Regime</th>
      <th style="padding:6px 12px;color:#888;text-align:right">Mean return</th>
      <th style="padding:6px 12px;color:#888;text-align:right">Daily vol</th>
      <th style="padding:6px 12px;color:#888;text-align:right">Days</th>
      <th style="padding:6px 12px;color:#888;text-align:right">Stat. dist.</th>
    </tr>
    {hmm_regime_rows}
  </table>
  <div style="margin-top:12px;display:flex;gap:16px;flex-wrap:wrap">
    <div style="flex:1;min-width:280px">
      <div style="background:#1e2130;border-radius:8px;padding:10px 14px;overflow-x:auto">{hmm_tm_html}</div>
    </div>
  </div>
  '''
  )}"""


def _status_badge(label: str, colour: str) -> str:
    return (f'<span style="color:{colour};font-size:0.85em;'
            f'padding:2px 8px;background:#0f1117;border-radius:4px;'
            f'border:1px solid {colour}">{label}</span>')


def _build_exit_indicators_section_html(exit_ind: dict | None) -> str:
    if exit_ind is None:
        return ""

    ei = exit_ind
    mc = _STATUS_COLOURS.get(ei.get("macd_status", "grey"), "#888")
    rc = _STATUS_COLOURS.get(ei.get("rsi_status", "grey"), "#888")
    bc = _STATUS_COLOURS.get(ei.get("bb_status", "grey"), "#888")
    ac = _STATUS_COLOURS.get(ei.get("atr_status", "grey"), "#888")

    warn_items = ""
    if ei["warnings"]:
        for w in ei["warnings"]:
            warn_items += f'<div style="color:#ffe082;padding:3px 0">&#x26A0; {w}</div>'
    else:
        warn_items = '<div style="color:#666;padding:3px 0">No warnings</div>'

    bb_avg = ei["bb_width_avg"]
    bb_pct = f' ({ei["bb_width"]/bb_avg*100:.0f}% of avg)' if bb_avg else ""
    atr_r = ei["atr_ratio"]
    atr_str = f'{atr_r:.2f}x avg' if atr_r else "N/A"

    return f"""
  <!-- exit indicators -->
  {_section("Exit indicators",
  f'''<div style="display:flex;gap:16px;flex-wrap:wrap">
    <div style="flex:1;min-width:280px">
      <table style="width:100%;border-collapse:collapse;background:#1e2130;border-radius:8px;overflow:hidden">
        <tr><td colspan="2" style="padding:6px 12px;color:#82b1ff;font-weight:bold;font-size:0.9em">MACD</td></tr>
        <tr><td style="padding:5px 12px;color:#aaa">Histogram</td>
            <td style="padding:5px 12px;color:{mc};text-align:right;font-weight:bold;white-space:nowrap">{ei["macd_histogram"]:+.3f} ({ei.get("macd_hist_pct",0):+.2f}% of price)</td></tr>
        <tr><td style="padding:5px 12px;color:#aaa">Momentum</td>
            <td style="padding:5px 12px;text-align:right">{_status_badge(ei["macd_label"], mc)}</td></tr>
        <tr><td style="padding:5px 12px;color:#aaa">Bearish cross (5d)</td>
            <td style="padding:5px 12px;color:{"#ef9a9a" if ei["macd_bearish_cross"] else "#666"};text-align:right">{"YES" if ei["macd_bearish_cross"] else "no"}</td></tr>
        <tr><td style="padding:5px 12px;color:#aaa">Bullish cross (5d)</td>
            <td style="padding:5px 12px;color:{"#69f0ae" if ei["macd_bullish_cross"] else "#666"};text-align:right">{"YES" if ei["macd_bullish_cross"] else "no"}</td></tr>
      </table>
    </div>
    <div style="flex:1;min-width:280px">
      <table style="width:100%;border-collapse:collapse;background:#1e2130;border-radius:8px;overflow:hidden">
        <tr><td colspan="2" style="padding:6px 12px;color:#82b1ff;font-weight:bold;font-size:0.9em">RSI reversals</td></tr>
        <tr><td style="padding:5px 12px;color:#aaa">RSI status</td>
            <td style="padding:5px 12px;text-align:right">{_status_badge(ei["rsi_status_label"], rc)}</td></tr>
        <tr><td style="padding:5px 12px;color:#aaa">Overbought exit (5d)</td>
            <td style="padding:5px 12px;color:{"#ef9a9a" if ei["rsi_exit_overbought"] else "#666"};text-align:right">{"YES" if ei["rsi_exit_overbought"] else "no"}</td></tr>
        <tr><td style="padding:5px 12px;color:#aaa">Momentum loss (5d)</td>
            <td style="padding:5px 12px;color:{"#ef9a9a" if ei["rsi_momentum_loss"] else "#666"};text-align:right">{"YES" if ei["rsi_momentum_loss"] else "no"}</td></tr>
        <tr><td colspan="2" style="padding:6px 12px;color:#82b1ff;font-weight:bold;font-size:0.9em">Consolidation</td></tr>
        <tr><td style="padding:5px 12px;color:#aaa">Bollinger bands</td>
            <td style="padding:5px 12px;text-align:right;white-space:nowrap"><span style="color:#ddd">{ei["bb_width"]:.4f}{bb_pct}</span><br>{_status_badge(ei["bb_label"], bc)}</td></tr>
        <tr><td style="padding:5px 12px;color:#aaa">ATR</td>
            <td style="padding:5px 12px;text-align:right;white-space:nowrap"><span style="color:#ddd">{atr_str}</span><br>{_status_badge(ei["atr_label"], ac)}</td></tr>
        <tr><td style="padding:5px 12px;color:#aaa">Consolidating</td>
            <td style="padding:5px 12px;color:{"#ffe082" if ei["consolidating"] else "#666"};text-align:right">{"YES" if ei["consolidating"] else "no"}</td></tr>
      </table>
    </div>
  </div>
  <div style="background:#1e2130;border-radius:8px;padding:10px 14px;margin-top:10px">
    <div style="color:#888;font-size:0.85em;margin-bottom:4px">Warnings</div>
    {warn_items}
  </div>'''
  )}"""


def write_daily_summary(
    *,
    ticker: str,
    company_name: str = "",
    company_sector: str = "",
    run_date: date,
    close: pd.Series,
    current_state_name: str,
    markov_sig: float,
    sig: dict,
    mom: dict,
    bt: dict,
    stationary: dict,
    transition_matrix,
    state_names: list[str],
    eff_stop_today: float,
    vol_stop_mult: float,
    vol_stop_window: int,
    trailing_stop: float,
    hmm: dict | None = None,
    exit_ind: dict | None = None,
    out_path: Path = None,
) -> None:
    """Write daily_summary.html to out_path."""

    flag      = sig["flag"]
    score     = sig["score"]
    max_score = sig["max_score"]
    votes     = sig["votes"]

    cur_price = float(close.iloc[-1])
    prev_price = float(close.iloc[-2]) if len(close) >= 2 else cur_price
    day_chg   = (cur_price - prev_price) / prev_price

    detail = bt.get("detail", pd.DataFrame())
    in_market = not detail.empty and float(detail["position"].iloc[-1]) > 0

    # Last 10 trade events
    if not detail.empty:
        trades_df = detail[detail["trade_event"].isin(["BUY", "SELL"])].copy()
        trades_df = trades_df[["close", "trade_event", "score", "sell_reason",
                                "effective_stop", "portfolio_value"]].tail(10)
    else:
        trades_df = pd.DataFrame()

    vote_rows = _build_vote_rows_html(votes, current_state_name)
    trade_rows = _build_trade_history_rows_html(trades_df)
    stop_desc = _build_stop_description(eff_stop_today, vol_stop_mult, vol_stop_window,
                                         trailing_stop, bt)
    stat_dist_rows = _build_stationary_dist_rows_html(stationary)
    tm_html = _build_transition_matrix_html(transition_matrix, state_names)

    if not detail.empty:
        bt_start = str(detail.index[0])[:10]
        bt_end   = str(detail.index[-1])[:10]
        date_range = f"{bt_start} to {bt_end}"
    else:
        date_range = "—"

    pl_data = _compute_portfolio_pl(bt)
    ic, fp, pl, pl_colour = pl_data["ic"], pl_data["fp"], pl_data["pl"], pl_data["pl_colour"]
    bh_tr, bh_fp, bh_pl, bh_pl_colour = (pl_data["bh_tr"], pl_data["bh_fp"],
                                          pl_data["bh_pl"], pl_data["bh_pl_colour"])

    status_txt = "IN MARKET" if in_market else "OUT OF MARKET"
    status_bg  = "#1a5c30" if in_market else "#5a3a1a"
    status_fg  = "#b9f6ca" if in_market else "#ffe082"

    day_chg_c = "#69f0ae" if day_chg >= 0 else "#ef9a9a"

    hmm_html = _build_hmm_section_html(hmm)
    exit_html = _build_exit_indicators_section_html(exit_ind)

    # ── assemble HTML ─────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{ticker} Daily Signal — {run_date}</title>
<style>
  body{{margin:0;padding:20px;background:#0f1117;font-family:system-ui,sans-serif;color:#e0e0e0}}
  h1,h2,h3{{margin:0}}
  a{{color:#82b1ff}}
  small{{font-size:0.8em}}
</style>
</head>
<body>
<div style="max-width:760px;margin:0 auto">

  <!-- header -->
  <div style="display:flex;align-items:center;justify-content:space-between;
              border-bottom:1px solid #2a2d3e;padding-bottom:14px;margin-bottom:20px">
    <div>
      <h1 style="font-size:1.8em;color:#fff">{ticker}</h1>
      <div style="color:#bbb;font-size:0.95em">{company_name if company_name != ticker else ""}</div>
      <div style="color:#888;font-size:0.85em">{company_sector + ' &nbsp;|&nbsp; ' if company_sector else ''}{run_date.strftime("%A, %d %B %Y")}</div>
    </div>
    <div style="text-align:right">
      <div style="font-size:1.6em;color:#fff">${cur_price:,.2f}</div>
      <div style="color:{day_chg_c}">{_pct(day_chg, sign=True)} today</div>
    </div>
  </div>

  <!-- signal banner -->
  <div style="text-align:center;margin:24px 0">
    {_badge(flag, _FLAG_COLOUR[flag], _FLAG_TEXT[flag], "2.4em")}
    <div style="margin-top:10px;color:#aaa">
      Composite score &nbsp;
      <strong style="color:#fff">{score:+.1f} / {max_score}</strong>
      &nbsp;&nbsp;|&nbsp;&nbsp;
      {_badge(status_txt, status_bg, status_fg, "0.85em")}
    </div>
  </div>

  <!-- votes + momentum side by side -->
  <div style="display:flex;gap:16px;flex-wrap:wrap">

    <div style="flex:1;min-width:280px">
      {_section("Signal votes", _table(vote_rows))}
    </div>

    <div style="flex:1;min-width:280px">
      {_section("Momentum indicators", _table(
        _stat_row("RSI(14)", f"{mom['cur_rsi']:.1f}", mom['rsi_label']),
        _stat_row("Price", f"${mom['cur_close']:.2f}"),
        _stat_row("vs SMA 20",
                 f"{'ABOVE' if mom['above_sma20'] else 'BELOW'} ${mom['cur_sma20']:.2f}",
                 f"{_pct(mom['pct_from_sma20'], sign=True)}"),
        _stat_row("vs SMA 50",
                 f"{'ABOVE' if mom['above_sma50'] else 'BELOW'} ${mom['cur_sma50']:.2f}",
                 f"{_pct(mom['pct_from_sma50'], sign=True)}"),
        *(
            [_stat_row("vs SMA 200",
                      f"{'ABOVE' if mom['above_sma200'] else 'BELOW'} ${mom['cur_sma200']:.2f}",
                      f"{_pct(mom['pct_from_sma200'], sign=True)}")]
            if mom.get("cur_sma200") else []
        ),
      ))}
    </div>

  </div>

  <!-- trailing stop -->
  {_section("Trailing stop", f'<div style="background:#1e2130;border-radius:8px;padding:10px 14px;color:#ccc">{stop_desc}</div>')}

  {exit_html}

  <!-- P&L -->
  {_section("P&amp;L simulation &nbsp;<small style='color:#888;font-weight:normal'>{} &middot; £{:,.0f} initial &middot; £{:.0f}/trade</small>".format(date_range, ic, bt.get("transaction_cost", 10)), _table(
    _stat_row("Trades", f"{bt.get('n_buys',0)} buys + {bt.get('n_sells',0)} sells"),
    _stat_row("Transaction costs", f"£{bt.get('total_transaction_costs',0):,.0f}"),
    _stat_row("Strategy final portfolio", f"£{fp:,.2f}"),
    _stat_row("Strategy P&amp;L",
             f'<span style="color:{pl_colour}">{_gbp(pl, sign=True)} ({_pct(pl/ic if ic else 0, sign=True)})</span>'),
    _stat_row("Buy &amp; Hold final",
             f'£{bh_fp:,.2f}' if np.isfinite(bh_fp) else "—"),
    _stat_row("Buy &amp; Hold P&amp;L",
             f'<span style="color:{bh_pl_colour}">{_gbp(bh_pl, sign=True)} ({_pct(bh_tr, sign=True)})</span>'
             if np.isfinite(bh_pl) else "—"),
  ))}

  <!-- backtest summary -->
  {_section(f"Backtest summary &nbsp;<small style='color:#888;font-weight:normal'>{date_range} &middot; walk-forward, no lookahead</small>",
  (lambda rows: f'<table style="width:100%;border-collapse:collapse;background:#1e2130;border-radius:8px;overflow:hidden">{"".join(rows)}</table>')([
    f'<tr>'
    f'<th style="padding:6px 12px;color:#888;text-align:left"></th>'
    f'<th style="padding:6px 12px;color:#69f0ae;text-align:right;white-space:nowrap">Strategy</th>'
    f'<th style="padding:6px 12px;color:#82b1ff;text-align:right;white-space:nowrap">Buy &amp; Hold</th>'
    f'</tr>',
    f'<tr>'
    f'<td style="padding:5px 12px;color:#aaa">Sharpe (ann.)</td>'
    f'<td style="padding:5px 12px;color:#69f0ae;text-align:right;white-space:nowrap">{_f2(bt.get("sharpe_strategy",float("nan")))}</td>'
    f'<td style="padding:5px 12px;color:#82b1ff;text-align:right;white-space:nowrap">{_f2(bt.get("sharpe_bh",float("nan")))}</td>'
    f'</tr>',
    f'<tr>'
    f'<td style="padding:5px 12px;color:#aaa">Sortino (ann.)</td>'
    f'<td style="padding:5px 12px;color:#69f0ae;text-align:right;white-space:nowrap">{_f2(bt.get("sortino_strategy",float("nan")))}</td>'
    f'<td style="padding:5px 12px;color:#82b1ff;text-align:right;white-space:nowrap">{_f2(bt.get("sortino_bh",float("nan")))}</td>'
    f'</tr>',
    f'<tr>'
    f'<td style="padding:5px 12px;color:#aaa">Calmar</td>'
    f'<td style="padding:5px 12px;color:#69f0ae;text-align:right;white-space:nowrap">{_f2(bt.get("calmar_strategy",float("nan")))}</td>'
    f'<td style="padding:5px 12px;color:#82b1ff;text-align:right;white-space:nowrap">{_f2(bt.get("calmar_bh",float("nan")))}</td>'
    f'</tr>',
    f'<tr>'
    f'<td style="padding:5px 12px;color:#aaa">Total return</td>'
    f'<td style="padding:5px 12px;color:#69f0ae;text-align:right;white-space:nowrap">{_pct(bt.get("total_return_strategy",float("nan")), sign=True)}</td>'
    f'<td style="padding:5px 12px;color:#82b1ff;text-align:right;white-space:nowrap">{_pct(bt.get("total_return_bh",float("nan")), sign=True)}</td>'
    f'</tr>',
    f'<tr>'
    f'<td style="padding:5px 12px;color:#aaa">Max drawdown</td>'
    f'<td style="padding:5px 12px;color:#69f0ae;text-align:right;white-space:nowrap">{_pct(bt.get("max_drawdown_strategy",float("nan")))}</td>'
    f'<td style="padding:5px 12px;color:#82b1ff;text-align:right;white-space:nowrap">{_pct(bt.get("max_drawdown_bh",float("nan")))}</td>'
    f'</tr>',
    f'<tr>'
    f'<td style="padding:5px 12px;color:#aaa">Days in market</td>'
    f'<td style="padding:5px 12px;color:#69f0ae;text-align:right;white-space:nowrap">{bt.get("n_active_days",0)} / {bt.get("n_days",0)}</td>'
    f'<td style="padding:5px 12px;color:#888;text-align:right;white-space:nowrap">—</td>'
    f'</tr>',
  ]))}

  <!-- backtest chart -->
  <div style="margin:18px 0">
    <img src="backtest_chart.png"
         style="width:100%;border-radius:8px;display:block"
         alt="Backtest chart">
  </div>

  <!-- regime -->
  <div style="display:flex;gap:16px;flex-wrap:wrap">
    <div style="flex:1;min-width:200px">
      {_section("Current regime", _table(
        _stat_row("State", current_state_name),
        _stat_row("Markov signal", f"{markov_sig:+.3f}"),
      ))}
    </div>
    <div style="flex:1;min-width:280px">
      {_section("Stationary distribution", _table(stat_dist_rows))}
    </div>
    <div style="flex:2;min-width:300px">
      {_section("Transition matrix", f'<div style="background:#1e2130;border-radius:8px;padding:10px 14px;overflow-x:auto">{tm_html}</div>')}
    </div>
  </div>

  {hmm_html}

  <!-- recent trades -->
  {_section("Recent trades (last 10 events)", f'''
  <table style="width:100%;border-collapse:collapse;background:#1e2130;border-radius:8px;overflow:hidden;font-size:0.88em">
    <tr style="color:#888;font-size:0.85em">
      <th style="padding:6px 10px;text-align:left">Date</th>
      <th style="padding:6px 10px;text-align:left">Price</th>
      <th style="padding:6px 10px;text-align:left">Event</th>
      <th style="padding:6px 10px;text-align:center">Score</th>
      <th style="padding:6px 10px;text-align:left">Exit reason</th>
      <th style="padding:6px 10px;text-align:right">Stop</th>
      <th style="padding:6px 10px;text-align:right">Portfolio</th>
    </tr>
    {trade_rows}
  </table>''')}

  <div style="margin-top:28px;border-top:1px solid #2a2d3e;padding-top:12px;
              color:#555;font-size:0.8em;text-align:center">
    Generated {run_date} &nbsp;|&nbsp; {ticker} &nbsp;|&nbsp;
    Markov + momentum composite model &nbsp;|&nbsp;
    Historical backtest — not investment advice
  </div>

</div>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"  Daily summary: {out_path}")
