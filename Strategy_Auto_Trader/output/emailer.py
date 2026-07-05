"""Email alerts: trade signals and daily roundup via SMTP."""

from __future__ import annotations

import base64
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.mail.yahoo.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))


def _get_smtp_creds() -> tuple[str, str]:
    user = os.environ.get("SMTP_USER", "")
    password = os.environ.get("SMTP_PASSWORD", "")
    if not user or not password:
        raise RuntimeError(
            "SMTP_USER and SMTP_PASSWORD environment variables not set. "
            "For Yahoo Mail, generate an App Password at: "
            "https://login.yahoo.com/account/security/app-passwords"
        )
    return user, password


def _send(subject: str, html_body: str, to: str | None = None) -> None:
    user, password = _get_smtp_creds()
    to = to or os.environ.get("SMTP_TO") or user

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(user, password)
        server.sendmail(user, [to], msg.as_string())


_SIG_COLOUR = {"BUY": "#1a7a3f", "SELL": "#a02020", "HOLD": "#5a5a5a"}
_SIG_TEXT = {"BUY": "#b9f6ca", "SELL": "#ffcdd2", "HOLD": "#e0e0e0"}


def _embed_chart(html: str, run_dir: Path) -> str:
    """Replace <img src="backtest_chart.png"> with a base64 data URI."""
    chart_path = run_dir / "backtest_chart.png"
    if not chart_path.exists():
        return html
    b64 = base64.b64encode(chart_path.read_bytes()).decode("ascii")
    return html.replace(
        'src="backtest_chart.png"',
        f'src="data:image/png;base64,{b64}"',
    )


def send_trade_alert(result: dict) -> None:
    """Send the full daily summary HTML (with embedded chart) as a trade alert."""
    ticker = result["ticker"]
    signal = result["current_signal"]
    price = result["close"]
    score = result["score"]
    run_dir = Path(result["run_dir"])

    # Read the full daily summary HTML
    summary_path = run_dir / "daily_summary.html"
    if summary_path.exists():
        html = summary_path.read_text(encoding="utf-8")
        html = _embed_chart(html, run_dir)
    else:
        html = (f"<html><body><h1>{ticker} — {signal}</h1>"
                f"<p>Score {score:+.1f}, price ${price:,.2f}</p>"
                f"<p>Daily summary not found in {run_dir}</p></body></html>")

    subject = f"[{signal}] {ticker} at ${price:,.2f}  (score {score:+.1f})"
    _send(subject, html)
    print(f"  Email sent: {subject}")


def send_daily_roundup(results: list[dict], failed: list[dict]) -> None:
    """Send a daily summary email covering all tickers."""
    total = len(results) + len(failed)
    ok = len(results)

    # Sort: trade events first, then by strategy return descending
    sorted_results = sorted(results, key=lambda r: (
        0 if r.get("trade_event") else 1,
        -r.get("strategy_return", 0),
    ))

    # Count signals
    buys = sum(1 for r in results if r.get("current_signal") == "BUY")
    sells = sum(1 for r in results if r.get("current_signal") == "SELL")
    holds = sum(1 for r in results if r.get("current_signal") == "HOLD")
    trade_events = [r for r in results if r.get("trade_event")]
    profitable = sum(1 for r in results if r.get("portfolio_value", 20000) > 20000)
    outperforming = sum(1 for r in results
                        if r.get("strategy_return", 0) > r.get("bh_return", 0))

    rows_html = ""
    for r in sorted_results:
        sig = r.get("current_signal", "?")
        bg = _SIG_COLOUR.get(sig, "#333")
        fg = _SIG_TEXT.get(sig, "#eee")
        te = r.get("trade_event", "")
        te_html = (f'<span style="color:{"#69f0ae" if te == "BUY" else "#ef9a9a"};'
                   f'font-weight:bold"> {te}</span>') if te else ""
        strat_pct = r.get("strategy_return", 0) * 100
        bh_pct = r.get("bh_return", 0) * 100
        pl = r.get("portfolio_value", 20000) - 20000
        pl_colour = "#69f0ae" if pl >= 0 else "#ef9a9a"
        outperf = strat_pct > bh_pct

        rows_html += f"""<tr style="border-top:1px solid #2a2d3e">
          <td style="padding:6px 10px;color:#eee">{r['ticker']}</td>
          <td style="padding:6px 10px;text-align:center">
            <span style="background:{bg};color:{fg};padding:2px 8px;border-radius:4px;font-size:0.85em">{sig}</span>
            {te_html}</td>
          <td style="padding:6px 10px;color:#ddd;text-align:right;white-space:nowrap">${r.get('close',0):,.2f}</td>
          <td style="padding:6px 10px;color:#ddd;text-align:center">{r.get('score',0):+.1f}</td>
          <td style="padding:6px 10px;color:{pl_colour};text-align:right;white-space:nowrap">
            {"+" if pl >= 0 else ""}£{abs(pl):,.0f}</td>
          <td style="padding:6px 10px;color:{'#69f0ae' if outperf else '#ef9a9a'};text-align:right;white-space:nowrap">
            {strat_pct:+.1f}%</td>
          <td style="padding:6px 10px;color:#82b1ff;text-align:right;white-space:nowrap">{bh_pct:+.1f}%</td>
        </tr>"""

    failed_html = ""
    if failed:
        failed_html = """<div style="margin:16px 0">
          <h3 style="color:#ef9a9a;font-size:0.9em;text-transform:uppercase;letter-spacing:1px">Failed tickers</h3>
          <ul style="color:#ef9a9a;font-size:0.9em">"""
        for f in failed:
            failed_html += f'<li>{f["ticker"]}: {f.get("error", "unknown")}</li>'
        failed_html += "</ul></div>"

    trade_event_summary = ""
    if trade_events:
        lines = []
        for r in trade_events:
            te = r["trade_event"]
            c = "#69f0ae" if te == "BUY" else "#ef9a9a"
            lines.append(f'<span style="color:{c};font-weight:bold">{te}</span> {r["ticker"]} '
                         f'at ${r.get("close",0):,.2f}')
        trade_event_summary = (
            '<div style="background:#1a2a1a;border:1px solid #2a4a2a;border-radius:8px;'
            'padding:12px 16px;margin:16px 0">'
            '<strong style="color:#69f0ae">Trade events today:</strong><br>'
            + "<br>".join(lines) + '</div>'
        )

    html = f"""<html><body style="margin:0;padding:20px;background:#0f1117;font-family:system-ui,sans-serif;color:#e0e0e0">
<div style="max-width:760px;margin:0 auto">
  <h1 style="color:#fff;margin:0 0 4px">Daily Roundup</h1>
  <div style="color:#888;margin-bottom:16px">{ok} tickers processed | {len(failed)} failed</div>

  <div style="display:flex;gap:12px;flex-wrap:wrap;margin:16px 0">
    <div style="background:#1e2130;padding:10px 16px;border-radius:8px;flex:1;min-width:100px;text-align:center">
      <div style="color:#69f0ae;font-size:1.4em;font-weight:bold">{buys}</div>
      <div style="color:#888;font-size:0.8em">BUY</div>
    </div>
    <div style="background:#1e2130;padding:10px 16px;border-radius:8px;flex:1;min-width:100px;text-align:center">
      <div style="color:#e0e0e0;font-size:1.4em;font-weight:bold">{holds}</div>
      <div style="color:#888;font-size:0.8em">HOLD</div>
    </div>
    <div style="background:#1e2130;padding:10px 16px;border-radius:8px;flex:1;min-width:100px;text-align:center">
      <div style="color:#ef9a9a;font-size:1.4em;font-weight:bold">{sells}</div>
      <div style="color:#888;font-size:0.8em">SELL</div>
    </div>
    <div style="background:#1e2130;padding:10px 16px;border-radius:8px;flex:1;min-width:100px;text-align:center">
      <div style="color:#69f0ae;font-size:1.4em;font-weight:bold">{profitable}</div>
      <div style="color:#888;font-size:0.8em">Profitable</div>
    </div>
    <div style="background:#1e2130;padding:10px 16px;border-radius:8px;flex:1;min-width:100px;text-align:center">
      <div style="color:#82b1ff;font-size:1.4em;font-weight:bold">{outperforming}</div>
      <div style="color:#888;font-size:0.8em">Beating B&amp;H</div>
    </div>
  </div>

  {trade_event_summary}

  <table style="width:100%;border-collapse:collapse;background:#1e2130;border-radius:8px;overflow:hidden;font-size:0.88em;margin:16px 0">
    <tr style="color:#888;font-size:0.85em">
      <th style="padding:8px 10px;text-align:left">Ticker</th>
      <th style="padding:8px 10px;text-align:center">Signal</th>
      <th style="padding:8px 10px;text-align:right">Price</th>
      <th style="padding:8px 10px;text-align:center">Score</th>
      <th style="padding:8px 10px;text-align:right">Strategy P&amp;L</th>
      <th style="padding:8px 10px;text-align:right">Strategy %</th>
      <th style="padding:8px 10px;text-align:right">B&amp;H %</th>
    </tr>
    {rows_html}
  </table>

  {failed_html}

  <div style="color:#555;font-size:0.8em;margin-top:20px;text-align:center">
    Markov + momentum composite model — not investment advice
  </div>
</div></body></html>"""

    n_events = len(trade_events)
    subject = (f"Roundup: {buys} BUY / {sells} SELL / {holds} HOLD"
               f" | {profitable} profitable | {n_events} trade event{'s' if n_events != 1 else ''}")
    _send(subject, html)
    print(f"  Roundup email sent: {subject}")


def send_reconciliation_alert(discrepancies: list[str]) -> None:
    """Alert that broker account positions disagree with internal state."""
    items = "".join(
        f'<li style="color:#ffcdd2;padding:4px 0">{d}</li>' for d in discrepancies
    )
    html = f"""<html><body style="margin:0;padding:20px;background:#0f1117;font-family:system-ui,sans-serif;color:#e0e0e0">
<div style="max-width:700px;margin:0 auto">
  <h1 style="color:#ef9a9a;margin:0 0 4px">Reconciliation mismatch</h1>
  <div style="color:#888;margin-bottom:16px">IBKR account positions disagree with execution_state.json</div>
  <div style="background:#2a1a1a;border:1px solid #4a2a2a;border-radius:8px;padding:12px 16px;margin:16px 0">
    <ul style="margin:0;padding-left:20px">{items}</ul>
  </div>
  <p style="color:#ddd">New entries are halted until reconciliation passes clean.
  Resolve the discrepancy manually (TWS and/or state/execution_state.json), then
  the next nightly check — or a daemon restart — will re-enable buying.</p>
</div></body></html>"""

    n = len(discrepancies)
    subject = f"RECONCILIATION MISMATCH: {n} discrepanc{'ies' if n != 1 else 'y'} — new entries halted"
    _send(subject, html)
    print(f"  Reconciliation alert sent: {subject}")


def send_portfolio_status(positions: list[dict]) -> None:
    """Send a portfolio status email showing all active trades with P&L since entry."""
    if not positions:
        return

    n = len(positions)
    winners = sum(1 for p in positions if p.get("pl_pct", 0) > 0)
    losers = sum(1 for p in positions if p.get("pl_pct", 0) < 0)

    sorted_pos = sorted(positions, key=lambda p: p.get("pl_pct", 0), reverse=True)

    rows_html = ""
    for p in sorted_pos:
        pl = p.get("pl_pct", 0)
        pl_colour = "#69f0ae" if pl >= 0 else "#ef9a9a"
        signal = p.get("current_signal", "?")
        bg = _SIG_COLOUR.get(signal, "#333")
        fg = _SIG_TEXT.get(signal, "#eee")
        days = p.get("days_held", 0)

        rows_html += f"""<tr style="border-top:1px solid #2a2d3e">
          <td style="padding:6px 10px;color:#eee">{p['ticker']}</td>
          <td style="padding:6px 10px;color:#aaa;white-space:nowrap">{p.get('buy_date','?')}</td>
          <td style="padding:6px 10px;color:#ddd;text-align:right;white-space:nowrap">${p.get('entry_price',0):,.2f}</td>
          <td style="padding:6px 10px;color:#ddd;text-align:right;white-space:nowrap">${p.get('current_price',0):,.2f}</td>
          <td style="padding:6px 10px;color:{pl_colour};text-align:right;font-weight:bold;white-space:nowrap">
            {pl:+.1f}%</td>
          <td style="padding:6px 10px;text-align:center">
            <span style="background:{bg};color:{fg};padding:2px 8px;border-radius:4px;font-size:0.85em">{signal}</span></td>
          <td style="padding:6px 10px;color:#aaa;text-align:right">{days}d</td>
        </tr>"""

    html = f"""<html><body style="margin:0;padding:20px;background:#0f1117;font-family:system-ui,sans-serif;color:#e0e0e0">
<div style="max-width:700px;margin:0 auto">
  <h1 style="color:#fff;margin:0 0 4px">Active Trades</h1>
  <div style="color:#888;margin-bottom:16px">{n} open positions</div>

  <div style="display:flex;gap:12px;flex-wrap:wrap;margin:16px 0">
    <div style="background:#1e2130;padding:10px 16px;border-radius:8px;flex:1;min-width:100px;text-align:center">
      <div style="color:#69f0ae;font-size:1.4em;font-weight:bold">{winners}</div>
      <div style="color:#888;font-size:0.8em">Winning</div>
    </div>
    <div style="background:#1e2130;padding:10px 16px;border-radius:8px;flex:1;min-width:100px;text-align:center">
      <div style="color:#ef9a9a;font-size:1.4em;font-weight:bold">{losers}</div>
      <div style="color:#888;font-size:0.8em">Losing</div>
    </div>
    <div style="background:#1e2130;padding:10px 16px;border-radius:8px;flex:1;min-width:100px;text-align:center">
      <div style="color:#fff;font-size:1.4em;font-weight:bold">{n}</div>
      <div style="color:#888;font-size:0.8em">Total</div>
    </div>
  </div>

  <table style="width:100%;border-collapse:collapse;background:#1e2130;border-radius:8px;overflow:hidden;font-size:0.88em;margin:16px 0">
    <tr style="color:#888;font-size:0.85em">
      <th style="padding:8px 10px;text-align:left">Ticker</th>
      <th style="padding:8px 10px;text-align:left">Entry date</th>
      <th style="padding:8px 10px;text-align:right">Entry price</th>
      <th style="padding:8px 10px;text-align:right">Current</th>
      <th style="padding:8px 10px;text-align:right">P&amp;L %</th>
      <th style="padding:8px 10px;text-align:center">Signal</th>
      <th style="padding:8px 10px;text-align:right">Held</th>
    </tr>
    {rows_html}
  </table>

  <div style="color:#555;font-size:0.8em;margin-top:20px;text-align:center">
    Prices at last close — not investment advice
  </div>
</div></body></html>"""

    subject = f"Active trades: {n} positions | {winners} winning, {losers} losing"
    _send(subject, html)
    print(f"  Portfolio status email sent: {subject}")
