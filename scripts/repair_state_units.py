"""One-time repair: normalize pence-unit rows in state/execution_state.json to pounds.

Run ONLY while the live daemon is STOPPED — the daemon holds this file in
memory and will clobber on-disk edits.

Background (2026-07-18): IBKR returned LSE fill prices inconsistently
(HSBA.L in pence, VOD.L in pounds) and nothing normalized them, so the
trade_log mixes units — e.g. HSBA.L's realised pl was logged as 2640.0
(pence) instead of 26.40 (pounds). Execution code now normalizes fills to
pounds at the boundaries (broker/symbols.normalize_fill_price); this script
fixes the rows written before that fix.

Detection: for .L rows, a fill/entry price is pence-scale when it exceeds
POUNDS_CEILING (no UK share in these watchlists trades above £50; pence
values for those same shares are 100x higher). Fields are rescaled /100 and
SELL rows get pl recomputed. A .bak copy is written first.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state" / "execution_state.json"
POUNDS_CEILING = 50.0

PRICE_FIELDS = ("fill_price", "signal_price", "stop_level", "target_level",
                "stop_price", "cost_value")


def _pence_scale(value: float) -> bool:
    return value is not None and value > POUNDS_CEILING


def main() -> int:
    if not STATE.exists():
        print(f"no state file at {STATE}")
        return 1
    backup = STATE.with_suffix(".json.bak_units")
    shutil.copy2(STATE, backup)
    print(f"backup written: {backup}")

    state = json.loads(STATE.read_text(encoding="utf-8"))
    changed = 0

    for ticker, pos in state.get("positions", {}).items():
        if not ticker.upper().endswith(".L"):
            continue
        if _pence_scale(pos.get("fill_price", 0.0)):
            for f in PRICE_FIELDS:
                if pos.get(f):
                    pos[f] = round(pos[f] / 100.0, 6)
            pos["cost_value"] = round(pos["fill_price"] * pos["quantity"], 4)
            changed += 1
            print(f"position {ticker}: rescaled to pounds")

    for row in state.get("trade_log", []):
        ticker = row.get("ticker", "")
        if not ticker.upper().endswith(".L"):
            continue
        if _pence_scale(row.get("fill_price", 0.0)):
            for f in ("fill_price", "signal_price", "stop_price"):
                if row.get(f):
                    row[f] = round(row[f] / 100.0, 6)
            if row.get("action") == "SELL" and row.get("pl") is not None:
                row["pl"] = round(row["pl"] / 100.0, 4)
            changed += 1
            print(f"trade_log {ticker} {row.get('action')} {row.get('date')}: rescaled to pounds"
                  + (f" (pl now {row.get('pl')})" if row.get("action") == "SELL" else ""))

    if changed:
        STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")
        print(f"{changed} record(s) repaired; state written")
    else:
        print("nothing to repair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
