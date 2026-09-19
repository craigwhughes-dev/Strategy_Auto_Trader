"""Spliced hourly dataset for the tier-allocation backtest: bridged history + real IBKR hourly.

Per series, real hourly bars are used from their own first available date; before that,
correlated Brownian-bridge bars (see correlated_bridge.py) pinned to real (or chain-linked)
daily closes. Output goes to data_synthetic/hourly_spliced/<NAME>.csv with columns
    bar_start_utc (index), Close, bar_end_utc, source ('real' | 'bridged')
plus _meta.json (splice dates, rescale factors, estimated correlation/scale, seed).

Series -> instrument:
    VIX, VXN   the indices (levels are absolute, never rescaled)
    NASDAQ     EQQQ (LSE, GBP); pre-2005 proxy = QQQ daily, chain-linked to EQQQ
    SP500      IUSA (LSE, GBP); pre-2004 proxy = SPY daily, chain-linked to IUSA
    FTSE       ISF.L; pre-2003-04 proxy = existing synthetic ISF daily, chain-linked
    CASH       CSH2.L; pre-2015-09 = BoE-derived daily accrual (no intraday shape)

Known limitation: the QQQ/SPY proxies are USD prices while EQQQ/IUSA are GBP-quoted, so
pre-splice GBP/USD moves are missing from those two legs (no FX series in the repo).

Run:
    uv run python -m Strategy_Auto_Trader.synthetic_backtest_data.build_intraday_dataset
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.atomic_io import atomic_write_csv, atomic_write_json
from .real_clean import clean_fund_bars, remove_level_shifts
from .correlated_bridge import BridgeParams, accrue_deterministic, bridge_dataset, calibrate_params, estimate_params
from ..core.trading_sessions import LSE, US, Session, bar_end

_ROOT = Path(__file__).resolve().parents[2]
_REAL_DEEP = _ROOT / "data" / "cache" / "ibkr_hourly_deep"
_REAL_INDEX = _ROOT / "data" / "cache" / "ibkr_hourly"
_DAILY = _ROOT / "data" / "cache" / "ibkr_daily"
_SYNTH = _ROOT / "data_synthetic"
_DEFAULT_OUT = _SYNTH / "hourly_spliced"
_PARAM_SINCE = "2010-01-01"  # real, all-series era used to measure correlation and vol scale
_SINCE = pd.Timestamp("1999-03-11")  # first date every daily proxy has a prior close (QQQ starts 1999-03-10)


def _read_real(path: Path, clean: bool = False) -> tuple[pd.Series, dict[str, int]]:
    df = pd.read_csv(path, index_col=0)
    idx = pd.to_datetime(df.index, utc=True)
    close = pd.Series(df["Close"].astype(float).to_numpy(), index=idx).groupby(level=0).last().sort_index()
    return clean_fund_bars(close) if clean else (close, {})


def _daily_from_hourly_csv(path: Path) -> pd.Series:
    """Daily close from a bridged-flat hourly file (one value per date, weekdays only)."""
    df = pd.read_csv(path, index_col=0)
    idx = pd.to_datetime(df.index, utc=True, format="mixed").tz_localize(None).normalize()
    daily = pd.Series(df["Close"].astype(float).to_numpy(), index=idx).groupby(level=0).last()
    return daily[daily.index.dayofweek < 5]


def _daily_ibkr(name: str, fund: bool = False) -> pd.Series:
    """Daily closes. `fund=True` back-adjusts unit changes (IBKR ISF.L daily is ~100x larger before 2004-04-16)."""
    df = pd.read_csv(_DAILY / f"{name}.csv", index_col=0)
    idx = pd.to_datetime(df.index, utc=True).tz_localize(None).normalize()
    daily = pd.Series(df["Close"].astype(float).to_numpy(), index=idx).groupby(level=0).last().sort_index()
    return remove_level_shifts(daily) if fund else daily


def chain_link(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    """primary where it exists; earlier dates from `fallback` scaled to primary's first close."""
    first = primary.index[0]
    common = fallback.index[fallback.index >= first]
    link = primary.iloc[0] / fallback[common[0]]
    return pd.concat([fallback[fallback.index < first] * link, primary])


def _local_daily_close(real: pd.Series, session: Session) -> pd.Series:
    dates = real.index.tz_convert(session.tz).tz_localize(None).normalize()
    return real.groupby(dates).last()


def splice_scale(real_daily: pd.Series, proxy: pd.Series) -> float:
    """Factor putting the proxy on the real instrument's price level (second real day, to skip a partial first day)."""
    for day in real_daily.index[1:]:
        if day in proxy.index:
            return float(real_daily[day] / proxy[day])
    raise ValueError("no common date between real series and proxy")


@dataclass(frozen=True)
class SeriesSpec:
    name: str
    session: Session
    real_path: Path
    rescale: bool
    clean: bool = False  # funds only: the vol indices genuinely gap


SPECS = (
    SeriesSpec("VIX", US, _REAL_INDEX / "INDEX_VIX.csv", False),
    SeriesSpec("VXN", US, _REAL_INDEX / "INDEX_VXN.csv", False),
    SeriesSpec("NASDAQ", LSE, _REAL_DEEP / "EQQQ.csv", True, True),
    SeriesSpec("SP500", LSE, _REAL_DEEP / "IUSA.csv", True, True),
    SeriesSpec("FTSE", LSE, _REAL_DEEP / "ISF.L.csv", True, True),
)


def load_daily_proxies() -> dict[str, pd.Series]:
    return {
        "VIX": _daily_ibkr("INDEX_VIX"),
        "VXN": _daily_ibkr("INDEX_VXN").combine_first(_daily_from_hourly_csv(_SYNTH / "hourly" / "VXN_COMPLETE.csv")),
        "NASDAQ": _daily_ibkr("QQQ", fund=True),
        "SP500": _daily_ibkr("SPY", fund=True),
        "FTSE": chain_link(_daily_ibkr("ISF.L", fund=True), _daily_from_hourly_csv(_SYNTH / "hourly" / "ISF.L.csv")),
    }


def _cash_daily(calendar: pd.DatetimeIndex) -> pd.Series:
    df = pd.read_csv(_ROOT / "data" / "cache" / "csh2_daily_returns_extended.csv", index_col=0, parse_dates=True)
    return df["close"].reindex(df.index.union(calendar)).ffill().reindex(calendar)


def _finish(bridged: pd.DataFrame, real: pd.Series, session: Session) -> pd.DataFrame:
    real_frame = pd.DataFrame({"Close": real.to_numpy(), "bar_end": bar_end(real.index, session)}, index=real.index)
    out = pd.concat([bridged.assign(source="bridged"), real_frame.assign(source="real")])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index.name = "bar_start_utc"
    return out.rename(columns={"bar_end": "bar_end_utc"})


def build(seed: int = 20260919, out_dir: Path = _DEFAULT_OUT) -> dict:
    rng = np.random.default_rng(seed)
    proxies = load_daily_proxies()
    loaded = {s.name: _read_real(s.real_path, s.clean) for s in SPECS}
    real = {n: series for n, (series, _) in loaded.items()}
    cleaning = {n: info for n, (_, info) in loaded.items() if info}
    sessions = {s.name: s.session for s in SPECS}

    target: BridgeParams = estimate_params(real, sessions, since=_PARAM_SINCE)
    real_daily = {s.name: _local_daily_close(real[s.name], s.session) for s in SPECS}
    real_end = {n: d.index[-1] + pd.Timedelta(days=1) for n, d in real_daily.items()}
    params = calibrate_params(target, real_daily, sessions, pd.Timestamp(_PARAM_SINCE), real_end, seed)

    daily, until, factors = {}, {}, {}
    for s in SPECS:
        until[s.name] = real_daily[s.name].index[0]
        factors[s.name] = splice_scale(real_daily[s.name], proxies[s.name]) if s.rescale else 1.0
        daily[s.name] = proxies[s.name] * factors[s.name]

    bridged = bridge_dataset(daily, sessions, params, rng, _SINCE, until)

    cash_real, _ = _read_real(_REAL_DEEP / "CSH2.L.csv")
    cash_real_daily = _local_daily_close(cash_real, LSE)
    lse_calendar = proxies["FTSE"].index[proxies["FTSE"].index >= _SINCE - pd.Timedelta(days=10)]
    cash_daily = _cash_daily(lse_calendar)
    factors["CASH"] = splice_scale(cash_real_daily, cash_daily)
    cash_bridged = accrue_deterministic(cash_daily * factors["CASH"], LSE, _SINCE, cash_real_daily.index[0])

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {s.name: _finish(bridged[s.name], real[s.name], s.session) for s in SPECS}
    result["CASH"] = _finish(cash_bridged, cash_real, LSE)
    for name, df in result.items():
        atomic_write_csv(out_dir / f"{name}.csv", df)

    meta = {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "param_since": _PARAM_SINCE,
        "splice_date": {n: str(d.date()) for n, d in {**until, "CASH": cash_real_daily.index[0]}.items()},
        "rescale_factor": factors,
        "real_data_cleaning": cleaning,
        "corr_names": list(target.names),
        "corr": target.corr.round(4).tolist(),
        "intraday_vol_scale": {n: round(v, 4) for n, v in target.scale.items()},
        "bridge_input_corr": params.corr.round(4).tolist(),
        "bridge_input_scale": {n: round(v, 4) for n, v in params.scale.items()},
        "rows": {n: {"bridged": int((df.source == "bridged").sum()), "real": int((df.source == "real").sum())} for n, df in result.items()},
    }
    atomic_write_json(out_dir / "_meta.json", meta)
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=20260919)
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = p.parse_args()
    meta = build(args.seed, args.out)
    print(json.dumps({k: meta[k] for k in ("splice_date", "rescale_factor", "real_data_cleaning", "intraday_vol_scale", "rows")}, indent=2))
    print("corr", meta["corr_names"])
    for row in meta["corr"]:
        print("  ", [f"{v:+.2f}" for v in row])


if __name__ == "__main__":
    main()
