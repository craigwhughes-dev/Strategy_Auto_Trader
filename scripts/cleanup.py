"""Cleanup script for generated/runtime artifacts (data/, logs/, reports/, caches).

Dry-run by default. Never touches state/ (live daemon depends on it).

Usage:
    uv run python scripts/cleanup.py                      # dry-run, default categories
    uv run python scripts/cleanup.py --execute             # actually delete
    uv run python scripts/cleanup.py --execute --purge-full-scan
    uv run python scripts/cleanup.py --data-retention-days 30
"""
import argparse
import shutil
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Active daemon log; matched by mtime freshness too, but name-pinned as a hard exclude.
ACTIVE_LOG_NAMES = {"daemon_restart.log"}


def _active_daemon_log() -> str | None:
    daemon_pid = ROOT / "state" / "daemon.pid"
    if not daemon_pid.exists():
        return None
    logs_dir = ROOT / "logs"
    daemon_logs = sorted(logs_dir.glob("daemon_*.log"), key=lambda p: p.stat().st_mtime)
    return daemon_logs[-1].name if daemon_logs else None


def find_old_logs() -> list[Path]:
    logs_dir = ROOT / "logs"
    if not logs_dir.exists():
        return []
    active = _active_daemon_log()
    keep = ACTIVE_LOG_NAMES | ({active} if active else set())
    return [p for p in logs_dir.glob("*.log") if p.name not in keep]


def find_old_data_dirs(retention_days: int) -> list[Path]:
    data_dir = ROOT / "data"
    if not data_dir.exists():
        return []
    cutoff = datetime.now() - timedelta(days=retention_days)
    return [
        p for p in data_dir.iterdir()
        if p.is_dir() and datetime.fromtimestamp(p.stat().st_mtime) < cutoff
    ]


def find_full_scan_dir() -> Path | None:
    d = ROOT / "reports" / "full_scan"
    return d if d.exists() else None


def find_cache_dirs() -> list[Path]:
    targets = []
    pytest_cache = ROOT / ".pytest_cache"
    if pytest_cache.exists():
        targets.append(pytest_cache)
    egg_info = ROOT / "Strategy_Auto_Trader.egg-info"
    if egg_info.exists():
        targets.append(egg_info)
    for src_dir in ("Strategy_Auto_Trader", "tests", "scripts"):
        targets.extend((ROOT / src_dir).glob("**/__pycache__"))
    return targets


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def run(execute: bool, retention_days: int, purge_full_scan: bool) -> None:
    groups: list[tuple[str, list[Path]]] = [
        ("old logs", find_old_logs()),
        (f"data/ dirs older than {retention_days}d", find_old_data_dirs(retention_days)),
        ("caches (.pytest_cache, egg-info, __pycache__)", find_cache_dirs()),
    ]
    if purge_full_scan:
        fs = find_full_scan_dir()
        if fs:
            groups.append(("reports/full_scan (raw detail)", [fs]))

    total = 0
    for label, paths in groups:
        if not paths:
            continue
        group_size = sum(_size(p) for p in paths)
        total += group_size
        print(f"\n{label}: {len(paths)} item(s), {_human(group_size)}")
        for p in sorted(paths)[:10]:
            print(f"  {p.relative_to(ROOT)}")
        if len(paths) > 10:
            print(f"  ... and {len(paths) - 10} more")

    print(f"\nTotal reclaimable: {_human(total)}")

    if not execute:
        print("\nDry-run only. Re-run with --execute to delete.")
        return

    for _, paths in groups:
        for p in paths:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
    print("\nDeleted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Actually delete (default: dry-run report only)")
    parser.add_argument("--data-retention-days", type=int, default=7, dest="retention_days",
                         help="Delete data/ run dirs older than this many days (default: 7)")
    parser.add_argument("--purge-full-scan", action="store_true",
                         help="Also delete reports/full_scan/ raw per-ticker detail CSVs (32GB)")
    args = parser.parse_args()
    run(args.execute, args.retention_days, args.purge_full_scan)
