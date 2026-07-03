"""Strategy package.

Each strategy file defines a paired Entry + Exit class. Entry controls which
signals to use, how to weight them, and when to buy. Exit controls stop levels,
take-profit targets, and which exit indicators to enable.

Available strategies (see registry.py):
  default       — balanced HMM + RSI + SMA200 vote, 5% stop / 15% target
  conservative  — heavier SMA200 + HMM weighting, tighter 3% stop / 10% target
  trend         — trend-following weights, wide 8% stop / 30% target, vol-trailing stop

To add a new strategy:
  1. Create a file in strategy/ with an Entry class and an Exit class.
  2. Register them in strategy/registry.py under a unique name.
  3. Reference the name via --strategy <name> on the CLI or in watchlist.json.
"""
