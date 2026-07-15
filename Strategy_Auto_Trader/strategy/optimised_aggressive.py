"""Optimised-aggressive — same signal/exit logic as `optimised`, no flip-guard.

What this strategy is trying to do
------------------------------------
Identical entry scoring, thresholds, vetoes and exit rules to
[[optimised]] (see that module's docstring for the journal analysis behind
the weights). The only difference: `require_flip_entry = False`, so a BUY
signal is acted on even when the previous bar was already BUY. `optimised`
only enters on a non-BUY -> BUY transition (see
quant_hmm/consolidated_engine.py's transition guard), which means it can
sit flat for long stretches while the score stays above threshold. This
variant re-enters on every qualifying bar instead, trading the same edge
with more market exposure.

Entry
-----
Same as `optimised`: HMM (2.0) + RSI (1.0) + SMA200 (3.0) + trend (2.0) +
volume (1.0), buy threshold 6.0, sell -4.5, RSI>70 and regime_signal<=0
vetoes. Flip-guard disabled.

Exit
----
Identical to `optimised` (8% stop, 30% target, vol-scaled trailing stop).

Known weaknesses: more time in market means more exposure to whipsaws
between the buy threshold and the flip point; not validated separately
from `optimised` — same in-sample caveat applies.
"""

from __future__ import annotations

from .optimised import OptimisedEntry, OptimisedExit


class OptimisedAggressiveEntry(OptimisedEntry):
    """OptimisedEntry with the flip-guard disabled (re-enters every BUY bar).

    Satisfies EntryStrategyProtocol.
    """

    require_flip_entry: bool = False


class OptimisedAggressiveExit(OptimisedExit):
    """Identical exit rules to OptimisedExit.

    Satisfies ExitStrategyProtocol.
    """
