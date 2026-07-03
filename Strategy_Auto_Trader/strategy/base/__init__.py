"""Protocol base classes for named strategy pairs.

Any named strategy must supply one class satisfying EntryStrategyProtocol
and one satisfying ExitStrategyProtocol.  See strategy/registry.py for the
mapping from name to (entry_class, exit_class).
"""

from .protocols import EntryStrategyProtocol, ExitStrategyProtocol

__all__ = ["EntryStrategyProtocol", "ExitStrategyProtocol"]
