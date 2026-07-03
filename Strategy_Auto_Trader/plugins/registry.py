"""Plugin registry and factory for config-driven dispatch.

REGISTRY maps slot name → {plugin_name: class}.
resolve(slot, name) instantiates a plugin by name with no constructor args;
callers that need engine-specific constructor args (e.g. use_kelly, lookback)
should instantiate the class directly via the class from REGISTRY[slot][name].
"""

from __future__ import annotations

from .context_adjuster import NullAdjuster, SentimentAdjuster
from .exit_rules import StandardExitRules
from .hmm_regime import HMMRegimeModel
from .kelly_sizer import FixedSizer, KellySizer
from .prescreen import NullPrescreen, VolatilityPrescreen
from .quality_gate import NullQualityGate, QualityGatePlugin
from .vote_signal import CompositeSignalGenerator

REGISTRY: dict[str, dict[str, type]] = {
    "regime": {
        "hmm": HMMRegimeModel,
    },
    "signal": {
        "composite": CompositeSignalGenerator,
    },
    "gate": {
        "quality": QualityGatePlugin,
        "none": NullQualityGate,
    },
    "sizer": {
        "kelly": KellySizer,
        "fixed": FixedSizer,
    },
    "exits": {
        "standard": StandardExitRules,
    },
    "adjuster": {
        "sentiment": SentimentAdjuster,
        "none": NullAdjuster,
    },
    "prescreen": {
        "volatility": VolatilityPrescreen,
        "none": NullPrescreen,
    },
}


def resolve(slot: str, name: str) -> object:
    """Instantiate a plugin by slot and name with no constructor arguments.

    For plugins that require engine-specific constructor args (e.g. KellySizer
    needs use_kelly / lookback), retrieve the class via REGISTRY[slot][name]
    and instantiate it directly.

    Raises KeyError for unknown slot or name.
    """
    if slot not in REGISTRY:
        raise KeyError(
            f"Unknown plugin slot '{slot}'. Available: {sorted(REGISTRY)}"
        )
    slot_map = REGISTRY[slot]
    if name not in slot_map:
        raise KeyError(
            f"Unknown plugin '{name}' for slot '{slot}'. Available: {sorted(slot_map)}"
        )
    return slot_map[name]()
