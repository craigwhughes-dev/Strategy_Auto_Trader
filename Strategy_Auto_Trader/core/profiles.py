"""Per-profile filesystem layout, so independent daemons can share one checkout.

A profile is one isolated trading environment: its own config, state, logs,
lock and broker client id. Without it every daemon started from this checkout
reads `config/overnight_strategy.json` and writes `state/`, which makes running
a real-money instance alongside a paper one impossible — they would share a
ledger, a lock and a broker connection id.

The profile is chosen by `SAT_PROFILE` (or `--profile`). The default profile
deliberately keeps the historical paths (`config/`, `state/`, `logs/`) so an
existing deployment is unaffected; named profiles nest under
`<dir>/profiles/<name>/`.

Watchlists and universe files are NOT per-profile: they are shared reference
data, and the relative paths inside a config still resolve against the
checkout root.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_PROFILE = "default"
PROFILE_ENV_VAR = "SAT_PROFILE"

# Profile names become directory names and are read back from disk when
# checking sibling profiles, so keep them to a boring, safe alphabet.
_VALID_NAME = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


class ProfileError(ValueError):
    """The requested profile name or layout is unusable."""


def validate_profile_name(name: str) -> str:
    """Return `name` if it is a legal profile name, else raise ProfileError."""
    if not _VALID_NAME.match(name or ""):
        raise ProfileError(
            f"invalid profile name {name!r} — use 1-32 characters of "
            f"letters, digits, underscore or hyphen")
    return name


def active_profile_name(explicit: str | None = None) -> str:
    """Profile to use: explicit argument, else $SAT_PROFILE, else "default"."""
    name = (explicit or os.environ.get(PROFILE_ENV_VAR, "") or "").strip()
    return validate_profile_name(name) if name else DEFAULT_PROFILE


@dataclass(frozen=True)
class ProfilePaths:
    """Resolved filesystem layout for one profile."""

    name: str
    root: Path
    config_dir: Path
    state_dir: Path
    logs_dir: Path
    data_dir: Path

    @property
    def is_default(self) -> bool:
        return self.name == DEFAULT_PROFILE

    @property
    def config_path(self) -> Path:
        return self.config_dir / "overnight_strategy.json"

    @property
    def lock_path(self) -> Path:
        return self.state_dir / "daemon.lock"

    @property
    def pid_path(self) -> Path:
        return self.state_dir / "daemon.pid"

    @property
    def commands_dir(self) -> Path:
        return self.state_dir / "commands"

    @property
    def execution_state_path(self) -> Path:
        return self.state_dir / "execution_state.json"


def resolve_profile(name: str | None = None, root: Path | None = None) -> ProfilePaths:
    """Build the layout for `name` (default: the active profile).

    The default profile maps to the historical top-level directories; any
    other name nests under `profiles/<name>/` in each of them so two profiles
    can never write the same ledger, lock or log.
    """
    root = root if root is not None else ROOT
    profile = active_profile_name(name)
    if profile == DEFAULT_PROFILE:
        return ProfilePaths(
            name=profile, root=root,
            config_dir=root / "config",
            state_dir=root / "state",
            logs_dir=root / "logs",
            data_dir=root / "data",
        )
    leaf = Path("profiles") / profile
    return ProfilePaths(
        name=profile, root=root,
        config_dir=root / "config" / leaf,
        state_dir=root / "state" / leaf,
        logs_dir=root / "logs" / leaf,
        # Fetched market data is expensive and profile-independent, so the
        # cache stays shared; only per-run outputs are namespaced below it.
        data_dir=root / "data",
    )


def list_profile_state_dirs(root: Path | None = None) -> dict[str, Path]:
    """Every profile with a state directory on disk, keyed by name.

    Used to detect a sibling profile that would collide on broker client id
    or account — a copied config with an unedited `client_id` is the expected
    mistake, and it silently kicks the other daemon off its IBKR session.
    """
    root = root if root is not None else ROOT
    found: dict[str, Path] = {}
    default_state = root / "state"
    if default_state.is_dir():
        found[DEFAULT_PROFILE] = default_state
    profiles_dir = default_state / "profiles"
    if profiles_dir.is_dir():
        for child in sorted(profiles_dir.iterdir()):
            if child.is_dir() and _VALID_NAME.match(child.name):
                found[child.name] = child
    return found
