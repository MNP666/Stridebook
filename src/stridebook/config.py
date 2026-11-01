"""Where the data folder is, and the personal settings read from it.

Phase 1 writes nothing to disk; the data folder is only read for settings.toml.
"""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ENV_VAR = "STRIDEBOOK_DATA"
DEFAULT_DATA_DIR = Path("data")
SETTINGS_FILE = "settings.toml"


@dataclass(frozen=True)
class HrZone:
    name: str
    min: int  # bpm, inclusive
    max: int  # bpm, exclusive


def _default_zones() -> tuple[HrZone, ...]:
    return (
        HrZone("Z1 Recovery", 0, 125),
        HrZone("Z2 Aerobic", 125, 145),
        HrZone("Z3 Tempo", 145, 160),
        HrZone("Z4 Threshold", 160, 172),
        HrZone("Z5 VO2max", 172, 999),
    )


@dataclass(frozen=True)
class Settings:
    max_hr: int = 190
    zones: tuple[HrZone, ...] = field(default_factory=_default_zones)
    # Easy runs report the heart rate you would have at this pace (seconds per km).
    reference_pace_s_per_km: float = 330.0
    # Long runs: the first km, while pace and HR settle. Left out of drift and mechanics.
    burn_in_km: float = 3.0


def resolve_data_dir(arg: str | os.PathLike | None = None) -> Path:
    """--data-dir wins, then $STRIDEBOOK_DATA, then ./data."""
    if arg:
        return Path(arg).expanduser()
    if env := os.environ.get(ENV_VAR):
        return Path(env).expanduser()
    return DEFAULT_DATA_DIR


def load_settings(data_dir: Path) -> Settings:
    """Read <data_dir>/settings.toml if it exists, else defaults.

    The keys are a superset of Stridelab's hr_zones.toml (max_hr, [[zones]] name/min/max),
    so that file can be copied in as is; its `color` keys are ignored.
    """
    path = data_dir / SETTINGS_FILE
    if not path.exists():
        return Settings()
    raw = tomllib.loads(path.read_text())
    kwargs = {}
    if "max_hr" in raw:
        kwargs["max_hr"] = int(raw["max_hr"])
    if "zones" in raw:
        kwargs["zones"] = tuple(
            HrZone(z["name"], int(z["min"]), int(z["max"])) for z in raw["zones"]
        )
    if "reference_pace" in raw:
        kwargs["reference_pace_s_per_km"] = parse_pace(raw["reference_pace"])
    if "burn_in_km" in raw:
        kwargs["burn_in_km"] = float(raw["burn_in_km"])
    return Settings(**kwargs)


def parse_pace(text: str) -> float:
    """'5:30' -> 330.0 seconds per km."""
    minutes, seconds = text.strip().split(":")
    return int(minutes) * 60 + float(seconds)
