from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"


@dataclass(frozen=True)
class Config:
    serial: str = ""
    package: str = "com.supercell.brawlstars"
    fps: int = 8
    provider: str = "auto"
    brawler: str = "shelly"
    confidence: float = 0.55
    terrain_confidence: float = 0.6
    terrain_interval: float = 0.4
    stale_timeout: float = 5.0
    reconnect_attempts: int = 3
    play_again: bool = False
    gadgets: bool = True
    discord_owner: int = 0
    data_dir: Path = Path("data")


def load_config(path: Path) -> Config:
    with path.open("rb") as stream:
        values = tomllib.load(stream)
    unknown = values.keys() - Config.__dataclass_fields__.keys()
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
    defaults = Config()
    for name, value in values.items():
        if name == "data_dir":
            if not isinstance(value, str) or not value.strip():
                raise ValueError("data_dir must be a path")
        elif type(value) is not type(getattr(defaults, name)):
            if type(getattr(defaults, name)) is float and type(value) is int:
                try:
                    values[name] = float(value)
                except OverflowError as error:
                    raise ValueError(f"{name} must be finite") from error
            else:
                raise ValueError(f"Invalid type for {name}")
    values["data_dir"] = (path.resolve().parent / values.get("data_dir", "data")).resolve()
    config = Config(**values)
    if not 1 <= config.fps <= 60:
        raise ValueError("fps must be between 1 and 60")
    if config.provider not in ("auto", "cpu", "cuda", "directml"):
        raise ValueError("provider must be auto, cpu, cuda or directml")
    for name in ("confidence", "terrain_confidence"):
        if not 0 < getattr(config, name) <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    for name in ("terrain_interval", "stale_timeout"):
        if not math.isfinite(getattr(config, name)) or getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not 0 <= config.reconnect_attempts <= 10:
        raise ValueError("reconnect_attempts must be between 0 and 10")
    if config.discord_owner < 0:
        raise ValueError("discord_owner must be a positive Discord user ID")
    if not re.fullmatch(r"[a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)+", config.package):
        raise ValueError("Invalid Android package name")
    return config
