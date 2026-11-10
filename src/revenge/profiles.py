from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from .config import ASSETS


@dataclass(frozen=True)
class Profile:
    safe_range: float
    attack_range: float
    super_range: float
    super_type: str
    ignore_walls_for_attacks: bool
    ignore_walls_for_supers: bool
    hold_attack: float


def load_profiles() -> dict[str, Profile]:
    data = json.loads((ASSETS / "brawlers_info.json").read_text(encoding="utf-8"))
    profiles = {}
    for name, values in data.items():
        fields = {key: values[key] for key in Profile.__dataclass_fields__}
        for key in ("safe_range", "attack_range", "super_range", "hold_attack"):
            value = fields[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {key} in brawler profile {name}")
        for key in ("ignore_walls_for_attacks", "ignore_walls_for_supers"):
            if type(fields[key]) is not bool:
                raise ValueError(f"Invalid {key} in brawler profile {name}")
        if fields["super_type"] not in (
            "damage",
            "charge",
            "dash",
            "spawnable",
            "other",
            "other_target",
        ):
            raise ValueError(f"Unsupported super type in brawler profile {name}")
        profiles[name] = Profile(**fields)
    return profiles


def resolve_brawler(name: str, profiles: dict[str, Profile]) -> str:
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    if key not in profiles:
        raise ValueError(f"Unknown brawler: {name}")
    return key
