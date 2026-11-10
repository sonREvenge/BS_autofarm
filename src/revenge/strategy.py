from __future__ import annotations

import math
from dataclasses import dataclass

from .navigation import Navigation
from .profiles import Profile
from .vision import Entity


@dataclass(frozen=True)
class Decision:
    movement: tuple[float, float] | None = None
    attack: bool = False
    use_super: bool = False
    abilities: bool = False


def decide(entities: list[Entity], navigation: Navigation, profile: Profile) -> Decision:
    players = [e for e in entities if e.name == "player"]
    if not players:
        return Decision()
    player = max(players, key=lambda e: e.score).ground
    enemies = [e for e in entities if e.name in ("enemy", "enemy_spawnable")]
    use_super = abilities = False
    if not enemies:
        teammates = [e for e in entities if e.name == "teammate"]
        goal = (
            min(teammates, key=lambda e: math.dist(player, e.ground)).ground
            if teammates
            else (960, 540)
        )
        attack = False
    else:
        enemy = min(enemies, key=lambda e: math.dist(player, e.ground))
        dx, dy = enemy.ground[0] - player[0], enemy.ground[1] - player[1]
        distance = math.hypot(dx, dy)
        visible = navigation.clear_line(player, enemy.ground)
        attack = distance <= profile.attack_range and (profile.ignore_walls_for_attacks or visible)
        is_brawler = enemy.name == "enemy"
        abilities = is_brawler and attack
        use_super = is_brawler and (
            profile.super_type == "other"
            or (distance <= profile.super_range and (profile.ignore_walls_for_supers or visible))
        )
        if distance < profile.safe_range:
            length = max(distance, 1)
            goal = (player[0] - dx / length * 180, player[1] - dy / length * 180)
        elif attack and distance <= profile.attack_range * 0.85:
            return Decision(attack=attack, use_super=use_super, abilities=abilities)
        else:
            goal = enemy.ground
    step = navigation.next_step(player, goal)
    movement = None if step is None else (step[0] - player[0], step[1] - player[1])
    return Decision(movement, attack, use_super, abilities)
