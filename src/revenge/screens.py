from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import controls
from .config import ASSETS


@dataclass(frozen=True)
class Screen:
    name: str
    point: tuple[int, int] | None = None
    result: str | None = None
    hold: float = 0.03


@dataclass(frozen=True)
class Template:
    filename: str
    image: np.ndarray
    screen: Screen
    region: tuple[int, int, int, int]
    threshold: float


class ScreenDetector:
    SCALE = 0.5

    def __init__(self):
        self.templates: list[Template] = []
        result_region = (20, 10, 650, 200)
        placements = (
            ("1st.png", "victory"),
            ("2nd.png", "victory"),
            ("3rd.png", "draw"),
            ("3rd_alt.png", "draw"),
            ("4th.png", "defeat"),
        )
        for filename, result in placements:
            self._add(
                f"end_results/{filename}",
                Screen("end", controls.PROCEED, result),
                result_region,
                0.9,
            )
        for result in ("victory", "defeat", "draw"):
            self._add(
                f"end_results/{result}.png", Screen("end", controls.PROCEED, result), result_region
            )

        self._add("states/loading_logo.png", Screen("loading"), (450, 350, 1100, 450))
        self._add("states/lobby_menu.png", Screen("lobby", controls.PROCEED), (1790, 20, 75, 65))
        self._add(
            "states/friend_menu_icon.png",
            Screen("friends", controls.EXIT_FRIEND_MENU),
            (900, 0, 175, 110),
        )
        self._add("states/exit_match_making.png", Screen("matchmaking"), (0, 925, 340, 135))
        self._add(
            "states/brawler_menu_heart.png", Screen("selection", controls.BACK), (1050, 0, 500, 110)
        )
        self._add(
            "states/brawler_menu_search.png",
            Screen("selection", controls.BACK),
            (1320, 0, 300, 100),
        )
        self._add("states/powerpoint.png", Screen("shop", controls.BACK), (1000, 5, 80, 80))
        self._add("states/close_popup.png", Screen("popup"), (1740, 140, 140, 100))
        self._add("states/brawl_pass_house.png", Screen("pass", controls.BACK), (1750, 0, 169, 100))
        self._add("states/go_back_arrow.png", Screen("menu", controls.BACK), (0, 0, 175, 110))
        self._add(
            "states/prestige_continue.png",
            Screen("prestige", controls.CONTINUE),
            (535, 950, 345, 95),
        )
        for filename in (
            "star_drop.png",
            "angelic_star_drop.png",
            "demonic_star_drop.png",
            "starr_nova_star_drop.png",
        ):
            duration = 0.05 if filename == "star_drop.png" else 8.0
            self._add(
                f"star_drop_types/{filename}",
                Screen(filename.removesuffix(".png"), controls.PROCEED, hold=duration),
                (790, 350, 350, 350),
            )
        self._add("states/daily_wins.png", Screen("daily_wins", controls.CENTER), (0, 0, 360, 110))
        self._add(
            "states/trophies_screen.png", Screen("reward", controls.PROCEED), (1545, 915, 365, 165)
        )
        self._add(
            "states/exit_outline.png",
            Screen("showdown_exit", controls.EXIT_SHOWDOWN),
            (800, 930, 340, 130),
        )

    def _add(
        self,
        filename: str,
        screen: Screen,
        region: tuple[int, int, int, int],
        threshold: float = 0.75,
    ):
        image = cv2.imread(str(ASSETS / "images" / filename), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Missing template: {filename}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        if width > region[2] or height > region[3] or not image.std(axis=(0, 1)).any():
            raise ValueError(f"Invalid template or search region: {filename}")
        scaled = cv2.resize(
            image,
            (max(1, round(width * self.SCALE)), max(1, round(height * self.SCALE))),
            interpolation=cv2.INTER_AREA,
        )
        scaled.flags.writeable = False
        self.templates.append(Template(filename, scaled, screen, region, threshold))

    def detect(self, rgb: np.ndarray) -> Screen:
        scaled = cv2.resize(
            rgb,
            (round(controls.WIDTH * self.SCALE), round(controls.HEIGHT * self.SCALE)),
            interpolation=cv2.INTER_AREA,
        )
        for template in self.templates:
            x, y, w, h = template.region
            left, top = round(x * self.SCALE), round(y * self.SCALE)
            right, bottom = round((x + w) * self.SCALE), round((y + h) * self.SCALE)
            crop = scaled[top:bottom, left:right]
            th, tw = template.image.shape[:2]
            if crop.shape[0] < th or crop.shape[1] < tw:
                continue
            matches = cv2.matchTemplate(crop, template.image, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(matches)
            if np.isfinite(score) and score >= template.threshold:
                screen = template.screen
                if screen.name == "popup":
                    return Screen(
                        "popup",
                        (
                            round((left + location[0] + tw / 2) / self.SCALE),
                            round((top + location[1] + th / 2) / self.SCALE),
                        ),
                    )
                return screen
        return Screen("unknown")
