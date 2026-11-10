from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field

import cv2
from adbutils import AdbError

from . import controls
from .config import ASSETS, Config
from .device import Device, Frame
from .navigation import Navigation
from .profiles import Profile, load_profiles, resolve_brawler
from .screens import Screen, ScreenDetector
from .stats import Stats
from .strategy import Decision, decide
from .vision import EntityDetector, TerrainDetector

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Perception:
    entities: EntityDetector
    terrain: TerrainDetector
    screens: ScreenDetector


@dataclass
class Session:
    sequence: int = 0
    revision: int = -1
    last_frame: float = field(default_factory=time.monotonic)
    last_terrain: float = 0
    navigation: Navigation | None = None
    screen_key: tuple[str, str | None] | None = None
    confirmations: int = 0
    player_frames: int = 0
    next_action: float = 0
    attack_started: float | None = None
    last_attack: float = 0
    held_screen: str | None = None
    hold_until: float = 0
    ability_times: dict[str, float] = field(default_factory=dict)

    def reset(self):
        self.navigation = None
        self.screen_key = None
        self.confirmations = self.player_frames = 0
        self.attack_started = None
        self.held_screen = None


class Runtime:
    def __init__(self, config: Config, running: bool = True):
        self.config = config
        self.profiles = load_profiles()
        self.brawler = resolve_brawler(config.brawler, self.profiles)
        self.lock = threading.RLock()
        self.input_lock = threading.RLock()
        self.stop = threading.Event()
        self.running = running
        self.revision = 0
        self.restart_requested = False
        self.status = "starting"
        self.error: str | None = None
        self.input_deadline: float | None = None
        self.totals = {"victory": 0, "defeat": 0, "draw": 0}
        self.device: Device | None = None
        self.result_recorded = True
        self.match_brawler: str | None = None

    def set_running(self, value: bool):
        with self.input_lock:
            with self.lock:
                value = value and not self.stop.is_set()
                if value and self.running:
                    return
                self.running = value
                self.input_deadline = None
                self.revision += 1
                device = self.device
            if not value and device:
                device.release()

    def set_brawler(self, name: str):
        name = resolve_brawler(name, self.profiles)
        with self.input_lock:
            with self.lock:
                self.brawler = name
                self.input_deadline = None
                self.revision += 1
                device = self.device
            if device:
                device.release()

    def info(self):
        with self.lock:
            return {
                "running": self.running,
                "screen": self.status,
                "brawler": self.brawler,
                "results": dict(self.totals),
                "error": self.error,
            }

    def screenshot(self):
        with self.lock:
            device = self.device
        frame = device.frames.peek() if device is not None else None
        if frame is None:
            raise ValueError("No frame received yet")
        if time.monotonic() - frame.received > self.config.stale_timeout:
            raise ValueError("The latest screenshot is stale; waiting for a fresh frame")
        return frame.rgb.copy()

    def request_restart(self):
        with self.lock:
            self.restart_requested = True
            self.revision += 1

    def shutdown(self):
        self.stop.set()
        try:
            self.set_running(False)
        except OSError:
            log.debug("Input disconnected during shutdown", exc_info=True)

    def _status(self, value: str):
        with self.lock:
            if self.status != value:
                log.info("Screen: %s", value)
            self.status = value

    def _guard_input(self):
        while not self.stop.wait(0.05):
            with self.input_lock:
                with self.lock:
                    deadline, device = self.input_deadline, self.device
                    if deadline is None or time.monotonic() < deadline or device is None:
                        continue
                    self.input_deadline = None
                    self.revision += 1
                try:
                    device.release()
                except OSError:
                    log.warning("Input release failed after frame expiry", exc_info=True)
                    device.frames.close()
                self._status("stale_frame")

    def run(self):
        stats = None
        guard = threading.Thread(target=self._guard_input, name="input-guard", daemon=True)
        guard.start()
        try:
            perception = Perception(
                EntityDetector(ASSETS / "models/mainInGameModel.onnx", self.config.provider),
                TerrainDetector(ASSETS / "models/tileDetector.onnx", self.config.provider),
                ScreenDetector(),
            )
            stats = Stats(self.config.data_dir / "matches.sqlite3")
            with self.lock:
                self.totals = stats.summary()
            failures = 0
            while not self.stop.is_set():
                device = None
                connected_at = None
                backoff = 0
                try:
                    device = Device(self.config, cancel=self.stop)
                    device.check_game()
                    device.start()
                    connected_at = time.monotonic()
                    with self.input_lock, self.lock:
                        self.device = device
                    self._session(device, perception, stats)
                    break
                except (OSError, AdbError, subprocess.SubprocessError) as error:
                    if self.stop.is_set():
                        break
                    if connected_at is not None and time.monotonic() - connected_at >= 60:
                        failures = 0
                    failures += 1
                    self._status("disconnected")
                    if failures > self.config.reconnect_attempts:
                        raise ConnectionError("Reconnect limit reached") from error
                    log.warning(
                        "Connection lost (%s); reconnect %d/%d",
                        error,
                        failures,
                        self.config.reconnect_attempts,
                    )
                    backoff = min(2**failures, 10)
                finally:
                    with self.input_lock:
                        with self.lock:
                            self.device = None
                            self.input_deadline = None
                        if device:
                            device.close()
                self.stop.wait(backoff)
        except Exception as error:
            if isinstance(error, ValueError):
                log.error("Runtime stopped: %s", error)
            else:
                log.exception("Runtime stopped")
            with self.lock:
                self.error = str(error)
        finally:
            try:
                if stats:
                    stats.close()
            finally:
                with self.lock:
                    self.running = False
                    self.status = "stopped"
                self.stop.set()
                guard.join(timeout=8)

    def _session(self, device: Device, perception: Perception, stats: Stats):
        session = Session()
        while not self.stop.is_set():
            started = time.monotonic()
            with self.input_lock:
                with self.lock:
                    restart = self.restart_requested
                    self.restart_requested = False
                if self.stop.is_set():
                    return
                if restart:
                    with self.lock:
                        self.input_deadline = None
                    device.restart_game()
                    session.reset()
                    session.sequence = device.frames.mark()
                    session.last_frame = time.monotonic()
                    self._status("loading")
            try:
                frame = device.frames.get(session.sequence, min(0.5, self.config.stale_timeout))
            except TimeoutError:
                with self.input_lock:
                    with self.lock:
                        self.input_deadline = None
                    device.release()
                session.reset()
                self._status("waiting_for_frame")
                if time.monotonic() - session.last_frame >= self.config.stale_timeout:
                    raise
                continue
            session.sequence, session.last_frame = frame.sequence, frame.received
            raw = frame.rgb
            height, width = raw.shape[:2]
            controls.validate_resolution(width, height)
            reference = Frame(
                frame.sequence, frame.received, cv2.resize(raw, (controls.WIDTH, controls.HEIGHT))
            )
            with self.input_lock:
                with self.lock:
                    running, revision, brawler = self.running, self.revision, self.brawler
                if self.stop.is_set():
                    return
                if session.revision != revision:
                    device.release()
                    session.reset()
                    session.revision = revision
                if not running:
                    self._status("paused")
            if running:
                self._frame(
                    device,
                    reference,
                    session,
                    revision,
                    brawler,
                    perception,
                    stats,
                )
            self.stop.wait(max(0, 1 / self.config.fps - (time.monotonic() - started)))

    def _frame(
        self,
        device: Device,
        frame: Frame,
        session: Session,
        revision: int,
        brawler: str,
        perception: Perception,
        stats: Stats,
    ):
        rgb = frame.rgb
        screen = perception.screens.detect(rgb)
        key = (screen.name, screen.result)
        session.confirmations = session.confirmations + 1 if key == session.screen_key else 1
        session.screen_key = key
        decision = None
        now = time.monotonic()
        if screen.name == "unknown":
            found = perception.entities.detect(rgb, self.config.confidence)
            has_player = any(entity.name == "player" for entity in found)
            session.player_frames = session.player_frames + 1 if has_player else 0
            if session.player_frames >= 2:
                if (
                    session.navigation is None
                    or now - session.last_terrain >= self.config.terrain_interval
                ):
                    maps = perception.terrain.detect(rgb, self.config.terrain_confidence)
                    session.navigation = Navigation(maps.movement, maps.projectiles)
                    session.last_terrain = time.monotonic()
                decision = decide(found, session.navigation, self.profiles[brawler])
        else:
            session.player_frames = 0

        with self.input_lock:
            with self.lock:
                current = self.running and self.revision == revision and not self.stop.is_set()
            if not current:
                device.release()
                session.reset()
                return
            now = time.monotonic()
            max_age = min(self.config.stale_timeout, max(0.5, 2 / self.config.fps))
            if now - frame.received > max_age:
                with self.lock:
                    self.input_deadline = None
                device.release()
                session.reset()
                self._status("stale_frame")
                return
            with self.lock:
                self.input_deadline = frame.received + max_age
            self._status("match" if decision is not None else screen.name)
            if screen.name != "unknown":
                self._menu(device, screen, session, stats, brawler, now)
            elif decision is None:
                device.release()
                session.attack_started = session.held_screen = None
                session.navigation = None
            else:
                if session.held_screen is not None:
                    device.release()
                    session.held_screen = None
                self.result_recorded = False
                if self.match_brawler is None:
                    self.match_brawler = brawler
                self._play(device, rgb, session, decision, self.profiles[brawler], now)

    def _menu(
        self, device, screen: Screen, session: Session, stats: Stats, brawler: str, now: float
    ):
        session.navigation = None
        session.attack_started = None
        if session.held_screen == screen.name:
            if now < session.hold_until:
                return
            device.release()
            session.held_screen = None
            session.next_action = now + 0.5
            return
        device.release()
        session.held_screen = None
        if session.confirmations < 2:
            return
        if screen.result and not self.result_recorded:
            stats.record(self.match_brawler or brawler, screen.result)
            with self.lock:
                self.totals = stats.summary()
            self.result_recorded = True
            self.match_brawler = None
        if screen.name == "lobby":
            self.match_brawler = None
        if screen.point is None or now < session.next_action:
            return
        point = screen.point
        if screen.name == "end" and screen.result == "victory" and self.config.play_again:
            point = controls.PLAY_AGAIN
        if screen.hold > 0.1:
            device.touch(0, controls.ACTION_POINTER, point)
            session.held_screen = screen.name
            session.hold_until = now + screen.hold
        else:
            device.tap(point, screen.hold)
        session.next_action = now + (0.2 if screen.name == "star_drop" else 2)

    def _play(
        self,
        device: Device,
        rgb,
        session: Session,
        decision: Decision,
        profile: Profile,
        now: float,
    ):
        device.move(decision.movement)
        if decision.attack and profile.hold_attack:
            if session.attack_started is None:
                device.touch(0, controls.ACTION_POINTER, controls.ATTACK)
                session.attack_started = now
            elif now - session.attack_started >= profile.hold_attack + 1.5:
                device.touch(1, controls.ACTION_POINTER, controls.ATTACK)
                session.attack_started = None
        else:
            if session.attack_started is not None:
                device.touch(1, controls.ACTION_POINTER, controls.ATTACK)
                session.attack_started = None
            if decision.attack and now - session.last_attack >= 0.3:
                device.tap(controls.ATTACK)
                session.last_attack = now
        if session.attack_started is None:
            self._abilities(device, rgb, session, decision, now)

    def _abilities(self, device: Device, rgb, session: Session, decision: Decision, now: float):
        abilities = [
            (
                "hyper",
                decision.abilities,
                controls.HYPERCHARGE,
                (1350, 940, 1450, 1050),
                (137, 158, 159),
                (179, 255, 255),
                1800,
            ),
            (
                "super",
                decision.use_super,
                controls.SUPER,
                (1460, 830, 1560, 930),
                (17, 170, 200),
                (27, 255, 255),
                1800,
            ),
            (
                "gadget",
                decision.abilities and self.config.gadgets,
                controls.GADGET,
                (1580, 930, 1700, 1050),
                (57, 219, 165),
                (62, 255, 255),
                1300,
            ),
        ]
        for name, allowed, point, (x1, y1, x2, y2), low, high, minimum in abilities:
            if self.stop.is_set():
                return
            if not allowed or now - session.ability_times.get(name, 0) < 2:
                continue
            hsv = cv2.cvtColor(rgb[y1:y2, x1:x2], cv2.COLOR_RGB2HSV)
            if cv2.countNonZero(cv2.inRange(hsv, low, high)) >= minimum:
                device.tap(point)
                session.ability_times[name] = now
