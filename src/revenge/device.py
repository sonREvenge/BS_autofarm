from __future__ import annotations

import logging
import secrets
import socket
import struct
import subprocess
import threading
import time
from dataclasses import dataclass

import av
import numpy as np
from adbutils import AdbClient, AdbError, Network

from .config import ASSETS, Config
from .controls import ACTION_POINTER, HEIGHT, JOYSTICK, MOVEMENT_POINTER, WIDTH

log = logging.getLogger(__name__)
SERVER_VERSION = "5.0.1"


def select_device(serial: str):
    client = AdbClient(socket_timeout=5)
    if serial and ":" in serial:
        client.connect(serial, timeout=5)
    devices = client.device_list()
    if serial:
        devices = [d for d in devices if d.serial == serial]
    if len(devices) != 1:
        raise ConnectionError(
            "Select one ADB device with serial in config.toml; run revenge devices"
        )
    return devices[0]


@dataclass(frozen=True)
class Frame:
    sequence: int
    received: float
    rgb: np.ndarray


class FrameBuffer:
    def __init__(self):
        self.condition = threading.Condition()
        self.latest: Frame | None = None
        self.sequence = 0
        self.closed = False

    def publish(self, rgb):
        with self.condition:
            if self.closed:
                return
            rgb.flags.writeable = False
            self.sequence += 1
            self.latest = Frame(self.sequence, time.monotonic(), rgb)
            self.condition.notify_all()

    def get(self, after: int, timeout: float) -> Frame:
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.closed or (self.latest is not None and self.sequence > after),
                timeout=timeout,
            )
            if self.closed:
                raise ConnectionError("Video stream closed")
            if (
                not ready
                or self.latest is None
                or time.monotonic() - self.latest.received > timeout
            ):
                raise TimeoutError("No fresh video frame")
            return self.latest

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def mark(self) -> int:
        with self.condition:
            return self.sequence

    def peek(self) -> Frame | None:
        with self.condition:
            return None if self.closed else self.latest


def touch_packet(action: int, pointer: int, x: int, y: int, width: int, height: int):
    return struct.pack(
        ">BBQiiHHHII",
        2,
        action,
        pointer,
        max(0, min(width - 1, x)),
        max(0, min(height - 1, y)),
        width,
        height,
        0 if action == 1 else 65535,
        0,
        0,
    )


class Device:
    def __init__(self, config: Config, cancel: threading.Event | None = None):
        self.config = config
        self.device = select_device(config.serial)
        self.frames = FrameBuffer()
        self.video = self.control = self.server = None
        self.thread = None
        self.stop = threading.Event()
        self.cancel = cancel if cancel is not None else threading.Event()
        self.touches = {}
        self.lock = threading.RLock()
        self.resolution = None
        self.remote = None

    def check_game(self):
        output = self.device.shell(["pm", "path", self.config.package], timeout=5)
        if not any(line.startswith("package:") for line in output.splitlines()):
            raise ValueError(
                f"{self.config.package} is not installed on {self.device.serial}. "
                "Install the game or set package in config.toml."
            )

    def start(self):
        if self.cancel.is_set():
            raise InterruptedError("Device connection cancelled")
        scid = secrets.randbelow(0x7FFFFFFF)
        socket_name = f"scrcpy_{scid:08x}"
        remote = f"/data/local/tmp/revenge-scrcpy-{scid:08x}.jar"
        self.remote = remote
        self.device.sync.push(ASSETS / "server" / f"scrcpy-server-v{SERVER_VERSION}", remote)
        command = [
            f"CLASSPATH={remote}",
            "app_process",
            "/",
            "com.genymobile.scrcpy.Server",
            SERVER_VERSION,
            f"scid={scid:08x}",
            "tunnel_forward=true",
            "audio=false",
            "control=true",
            "video_codec=h264",
            "max_size=1920",
            f"max_fps={self.config.fps}",
            "video_bit_rate=4000000",
            "send_device_meta=false",
            "send_frame_meta=false",
            "send_stream_meta=false",
            "clipboard_autosync=false",
            "cleanup=true",
            "log_level=warn",
        ]
        try:
            self.server = self.device.shell(command, stream=True)
            deadline = time.monotonic() + 10
            while True:
                if self.cancel.is_set():
                    raise InterruptedError("Device connection cancelled")
                try:
                    self.video = self.device.create_connection(Network.LOCAL_ABSTRACT, socket_name)
                    break
                except AdbError:
                    if time.monotonic() >= deadline:
                        raise ConnectionError(
                            "scrcpy server did not open its video socket"
                        ) from None
                    self.cancel.wait(0.1)
            self.video.settimeout(self.config.stale_timeout)
            if self.video.recv(1) != b"\x00":
                raise ConnectionError("Invalid scrcpy handshake")
            self.control = self.device.create_connection(Network.LOCAL_ABSTRACT, socket_name)
            self.control.settimeout(3)
            self.thread = threading.Thread(target=self._decode, name="video", daemon=True)
            self.thread.start()
        except BaseException:
            self.close()
            raise

    def _decode(self):
        try:
            codec = av.CodecContext.create("h264", "r")
            while not self.stop.is_set():
                data = self.video.recv(1024 * 256)
                if not data:
                    break
                for packet in codec.parse(data):
                    for frame in codec.decode(packet):
                        rgb = frame.to_ndarray(format="rgb24")
                        with self.lock:
                            new_resolution = (rgb.shape[1], rgb.shape[0])
                            if self.resolution and new_resolution != self.resolution:
                                raise ConnectionError("Display size changed; reconnect required")
                            self.resolution = new_resolution
                        self.frames.publish(rgb)
        except (OSError, av.FFmpegError, ConnectionError):
            if not self.stop.is_set():
                log.exception("Video stream ended")
        finally:
            self.frames.close()

    def touch(self, action: int, pointer: int, point: tuple[float, float]):
        with self.lock:
            if action != 1 and (self.cancel.is_set() or self.stop.is_set()):
                return
            if action == 1 and pointer not in self.touches:
                return
            if not self.resolution or self.control is None:
                raise ConnectionError("Touch input is not ready")
            width, height = self.resolution
            x, y = round(point[0] * width / WIDTH), round(point[1] * height / HEIGHT)
            try:
                self.control.sendall(touch_packet(action, pointer, x, y, width, height))
            except OSError as error:
                raise ConnectionError("Touch input disconnected") from error
            if action == 1:
                self.touches.pop(pointer, None)
            else:
                self.touches[pointer] = point

    def tap(self, point, duration=0.03):
        self.touch(0, ACTION_POINTER, point)
        try:
            self.cancel.wait(duration)
        finally:
            self.touch(1, ACTION_POINTER, point)

    def move(self, direction):
        with self.lock:
            if direction is None:
                if MOVEMENT_POINTER in self.touches:
                    self.touch(1, MOVEMENT_POINTER, self.touches[MOVEMENT_POINTER])
                return
            if MOVEMENT_POINTER not in self.touches:
                self.touch(0, MOVEMENT_POINTER, JOYSTICK)
            dx, dy = direction
            length = max(1, (dx * dx + dy * dy) ** 0.5)
            self.touch(
                2,
                MOVEMENT_POINTER,
                (JOYSTICK[0] + 150 * dx / length, JOYSTICK[1] + 150 * dy / length),
            )

    def release(self):
        with self.lock:
            error = None
            for pointer, point in list(self.touches.items()):
                try:
                    self.touch(1, pointer, point)
                except OSError as caught:
                    error = caught
            if error is not None:
                raise error

    def restart_game(self):
        self.release()
        if self.cancel.is_set():
            return
        self.device.shell(["am", "force-stop", self.config.package], timeout=5)
        if self.cancel.is_set():
            return
        self.device.shell(
            ["monkey", "-p", self.config.package, "-c", "android.intent.category.LAUNCHER", "1"],
            timeout=5,
        )

    def close(self):
        try:
            self.release()
        except (OSError, ConnectionError):
            log.debug("Input already disconnected")
        self.stop.set()
        self.frames.close()
        for connection in (self.video, self.control, self.server):
            if connection is not None:
                try:
                    if isinstance(connection, socket.socket):
                        connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                finally:
                    try:
                        connection.close()
                    except OSError:
                        pass
        self.video = self.control = self.server = None
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=6)
        with self.lock:
            self.touches.clear()
            self.resolution = None
        if self.remote:
            try:
                self.device.shell(["rm", "-f", self.remote], timeout=5)
            except (AdbError, OSError, subprocess.SubprocessError):
                log.debug("Could not remove temporary server file", exc_info=True)
            self.remote = None
