from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from . import __version__
from .config import ASSETS, Config, load_config


def verify_assets():
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        path = ASSETS / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Asset missing or changed: {name}")
    return manifest


def logging_setup(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        RotatingFileHandler(
            data_dir / "revenge.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        ),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def list_devices():
    from adbutils import AdbClient, AdbError, adb_path

    try:
        executable = adb_path()
    except AdbError:
        executable = "not found; an existing ADB server is required"
    print("ADB executable:", executable)
    try:
        devices = AdbClient(socket_timeout=5).device_list()
    except (AdbError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"Cannot connect to ADB: {error}") from error
    for device in devices:
        print(device.serial)
    if not devices:
        print("No devices. Enable USB debugging or the emulator's ADB setting.")


def check_device(config: Config):
    from adbutils import AdbError

    from .controls import validate_resolution
    from .device import Device

    device = None
    try:
        device = Device(config)
        device.start()
        frame = device.frames.get(0, config.stale_timeout)
        height, width = frame.rgb.shape[:2]
        validate_resolution(width, height)
        print(f"Video connection: OK ({width}x{height})")
        device.check_game()
        print("Game installation: OK")
    except (AdbError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"Device check failed: {error}") from error
    finally:
        if device is not None:
            device.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="REvengeBSIndustry")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("devices", help="List devices connected to the ADB server")
    sub.add_parser("brawlers", help="List available brawler profiles")
    check = sub.add_parser("check", help="Validate configuration, assets and model inference")
    check.add_argument(
        "--device", action="store_true", help="Also check Android video and game installation"
    )
    run = sub.add_parser("run", help="Run locally; Ctrl+C stops and releases input")
    run.add_argument("--discord", action="store_true", help="Start paused, controlled by Discord")
    bench = sub.add_parser("benchmark", help="Time both models without connecting a device")
    bench.add_argument("--image", type=Path, help="Use a game screenshot instead of a blank frame")
    bench.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        if args.command == "brawlers":
            from .profiles import load_profiles

            print("\n".join(sorted(load_profiles())))
            return 0
        if args.command == "devices":
            list_devices()
            return 0
        config = load_config(args.config)
        if args.command == "benchmark" and not 1 <= args.iterations <= 10000:
            raise ValueError("iterations must be between 1 and 10000")
        verify_assets()
        if args.command in ("check", "benchmark"):
            import cv2
            import numpy as np

            from .profiles import load_profiles, resolve_brawler
            from .screens import ScreenDetector
            from .vision import EntityDetector, TerrainDetector

            profiles = load_profiles()
            resolve_brawler(config.brawler, profiles)
            ScreenDetector()
            entities = EntityDetector(ASSETS / "models/mainInGameModel.onnx", config.provider)
            terrain = TerrainDetector(ASSETS / "models/tileDetector.onnx", config.provider)
            rgb = np.zeros((1080, 1920, 3), dtype="uint8")
            if args.command == "benchmark" and args.image:
                bgr = cv2.imread(str(args.image))
                if bgr is None:
                    raise ValueError(f"Cannot read image: {args.image}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            entities.detect(rgb, config.confidence)
            terrain.detect(rgb, config.terrain_confidence)
            if args.command == "check":
                print("Configuration, assets, templates and both models: OK")
                print("Entity provider:", entities.model.get_providers()[0])
                print("Terrain provider:", terrain.model.get_providers()[0])
                print("Brawler profiles:", len(profiles))
                if args.device:
                    check_device(config)
                    print("Touch controls and in-game accuracy are not checked.")
                else:
                    print("Use check --device to verify video from a connected Android device.")
            else:
                for label, model, threshold in (
                    ("entities", entities, config.confidence),
                    ("terrain", terrain, config.terrain_confidence),
                ):
                    times = []
                    for _ in range(args.iterations):
                        started = time.perf_counter()
                        model.detect(rgb, threshold)
                        times.append((time.perf_counter() - started) * 1000)
                    print(
                        f"{label}: median {np.median(times):.1f} ms, "
                        f"p95 {np.percentile(times, 95):.1f} ms, "
                        f"{model.model.get_providers()[0]} ({args.iterations} runs)"
                    )
            return 0
        logging_setup(config.data_dir)
        from .runtime import Runtime

        runtime = Runtime(config, running=not args.discord)
        old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda *_: runtime.stop.set())
        try:
            if args.discord:
                token = os.environ.get("REVENGE_DISCORD_TOKEN", "")
                if not token or not config.discord_owner:
                    raise ValueError("Set REVENGE_DISCORD_TOKEN and discord_owner to use Discord")
                from .discord_control import serve

                asyncio.run(serve(runtime, config.discord_owner, token))
            else:
                runtime.run()
        finally:
            signal.signal(signal.SIGINT, old_handler)
            runtime.shutdown()
        return 1 if runtime.error else 0
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
