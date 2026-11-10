# BS_autofarm

Local Brawl Stars automation over ADB, with optional Discord commands.

## Install on Windows

Use 64-bit Python 3.11 or newer. Install Brawl Stars in your Android emulator,
start the emulator and enable ADB debugging. Use a landscape 16:9 resolution.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[cpu]"
```

Pip installs the Python dependencies and ADB. Models and the scrcpy server are included.
If Windows reports a DLL loading error, install or repair the
[Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist/)
for your Python architecture.

Copy `config.example.toml` to `config.toml` if it does not exist. Set `brawler` to the
character selected in the game. Set `serial` if several devices are connected; a network
emulator uses its device address, such as `127.0.0.1:5555`. Port 5037 belongs to the ADB server.

```powershell
.\.venv\Scripts\python.exe -m revenge devices
.\.venv\Scripts\python.exe -m revenge check --device
.\.venv\Scripts\python.exe -m revenge run
```

Select the brawler and game mode in the game before starting. Press Ctrl+C to stop.
`check --device` verifies assets, the selected model runtime, Android video and game installation.
It does not send touch input or evaluate gameplay. Logs and match history are saved in `data/`.

## Commands

Use `.venv\Scripts\python.exe -m revenge` followed by:

- `brawlers` to list profiles.
- `check` to validate assets and models without an Android device.
- `benchmark --image screenshot.png` to measure model inference time.
- `--config other.toml run` to use another configuration.
- `--help` for all options.

## Discord

Enable Message Content Intent for your Discord bot and set `discord_owner` to your user ID
in `config.toml`. Install the Discord dependency and set the token:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[discord]"
$env:REVENGE_DISCORD_TOKEN = "your-bot-token"
.\.venv\Scripts\python.exe -m revenge run --discord
```

Discord mode starts paused. Only the owner can use `!start`, `!stop`, `!restart`, `!status`,
`!screenshot`, `!brawler shelly` and `!help`. Changing the profile does not select a brawler
in the game. `!restart` restarts the game, not the emulator.

## GPU

Install one runtime extra: `.[cpu]`, `.[cuda]` or `.[directml]`. DirectML requires Windows.
Uninstall the previous ONNX Runtime distribution before switching. CUDA needs compatible
GPU drivers and runtime libraries. Set `provider` to `auto`, `cpu`, `cuda` or `directml`.
Auto mode allows CPU fallback; an unavailable explicitly requested provider fails.

The runner uses auto-aim and manual brawler selection. It has no PylaAI playstyle loader or
OCR trophy tracking. `play_again = true` rematches after a victory. UI templates and touch
coordinates may need recalibration after game updates. See [asset notices](THIRD_PARTY.md).
