"""
SHIELD Executive Launcher
-------------------------
Parses YAML configurations and hands full control to the Orchestrator.

Supports two modes:
  python run_shield.py        → CLI mode with OpenCV display windows
  python run_shield.py --api  → API server mode (headless, starts FastAPI)
"""
import argparse
import os
import sys
import yaml

from core.orchestrator import Orchestrator
from core.stream_manager import CameraConfig


def load_config() -> dict:
    """Reads config.yaml for master system configuration."""
    config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_camera_configs() -> list:
    """Reads cameras.yaml to build the camera architecture."""
    # --cameras / SHIELD_CAMERAS selects another topology file (e.g. config/cameras.live.yaml)
    cam_path = os.environ.get("SHIELD_CAMERAS") or os.path.join(os.path.dirname(__file__), "config", "cameras.yaml")
    if not os.path.isabs(cam_path):
        cam_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cam_path)
    with open(cam_path, "r") as f:
        cam_data = yaml.safe_load(f)

    configs = []
    for cam in cam_data.get("cameras", []):
        configs.append(
            CameraConfig(
                camera_id=cam["camera_id"],
                source=cam["source"],
                name=cam.get("name", "Unknown Node"),
                gps_lat=cam.get("gps_lat", 0.0),
                gps_lng=cam.get("gps_lng", 0.0),
                night_mode=str(cam.get("night_mode", "")),
                width=int(cam.get("width", 0) or 0),
                height=int(cam.get("height", 0) or 0),
                loop=bool(cam.get("loop", False)),
                rotate=int(cam.get("rotate", 0) or 0),
                realtime=bool(cam.get("realtime", False)),
            )
        )
    return configs


def run_cli_mode(config: dict, camera_configs: list):
    """Original CLI mode — runs the Orchestrator with OpenCV display windows."""
    print("⚙️ Initializing SHIELD Enterprise Architecture (CLI Mode)...")

    try:
        device = config.get("system", {}).get("device", "cpu")

        shield_system = Orchestrator(
            cameras=camera_configs,
            device=device,
            display=True,
            config=config,
        )

        print("\n🚀 ALL SYSTEMS NOMINAL. LAUNCHING COMMAND CENTER...")
        shield_system.run()

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Shutting down data streams...")


def _deep_merge(base: dict, extra: dict) -> None:
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def apply_camera_file_settings(config: dict) -> None:
    """A camera topology file may carry a `settings:` block that overrides config.yaml
    for that setup (e.g. overlapping laptop + phone views need cross_camera_min_gap: 0)."""
    cam_path = os.environ.get("SHIELD_CAMERAS")
    if not cam_path:
        return
    if not os.path.isabs(cam_path):
        cam_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cam_path)
    with open(cam_path, "r") as f:
        data = yaml.safe_load(f) or {}
    if data.get("settings"):
        _deep_merge(config, data["settings"])
        print(f"  [CONFIG] {os.path.basename(cam_path)} overrides: {', '.join(data['settings'])}")


def run_api_mode(config: dict):
    """API server mode — starts FastAPI with uvicorn (headless, no OpenCV windows)."""
    print("⚙️ Initializing SHIELD Enterprise Architecture (API Mode)...")

    from api import start_api_server

    api_config = config.get("api", {})
    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 8000)

    start_api_server(host=host, port=port, config=config)


def main():
    parser = argparse.ArgumentParser(
        description="SHIELD — AI-powered CCTV Video Analytics Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_shield.py          # CLI mode with OpenCV display
  python run_shield.py --api    # API server mode (headless)
  python run_shield.py --api --port 9000   # API on custom port
  python run_shield.py --api --cameras config/cameras.live.yaml   # phones + webcam
        """,
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Start in API server mode (headless, no OpenCV windows)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override the API server port (default: from config.yaml)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Override the API server host (default: from config.yaml)",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help="Camera topology YAML (default: config/cameras.yaml), e.g. config/cameras.live.yaml",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Override the compute device (default: from config.yaml)",
    )

    args = parser.parse_args()

    if args.cameras:
        os.environ["SHIELD_CAMERAS"] = args.cameras

    # Load configuration
    config = load_config()
    apply_camera_file_settings(config)

    # Apply CLI overrides
    if args.device:
        config.setdefault("system", {})["device"] = args.device
    if args.port:
        config.setdefault("api", {})["port"] = args.port
    if args.host:
        config.setdefault("api", {})["host"] = args.host

    if args.api:
        run_api_mode(config)
    else:
        camera_configs = load_camera_configs()
        run_cli_mode(config, camera_configs)


if __name__ == "__main__":
    main()
