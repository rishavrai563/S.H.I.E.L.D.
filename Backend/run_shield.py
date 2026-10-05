"""
SHIELD Executive Launcher
-------------------------
Parses YAML configurations and hands full control to the Orchestrator.
"""
import yaml
import os
from core.orchestrator import Orchestrator
from core.stream_manager import CameraConfig

def load_system_config():
    """Reads cameras.yaml to build the system architecture."""
    with open("config/cameras.yaml", 'r') as f:
        cam_data = yaml.safe_load(f)
    
    configs = []
    for cam in cam_data.get("cameras", []):
        configs.append(
            CameraConfig(
                camera_id=cam["camera_id"],
                source=cam["source"],
                name=cam.get("name", "Unknown Node")
            )
        )
    return configs

def main():
    print("⚙️ Initializing SHIELD Enterprise Architecture...")
    
    try:
        camera_configs = load_system_config()
        
        # Hand over complete control to the Orchestrator
        shield_system = Orchestrator(
            cameras=camera_configs, 
            device="cpu", 
            display=True 
        )
        
        print("\n🚀 ALL SYSTEMS NOMINAL. LAUNCHING COMMAND CENTER...")
        shield_system.run()

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline failed: {e}")
    finally:
        print("Shutting down data streams...")

if __name__ == "__main__":
    main()