import os

# Filesystem locations, resolved from this file's location (not snapshotted with the configs, see run_config.CONFIG_FILES)
PC_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # payload_control/
LOG_DIR = os.path.join(PC_DIR, "logs")
REPO_ROOT = os.path.abspath(os.path.join(PC_DIR, "..", ".."))
DRONE_PATH = os.path.join(REPO_ROOT, "system_model", "Starling2MaxURDF", "model", "starling2max.urdf")
PAYLOAD_PATH = os.path.join(REPO_ROOT, "system_model", "payload", "box", "box.urdf")
