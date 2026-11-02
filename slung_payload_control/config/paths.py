import os

SPC_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # slung_payload_control/
REPO_ROOT = os.path.abspath(os.path.join(SPC_DIR, "..")) # genesis_playground/

DATA_DIR = os.path.join(SPC_DIR, "data") # data
TRAJ_BANK_DIR = os.path.join(DATA_DIR, "traj") # data/traj

MODELS_DIR = os.path.join(SPC_DIR, "models") # models/

DRONE_PATH = os.path.join(REPO_ROOT, "system_model", "Starling2MaxURDF", "model", "starling2max.urdf")
PAYLOAD_PATH = os.path.join(REPO_ROOT, "system_model", "payload", "box", "box.urdf")

if __name__ == "__main__":
    print("================= paths.py entries =================")
    print("SPC_DIR:         ", SPC_DIR)
    print("REPO_ROOT:       ", REPO_ROOT)
    print("DATA_DIR:        ", DATA_DIR)
    print("MODELS_DIR:      ", MODELS_DIR)
    print("TRAJ_BANK_DIR:   ", TRAJ_BANK_DIR)
    print("DRONE_PATH:      ", DRONE_PATH)
    print("PAYLOAD_PATH:    ", PAYLOAD_PATH)
