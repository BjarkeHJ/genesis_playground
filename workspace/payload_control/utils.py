import torch
import glob
import re
import os
import time

from dataclasses import asdict, is_dataclass
from genesis.vis.viewer_plugins import ViewerPlugin
from config_env import SCRIPT_DIR

# Helper converting dataclass to python dict - Used for converting TrainingConfig to dict as expected by RSL_RL
def dataclass_to_dict(obj):
    assert is_dataclass(obj) and not isinstance(obj, type), f"{type(obj).__name__} is not a dataclass instance"
    return asdict(obj)

# Random bounded sampling
def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

# Resolving a model (.pt file)
def resolve_ckpt(run_name, ckpt):
    # Explicit path wins; otherwise pick model_<ckpt>.pt (or the latest one) from logs/<run_name>
    if ckpt is not None and os.path.isfile(ckpt):
        return ckpt
    log_dir = os.path.join(SCRIPT_DIR, "logs", run_name)
    if ckpt is not None:
        path = os.path.join(log_dir, f"model_{ckpt}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    ckpts = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No model_*.pt checkpoints in {log_dir}")
    return max(ckpts, key=lambda p: int(re.search(r"model_(\d+)\.pt", p).group(1)))

# Viewer text overlay - Lines of live values for one env, drawn in the bottom-left corner of the viewer
# (top-left holds the viewer help text). font_pt < 22 crashes the pyrender font loader on 1-pixel-tall glyphs ('-', '.')
class LogOverlay(ViewerPlugin):
    def __init__(self, env_idx=0, font_pt=22, color=(1.0, 1.0, 1.0, 1.0)):
        super().__init__()
        self.env_idx = env_idx
        self.font_pt = font_pt
        self.color = color
        self.entries = [] # (label, getter, fmt)
        self._lines = []

    def update_on_sim_step(self):
        # Sim thread: evaluate getters and do the GPU->CPU transfer here, so on_draw only touches strings
        lines = []
        for label, getter, fmt in self.entries:
            v = getter()
            if isinstance(v, torch.Tensor):
                v = v[self.env_idx] if v.dim() > 0 and v.shape[0] > self.env_idx else v
                v = v.detach().cpu().tolist()
            if isinstance(v, (list, tuple)):
                txt = "[" + " ".join(fmt.format(x) for x in v) + "]"
            else:
                txt = fmt.format(v)
            lines.append(f"{label:<{self._label_w}} {txt}")
        self._lines = lines # atomic swap, read by the render thread

    def on_draw(self):
        lines = self._lines
        if not lines:
            return
        # render_texts anchors the first line top-left and steps down by 1.1 * font_pt
        y = 20 + len(lines) * 1.1 * self.font_pt
        self.viewer._renderer.render_texts(lines, 20, y, font_pt=self.font_pt, color=self.color)

    @property
    def _label_w(self):
        return max(len(label) for label, _, _ in self.entries)

def attach_log(scene, label, getter, fmt="{:+.2f}", env_idx=0):
    # Display getter() live in the viewer. getter returns a scalar or a (num_envs, ...) tensor; row env_idx is shown.
    # One overlay per scene, created on first call. No-op without a viewer. Call after scene.build()
    if scene.viewer is None:
        return None
    overlay = next((p for p in scene.viewer.plugins if isinstance(p, LogOverlay)), None)
    if overlay is None:
        overlay = scene.viewer.add_plugin(LogOverlay(env_idx=env_idx))
    overlay.entries = overlay.entries + [(label, getter, fmt)]
    return overlay

# Wall-clock physics rate - Reads the scene step counter, so it holds when the viewer skips updates (update_visualizer=False)
class SimRateMeter:
    def __init__(self, scene, window_s=0.5):
        self.scene = scene
        self.window_s = window_s # averaging window, keeps the readout from flickering
        self._t0 = time.perf_counter()
        self._step0 = scene.sim.cur_step_global
        self._text = "-"

    def __call__(self):
        now = time.perf_counter()
        elapsed = now - self._t0
        if elapsed >= self.window_s:
            step = self.scene.sim.cur_step_global
            fps = (step - self._step0) / elapsed
            self._text = f"{fps:6.0f} Hz  ({fps * self.scene.dt:4.2f}x realtime)"
            self._t0, self._step0 = now, step
        return self._text

def attach_sim_rate(scene, window_s=0.5):
    # Overlay line with the measured physics steps/s and the real-time factor (sim seconds per wall second)
    return attach_log(scene, "sim rate", SimRateMeter(scene, window_s), fmt="{}")

