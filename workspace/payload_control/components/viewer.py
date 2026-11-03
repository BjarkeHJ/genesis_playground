import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.utils.misc import tensor_to_array
from genesis.utils.video_encoder import VideoEncoder
from genesis.vis.viewer_plugins import ViewerPlugin

import math
import time
import numpy as np
import torch

# Viewer-only helpers. The viewer renders env 0, so everything here shows env 0 (rendered_envs_idx in the scene setup)

class FollowCamera:
    # Chase view of env 0: above and behind the drone (opposite its heading), looking down at `pitch` onto the
    # drone-payload midpoint. Low-pass filtered (time constant tau) so attitude and yaw jitter don't shake the view
    def __init__(self, dist=4.5, pitch=math.radians(60.0), tau=0.3):
        self.dist, self.pitch, self.tau = dist, pitch, tau
        self.center = None # filtered look-at point, None: snap on the next update
        self.heading = None # filtered unit heading (xy)

    def reset(self):
        self.center = None

    def update(self, drone_pos, payload_pos, yaw, dt):
        # -> camera-to-world transform (4, 4) for the new state
        center = 0.5 * (tensor_to_array(drone_pos) + tensor_to_array(payload_pos))
        heading = np.array([math.cos(yaw), math.sin(yaw)])
        if self.center is None:
            self.center, self.heading = center, heading
        else:
            a = dt / (self.tau + dt)
            self.center = self.center + a * (center - self.center)
            h = self.heading + a * (heading - self.heading)
            self.heading = h / max(np.linalg.norm(h), 1e-9) # through a 180° flip h passes near 0: keep the last direction
        back = -math.cos(self.pitch) * np.array([*self.heading, 0.0])
        pos = self.center + self.dist * (back + np.array([0.0, 0.0, math.sin(self.pitch)]))
        return gu.pos_lookat_up_to_T(pos, self.center, np.array([0.0, 0.0, 1.0])) # world-z up: no roll creep

class VideoRecorder:
    # Offscreen camera on env 0 encoding one frame per policy step. Frames are captured by the env (not by the camera's
    # own recording, which spaces frames by the viewer realtime_factor -- set to the decimation for viewer pacing, see
    # attach_viewer_logs -- and would play the video sped up by it)
    def __init__(self, scene, res=(1280, 720), pos=(0.0, 7.0, 3.0), lookat=(0.0, 0.0, 2.0), fov=50):
        # debug=True: cameras skip debug objects (reference markers, route) otherwise. Must be added before scene.build()
        self.cam = scene.add_camera(res=res, pos=pos, lookat=lookat, fov=fov, far=200.0, GUI=False, debug=True)
        self.encoder = None
        self.path = None

    def start(self, path, fps):
        self.encoder = VideoEncoder(path, fps, is_threaded=True)
        self.path = path

    def capture(self):
        if self.encoder is None:
            return
        rgb = tensor_to_array(self.cam.render(rgb=True)[0])
        self.encoder.write(rgb[0] if rgb.ndim == 4 else rgb)

    def stop(self):
        if self.encoder is None:
            return None
        self.encoder.close()
        self.encoder = None
        print(f"Saved video: {self.path}")
        return self.path

class TargetMarker:
    # Debug target [x, y, z, yaw_ref] (the current reference point): sphere at the position, arrow along yaw_ref
    def __init__(self, scene, radius=0.05, arrow_len=0.3):
        self.scene = scene
        self.radius = radius
        self.arrow_len = arrow_len
        self.objs = [] # [sphere, arrow], drawn on the first update

    def update(self, command):
        pos = command[:3].cpu().numpy().astype(np.float32)
        yaw_ref = command[3].item()
        heading = np.array([math.cos(yaw_ref), math.sin(yaw_ref), 0.0], dtype=np.float32)
        arrow_pos = pos + self.radius * heading

        if not self.objs:
            self.objs = [
                self.scene.draw_debug_sphere(pos=pos.tolist(), radius=self.radius, color=(1.0, 0.0, 0.0, 1.0)),
                self.scene.draw_debug_arrow(pos=arrow_pos.tolist(), vec=(self.arrow_len * heading).tolist(), radius=0.008, color=(0.0, 0.0, 0.0, 1.0)),
            ]
            return
        # Move the existing nodes: clearing and re-drawing every step rebuilds meshes under the viewer lock (~40% of sim rate)
        self.scene.update_debug_objects(self.objs, (gu.trans_to_T(pos), gu.trans_R_to_T(arrow_pos, gu.z_up_to_R(heading))))

class RoutePath:
    # Debug polyline of a full reference route. Redrawn only when the route changes (a merged mesh: clearing and drawing
    # it rebuilds geometry, too slow for every step)
    def __init__(self, scene, radius=0.01, color=(0.0, 0.0, 0.0, 0.8)):
        self.scene, self.radius, self.color = scene, radius, color
        self.obj = None

    def set(self, poss):
        # poss (P, 3) points along the route; fewer than 2 (hover) just clears it
        if self.obj is not None:
            self.scene.clear_debug_object(self.obj)
        poss = poss.cpu().numpy().astype(np.float32)
        self.obj = self.scene.draw_debug_trajectory(poss, radius=self.radius, color=self.color) if len(poss) >= 2 else None

class PreviewMarkers:
    # Spheres at the reference preview points the policy observes, fading with horizon. Moved every step, never redrawn
    def __init__(self, scene, n, radius=0.04):
        self.scene, self.n, self.radius = scene, n, radius
        self.objs = [] # drawn on the first update

    def update(self, poss):
        # poss (n, 3) preview points
        poss = poss.cpu().numpy().astype(np.float32)
        if not self.objs:
            self.objs = [self.scene.draw_debug_sphere(pos=p.tolist(), radius=self.radius, color=(0.1, 0.4, 1.0, 0.9 - 0.6 * k / max(self.n - 1, 1)))
                         for k, p in enumerate(poss)]
            return
        self.scene.update_debug_objects(self.objs, tuple(gu.trans_to_T(p) for p in poss))

class TetherLines:
    # Straight segments drone attach point -> payload attach point, visual only. Each is a unit-length cylinder along z
    # (0..1), placed every step by a pose that rotates z onto the segment and scales it by the segment length
    def __init__(self, scene, n, radius=0.004, color=(0.1, 0.1, 0.1, 1.0)):
        self.scene, self.n, self.radius, self.color = scene, n, radius, color
        self.objs = [] # drawn on the first update

    def update(self, starts, ends):
        # starts, ends (n, 3) drone and payload attach points
        starts, ends = tensor_to_array(starts), tensor_to_array(ends)
        if not self.objs:
            unit = mu.create_line(np.zeros(3), np.array([0.0, 0.0, 1.0]), radius=self.radius, color=self.color)
            self.objs = [self.scene.draw_debug_mesh(unit.copy()) for _ in range(self.n)]
        poses = []
        for p0, p1 in zip(starts, ends):
            vec = p1 - p0
            T = gu.trans_R_to_T(p0, gu.z_up_to_R(vec))
            T[:3, 2] *= np.linalg.norm(vec)
            poses.append(T)
        self.scene.update_debug_objects(self.objs, tuple(poses))

def attach_viewer_logs(env):
    # The viewer paces one sim_dt per update, but env.step() updates it once per decimation substeps: scale to stay real time
    env.scene.viewer.realtime_factor = env.decimation
    # Text overlay of env 0 state. Buffers are heading frame
    attach_sim_rate(env.scene)
    attach_log(env.scene, "|pos err| [m]", lambda: torch.norm(env.payload_pos_err, dim=1))
    attach_log(env.scene, "|vel err| [m/s]", lambda: torch.norm(env.payload_vel_err, dim=1))
    attach_log(env.scene, "payload |v| [m/s]", lambda: torch.norm(env.payload_vel, dim=1))
    attach_log(env.scene, "payload yaw error [deg]", lambda: torch.rad2deg(env.payload_yaw_err))
    attach_log(env.scene, "reward", lambda: env.rew_buf, fmt="{:+.3f}")

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
