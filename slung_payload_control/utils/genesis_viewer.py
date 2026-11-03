
import math
import numpy as np
import torch
import time

from genesis import Scene
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.utils.misc import tensor_to_array
from genesis.utils.video_encoder import VideoEncoder
from genesis.vis.viewer_plugins import ViewerPlugin

class GenesisViewer:
    def __init__(self, scene: Scene, device, view_idx=0):
        self.scene = scene
        self.device = device
        self.view_idx = view_idx
        self._view_ids = torch.tensor([view_idx], device=device, dtype=torch.long)

        # Draw
        self.route = None
        self.preview = None
        self.tether_lines = None

    def draw_route(self, bank, traj_idx, envs_idx, spacing_s=0.1, max_points=1000, radius=0.01, color=(0.0, 0.0, 0.0, 0.8)):
        if not (envs_idx == self.view_idx).any():
            return # view_idx not reset

        idx = traj_idx[self._view_ids]
        t_end = bank.t_end[idx].item()
        n = int(min(max(t_end / spacing_s, 2), max_points))
        t = torch.linspace(0.0, t_end, n, device=self.device)[None]
        ps = bank.eval(idx, t, orders=(0, ))[0][0] # (n, 3)

        if self.route is not None:
            self.scene.clear_debug_object(self.route)
        self.route = self.scene.draw_debug_trajectory(ps.cpu().numpy().astype(np.float32), radius=radius, color=color)

    def draw_preview(self, ps, radius=0.03):
        ps = ps.cpu().numpy().astype(np.float32)
        if self.preview is None:
            n = len(ps)
            self.preview = []
            for k, p in enumerate(ps):
                scale = 0.5 * k / max(n - 1, 1)
                self.preview.append(
                    self.scene.draw_debug_sphere(
                        pos=p.tolist(),
                        radius=radius,
                        color=(
                            0.9 - scale, 
                            0.4, 
                            1.0,
                            0.9 - scale
                        )
                    )
                )
            return
        
        self.scene.update_debug_objects(self.preview, tuple(gu.trans_to_T(p) for p in ps))

    def draw_tether_lines(self, starts, ends, radius=0.004, color=(0.1, 0.1, 0.1, 1.0)):
        starts = tensor_to_array(starts)
        ends = tensor_to_array(ends)
        n = len(starts)
        if self.tether_lines is None:
            unit = mu.create_line(np.zeros(3), np.array([0.0, 0.0, 1.0]), radius=radius, color=color)
            self.tether_lines = []
            for i in range(n):
                self.tether_lines.append(self.scene.draw_debug_mesh(unit.copy()))
        poses = []
        for p0, p1 in zip(starts, ends):
            vec = p1 - p0
            T = gu.trans_R_to_T(p0, gu.z_up_to_R(vec))
            T[:3, 2] *= np.linalg.norm(vec)
            poses.append(T)
        self.scene.update_debug_objects(self.tether_lines, tuple(poses))

    def attach_log(self, label, getter, fmt="{:+.2f}", view_idx=0):
        overlay = next((p for p in self.scene.viewer.plugins if isinstance(p, LogOverlay)), None)
        if overlay is None:
            overlay = self.scene.viewer.add_plugin(LogOverlay(env_idx=view_idx))
        overlay.entries = overlay.entries + [(label, getter, fmt)]
        return overlay

    def attach_sim_rate(self, window_s=0.5):
        return self.attach_log("sim_rate", SimRateMeter(self.scene, window_s))


class LogOverlay(ViewerPlugin):
    def __init__(self, env_idx=0, font_pt=22, color=(1.0, 1.0, 1.0, 1.0)):
        super().__init__()
        self.env_idx = env_idx
        self.font_pt = font_pt
        self.color = color
        self.entries = []
        self._lines = []

    def update_on_sim_step(self):
        lines = []
        for label, getter, fmt in self.entries:
            v = getter()
            if isinstance(v, torch.Tensor):
                v = v[self.env_idx] if v.dim() > 0 and v.shape[0] > self.env_idx else v
                v = v.detach().cpu().tolist()
            if isinstance(v, str): # pre-formatted by the getter
                txt = v
            elif isinstance(v, (list, tuple)):
                txt = "[" + " ".join(fmt.format(x) for x in v) + "]"
            else:
                txt = fmt.format(v)
            lines.append(f"{label:<{self._label_w}} {txt}")
        self._lines = lines

    def on_draw(self):
        lines = self._lines
        if not lines:
            return
        y = 20 + len(lines) * 1.1 * self.font_pt
        self.viewer._renderer.render_texts(lines, 20, y, font_pt=self.font_pt, color=self.color)

    @property
    def _label_w(self):
        return max(len(label) for label, _ ,_ in self.entries)

class SimRateMeter:
    def __init__(self, scene: Scene, window_s=0.5):
        self.scene = scene
        self.window_s = window_s
        self._t0 = time.perf_counter()
        self._step0 = scene.sim.cur_step_global
        self._text = "-"

    def __call__(self):
        now = time.perf_counter()
        elapsed = now - self._t0
        if elapsed >= self.window_s:
            step = self.scene.sim.cur_step_global
            fps = (step - self._step0) / elapsed
            self._text = f"{fps:6.0f} Hz    ({fps * self.scene.dt:4.2f}x realtime)"
            self._t0 = now
            self._step0 = step
        return self._text

class GenesisRecorder:
    def __init__(self, scene: Scene, fps=30, res=(1280,720), pos=(0.0, 7.0, 3.0), lookat=(0.0, 0.0, 2.0), fov=50):
        self.scene = scene
        self.cam = scene.add_camera(res=res, pos=pos, lookat=lookat, fov=fov, far=200.0, GUI=False, debug=True)
        self.fps = fps
        self.encoder = None
        self.out_path = None
        self.default_view = dict(pos=pos, lookat=lookat)

    def reset_view(self):
        self.cam.set_pose(**self.home)

    def start_record(self, out_path):
        self.encoder = VideoEncoder(out_path, self.fps, is_threaded=True)
        self.out_path = out_path

    def capture(self, pose=None):
        if self.encoder is None:
            return None
        if pose is not None:
            self.cam.set_pose(transform=pose)

        # Render image and encode
        rgb = tensor_to_array(self.cam.render(rgb=True)[0])
        self.encoder.write(rgb[0] if rgb.ndim == 4 else rgb)

    def stop_record(self):
        if self.encoder is None:
            return None
        self.encoder.close()
        self.encoder = None
        print(f"Saved video: {self.out_path}")
        return self.out_path

class GenesisFollower:
    def __init__(self, scene: Scene, dist=4.5, pitch=math.radians(60.0), tau=0.3):
        self.scene = scene
        self.active = False
        self.dist = dist
        self.pitch = pitch
        self.tau = tau
        self.center = None
        self.heading = None

    def toggle(self):
        self.active = not self.active
        self.reset()

    def reset(self):
        self.center = None

    def update(self, drone_pos, payload_pos, yaw, dt):
        if not self.active: 
            return None
        
        center = 0.5 * (tensor_to_array(drone_pos) + tensor_to_array(payload_pos))
        heading = np.array([math.cos(yaw), math.sin(yaw)])
        if self.center is None:
            self.center = center
            self.heading = heading
        else:
            a = dt / (self.tau + dt)
            self.center = self.center + a * (center - self.center)
            h = self.heading + a * (heading - self.heading)
            self.heading = h / max(np.linalg.norm(h), 1e-9)
        back = -math.cos(self.pitch) * np.array([*self.heading, 0.0])
        pos = self.center + self.dist * (back + np.array([0.0, 0.0, math.sin(self.pitch)]))
        pose = gu.pos_lookat_up_to_T(pos, self.center, np.array([0.0, 0.0, 1.0])) # world-z up: no roll creep
        self.scene.viewer.set_camera_pose(pose=pose)
        return pose