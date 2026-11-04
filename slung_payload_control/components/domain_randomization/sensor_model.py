import torch
from dataclasses import fields

from config.config_components import SensorNoiseConfig, Sensor
from utils.torch_math_helpers import perturb_quat

class SensorModel:
    def __init__(self, cfg: SensorNoiseConfig, num_envs, policy_dt, device, dtype):
        self.num_envs = num_envs
        self.device = device
        self.dtype = dtype
        self.cfg = cfg
        self.enabled = cfg.enabled

        t = lambda v: torch.tensor(v, device=device, dtype=dtype)
        steps = lambda s: round(s / policy_dt)
        self.specs = {f.name: getattr(cfg, f.name) for f in fields(cfg) if isinstance(getattr(cfg, f.name), Sensor)}
        self.white = {k: t(s.white) for k, s in self.specs.items()}
        self.bias_max = {k: t(s.bias) for k, s in self.specs.items()}
        self.delay_range = {k: (steps(s.latency_s.lo), steps(s.latency_s.hi)) for k, s in self.specs.items()}
        self.period = {k: max(1, steps(1.0 / s.rate_hz)) if (s.rate_hz > 0 and self.enabled) else 1 for k, s in self.specs.items()}
        self.L = max(hi for _, hi in self.delay_range.values()) + 1 # ring buffer len

        zeros = lambda *shape, dtype=dtype: torch.zeros(shape, device=device, dtype=dtype)
        dim = lambda k: 4 if k.endswith("_att") else 3
        self.hist = {k: zeros(self.L, num_envs, dim(k)) for k in self.specs} # past samples (L, N, D)
        self.held = {k: zeros(num_envs, dim(k)) for k in self.specs} # last reported sample
        self.bias = {k: zeros(num_envs, 3) for k in self.specs}
        self.delay = {k: zeros(num_envs, dtype=torch.long) for k in self.specs}
        self.phase = {k: zeros(num_envs, dtype=torch.long) for k in self.specs}

        self.slots = torch.arange(self.L, device=device)[:, None, None] # ring buffer slot ids (L, 1, 1)
        self.fresh = torch.ones(num_envs, device=device, dtype=torch.bool) # history must be refilled (just reset)
        self.k = zeros(dtype=torch.long) # policy-step counter
        self.scale = t(1.0 if self.enabled else 0.0)
        self.env_ids = torch.arange(num_envs, device=device)

    def compile(self):
        self.measure = torch.compile(self.measure, dynamic=False)
        self._reset = torch.compile(self._reset, dynamic=False)

    def reset(self, envs_idx):
        mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        mask[envs_idx] = True
        self._reset(mask)

    def _reset(self, mask):
        m = mask[:, None]
        for k in self.specs:
            lo, hi = self.delay_range[k] if self.enabled else (0, 0)
            self.bias[k] = torch.where(m, (2.0 * torch.rand_like(self.bias[k]) - 1.0) * self.bias_max[k], self.bias[k])
            self.delay[k] = torch.where(mask, torch.randint_like(self.delay[k], lo, hi + 1), self.delay[k])
            self.phase[k] = torch.where(mask, torch.randint_like(self.phase[k], 0, self.period[k]), self.phase[k])
        self.fresh |= mask
        

    def set_scale(self, v):
        self.scale.fill_(v)

    def measure(self, pos, quat, vel, rates):
        slot = self.k % self.L
        out = {}
        for i, body in enumerate(("drone", "payload")):
            for q, x in (("pos", pos), ("vel", vel), ("att", quat), ("gyro", rates)):
                out[f"{body}_{q}"] = self._channel(f"{body}_{q}", x[:, i], slot)
        self.fresh[:] = False
        self.k += 1

        stack = lambda q: torch.stack([out[f"drone_{q}"], out[f"payload_{q}"]], dim=1)
        return stack("pos"), stack("att"), stack("vel"), stack("gyro")

    def _channel(self, k, x, slot):
        # latency: push the newest sample, read the on delay[k] steps back
        hist = torch.where((self.slots == slot) | self.fresh[None, :, None], x[None], self.hist[k])
        self.hist[k] = hist
        x = hist[(slot - self.delay[k]) % self.L, self.env_ids]

        # bias + white: 
        noise = self.scale * (self.bias[k] + self.white[k] * torch.randn_like(self.bias[k]))
        x = perturb_quat(x, noise, keepdim=True) if k.endswith("_att") else x + noise

        # update rate: refresh on tick, hold in between
        tick = ((self.k + self.phase[k]) %self.period[k] == 0) | self.fresh
        self.held[k] = torch.where(tick[:, None], x, self.held[k])
        return self.held[k]



