import torch
import math
from dataclasses import fields
from config.config_components import DomainRandomizationConfig, Uniform, LogUniform

class DomainRandomizer:
    def __init__(self, cfg: DomainRandomizationConfig, num_envs: int, device, dtype):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.dtype = dtype
        self.enabled = self.cfg.enabled

        # Find every Uniform/LogUniform types in cfg
        specs = {f.name: getattr(cfg, f.name) for f in fields(cfg) if isinstance(getattr(cfg, f.name), Uniform)}
        self.col = {name: i for i, name in enumerate(specs)}
        t = lambda v: torch.tensor(v, device=self.device, dtype=self.dtype)
        is_log = [isinstance(s, LogUniform) for s in specs.values()]
        f = lambda s, log: math.log(s) if log else s
        self.lo = t([f(s.lo, l) for s, l in zip(specs.values(), is_log)])
        self.hi = t([f(s.hi, l) for s, l in zip(specs.values(), is_log)])
        self.is_log = torch.tensor(is_log, device=self.device)
        self.nominal = t([s.nominal for s in specs.values()])

        self.values = self.nominal.repeat(num_envs, 1)
        self._bindings = []

    def __getitem__(self, name):
            return self.values[:, self.col[name]]
    
    def bind(self, fn):
        self._bindings.append(fn)

    def randomize(self, envs_idx):
        n = len(envs_idx)

        # Sample random scalings 
        if self.enabled:
             x = self.lo + torch.rand((n, len(self.col)), device=self.device, dtype=self.dtype) * (self.hi - self.lo)
             x = torch.where(self.is_log, torch.exp(x), x)
        else:
             x = self.nominal.expand(n, -1)
        self.values[envs_idx] = x

        # Use bound function from spc_env.py to set DR parameters
        p = Params(x, self.col)
        for fn in self._bindings:
             fn(envs_idx, p)

    def priviledged_obs(self, normalize=True):
        # True sampled DR values (num_envs, num_params), column order given by self.col
        if not normalize:
            return self.values
        # Map to [-1, 1] in sampling space (log-space for LogUniform)
        x = torch.where(self.is_log, torch.log(self.values), self.values)
        return 2.0 * (x - self.lo) / (self.hi - self.lo) - 1.0

class Params:
    def __init__(self, x, col):
        self.x = x
        self.col = col

    def __getitem__(self, name):
        return self.x[:, self.col[name]]