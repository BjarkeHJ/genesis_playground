import torch
import math

class LowPassFilter2:
    # 2nd order Butterworth
    def __init__(self, shape, sample_freq, cutoff_freq, device, dtype):
        self.d1 = torch.zeros(shape, device=device, dtype=dtype)
        self.d2 = torch.zeros(shape, device=device, dtype=dtype)
        self.enabled = sample_freq > 0.0 and 0.0 < cutoff_freq < sample_freq / 2.0
        if not self.enabled:
            self.b0, self.b1, self.b2, self.a1, self.a2 = 1.0, 0.0, 0.0, 0.0, 0.0
            return

        cutoff = max(cutoff_freq, sample_freq * 0.001)
        ohm = math.tan(math.pi / (sample_freq / cutoff))
        c = 1.0 + 2.0 * math.cos(math.pi / 4.0) * ohm + ohm * ohm

        self.b0 = ohm * ohm / c
        self.b1 = 2.0 * self.b0
        self.b2 = self.b0
        self.a1 = 2.0 * (ohm * ohm - 1.0) / c
        self.a2 = (1.0 - 2.0 * math.cos(math.pi / 4.0) * ohm + ohm * ohm) / c

    def apply(self, x):
        if not self.enabled:
            return x
        d0 = x - self.d1 * self.a1 - self.d2 * self.a2
        y = d0 * self.b0 + self.d1 * self.b1 + self.d2 * self.b2
        self.d2 = self.d1
        self.d1 = d0
        return y

    def reset(self, envs_idx, value):
        # steady state of the delay line for a constant input
        dval = value / (1.0 + self.a1 + self.a2)
        self.d1[envs_idx] = dval
        self.d2[envs_idx] = dval

class AlphaFilter:
    # 1st order low-pass y += alpha * (x - y), alpha = dt / (dt + 1/(2pi*fc))
    def __init__(self, shape, sample_freq, cutoff_freq, device, dtype):
        self.state = torch.zeros(shape, device=device, dtype=dtype)
        if sample_freq > 0.0 and 0.0 < cutoff_freq < sample_freq / 2.0:
            dt = 1.0 / sample_freq
            tau = 1.0 / (2.0 * math.pi * cutoff_freq)
            self.alpha = dt / (dt + tau)
        else:
            self.alpha = 1.0 

    def apply(self, x):
        self.state = self.state + self.alpha * (x - self.state)
        return self.state

    def reset(self, envs_idx, value):
        self.state[envs_idx] = value