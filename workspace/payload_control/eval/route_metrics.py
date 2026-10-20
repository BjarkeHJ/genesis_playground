"""Shared by eval_route.py and eval_route_batch.py: rollout of fixed routes (one per env) into histories, per-case
metrics and the table printout. Imports no configs (the scripts load the run's config snapshot before anything else)."""
import math

import numpy as np
import torch

MOVING_SPEED = 0.3 # reference speed above which along/cross-track and lag are defined [m/s]
FINAL_WINDOW_S = 1.0 # final error: mean over the last second of the settle phase [s]

# ====== ROLLOUT ======
def payload_roll_pitch(e):
    # ZYX roll/pitch from gravity in payload body axes g_b = [sin(p), -sin(r)cos(p), -cos(r)cos(p)] (yaw-independent)
    g = e.payload_proj_g
    return torch.stack([torch.atan2(-g[:, 1], -g[:, 2]), torch.asin(torch.clamp(g[:, 0], -1.0, 1.0))], dim=-1)

def cone_angle(swing_xy):
    # Cable angle from vertical, from the heading-frame swing angles atan2(r_x, -r_z), atan2(r_y, -r_z)
    return torch.atan(torch.sqrt(torch.tan(swing_xy[..., 0]) ** 2 + torch.tan(swing_xy[..., 1]) ** 2))

def rollout(e, policy, durations, settle_s, dt):
    # Run every env through its fixed route once, until --settle after the longest route ends -> numpy histories
    # (n, T, ...) and the crash step per env (-1: none). Crashed envs are auto-reset by the env and no longer recorded
    n, dev = e.num_envs, e.device
    T = int(math.ceil((float(durations.max()) + settle_s) / dt))
    shapes = {"payload": 3, "ref": 3, "ref_vel": 3, "pl_rp": 2, "swing_xy": 2, "actions": 4, "actions_raw": 4,
              "swing": 0, "cone": 0, "yaw_err": 0, "drone_tilt": 0, "payload_speed": 0}
    hist = {k: torch.zeros((n, T, d) if d else (n, T), device=dev) for k, d in shapes.items()}
    crash_step = torch.full((n, ), -1, device=dev, dtype=torch.long)
    zero = torch.zeros(1, device=dev)

    obs = e.reset()
    with torch.inference_mode():
        for k in range(T):
            obs, _, dones, _ = e.step(policy(obs))
            crash_step[dones.bool() & (crash_step < 0)] = k
            active = crash_step < 0
            vals = {"payload": e.payload_pos, "ref": e.ref_pos, "ref_vel": e.ref.eval(zero, 1)[1, :, 0],
                    "pl_rp": payload_roll_pitch(e), "swing_xy": e.payload_swing_angles, "actions": e.actions,
                    "actions_raw": e.actions_raw, "swing": e.swing_amp, "cone": cone_angle(e.payload_swing_angles),
                    "yaw_err": e.payload_yaw_err.abs(), "drone_tilt": torch.acos(torch.clamp(-e.drone_proj_g[:, 2], -1.0, 1.0)),
                    "payload_speed": torch.norm(e.payload_vel, dim=1)}
            for key, val in vals.items():
                hist[key][active, k] = val[active]
    return {k: v.cpu().numpy() for k, v in hist.items()}, crash_step.cpu().numpy()

def recorded_steps(crash, duration, settle_s, dt):
    # Steps of a case's history that hold data: until the crash, else until settle_s after its route end
    return int(crash) if crash >= 0 else int(round((duration + settle_s) / dt))

# ====== METRICS ======
def case_metrics(case, h, i, crash, duration, settle_s, settle_tol, dt):
    T = h["payload"].shape[1]
    t = (np.arange(T) + 1) * dt
    end = recorded_steps(crash, duration, settle_s, dt)
    route_ph = (t <= duration) & (np.arange(T) < end) # following the route
    hold_ph = (t > duration) & (np.arange(T) < end) # after the route end: final stop and hold
    if not route_ph.any(): # hover: the whole case is the hold
        route_ph = hold_ph
    err_vec = h["ref"][i] - h["payload"][i]
    err = np.linalg.norm(err_vec, axis=-1)
    v = h["ref_vel"][i]
    speed = np.linalg.norm(v, axis=-1)
    moving = route_ph & (speed > MOVING_SPEED)
    tan = v / np.maximum(speed, 1e-9)[:, None]
    along = np.sum(err_vec * tan, axis=-1) # > 0: payload behind the reference
    cross = np.linalg.norm(err_vec - along[:, None] * tan, axis=-1)
    dev = case["route"].deviation(h["payload"][i][:end]) if end > 0 else np.zeros(1)
    ref_dev = case["route"].deviation(h["ref"][i][:end]) if end > 0 else np.zeros(1)

    final = hold_ph & (t > duration + settle_s - FINAL_WINDOW_S)
    above = np.nonzero(hold_ph & (err > settle_tol))[0]
    if crash >= 0 or not hold_ph.any():
        settle = float("nan")
    elif len(above) == 0:
        settle = 0.0
    elif above[-1] == np.nonzero(hold_ph)[0][-1]:
        settle = float("nan") # still outside the tolerance at the end
    else:
        settle = float(t[above[-1]] - duration)

    mean = lambda x, m: float(np.mean(x[m])) if m.any() else float("nan")
    return {
        "name": case["name"], "group": case["group"], "speed": case["speed"], "rep": case.get("rep", 0),
        "duration_s": duration, "crashed": bool(crash >= 0), "crash_t": float(crash * dt) if crash >= 0 else None,
        "err_mean": mean(err, route_ph), "err_rms": float(np.sqrt(np.mean(err[route_ph] ** 2))) if route_ph.any() else float("nan"),
        "err_max": float(np.max(err[route_ph])) if route_ph.any() else float("nan"),
        "cross_mean": mean(cross, moving), "lag_s": mean(along / np.maximum(speed, 1e-9), moving),
        "route_dev_mean": float(np.mean(dev)), "route_dev_max": float(np.max(dev)),
        "ref_dev_mean": float(np.mean(ref_dev)), "ref_dev_max": float(np.max(ref_dev)),
        "swing_mean_deg": math.degrees(mean(h["swing"][i], route_ph)),
        "swing_max_deg": math.degrees(float(np.max(h["swing"][i][: max(end, 1)]))),
        "yaw_err_mean_deg": math.degrees(mean(h["yaw_err"][i], route_ph)),
        "drone_tilt_max_deg": math.degrees(float(np.max(h["drone_tilt"][i][: max(end, 1)]))),
        "final_err": mean(err, final), "settle_s": settle,
    }

def print_table(rows, title):
    cols = [("crash", "crash_rate", "{:>5.0%}"), ("err", "err_mean", "{:5.2f}"), ("rms", "err_rms", "{:5.2f}"),
            ("max", "err_max", "{:5.2f}"), ("cross", "cross_mean", "{:5.2f}"), ("lag[s]", "lag_s", "{:6.2f}"),
            ("dev", "route_dev_mean", "{:5.2f}"), ("devmax", "route_dev_max", "{:6.2f}"), ("refdev", "ref_dev_max", "{:6.2f}"),
            ("swing", "swing_mean_deg", "{:5.1f}"),
            ("swmax", "swing_max_deg", "{:5.1f}"), ("yaw", "yaw_err_mean_deg", "{:5.1f}"), ("tilt", "drone_tilt_max_deg", "{:5.1f}"),
            ("final", "final_err", "{:5.2f}"), ("settle", "settle_s", "{:6.1f}")]
    w = max(len(r["name"]) for r in rows) + 2
    print(f"\n{title}\n" + "case".ljust(w) + " ".join(f"{c:>{max(len(c), 5)}}" for c, _, _ in cols))
    for r in rows:
        print(r["name"].ljust(w) + " ".join(f.format(r[k]).rjust(max(len(c), 5)) if k in r and not (isinstance(r[k], float) and np.isnan(r[k]))
                                            else "-".rjust(max(len(c), 5)) for c, k, f in cols))
