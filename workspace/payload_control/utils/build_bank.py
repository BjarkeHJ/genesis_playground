"""Build the trajectory bank for the current configs (RouteConfig, RouteSamplerConfig, TrajectoryBankConfig) and cache it
under logs/route_bank. The env only loads it: run this after changing those configs or the generator code
(components/route.py, components/trajectory.py), which change the cache key. Bank size, seed and keyword weights are
set in TrajectoryBankConfig, not here, so the env finds the same bank.

    python utils/build_bank.py            build if missing, then print a summary
    python utils/build_bank.py --force    rebuild even if cached
    python utils/build_bank.py --list     cached banks, marking the one matching the current configs
    python utils/build_bank.py --clean    delete cached banks that don't match the current configs
"""
import argparse
import dataclasses
import glob
import os
import sys

import torch

PC_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # payload_control/
sys.path.insert(0, PC_DIR)
from components.reference import BANK_DIR, TrajectoryBank  # noqa: E402
from config.config_env import RouteConfig, RouteSamplerConfig, TrajectoryBankConfig  # noqa: E402

def summary(bank, path):
    q = lambda x, p: torch.quantile(x.float(), p).item()
    hover = bank.duration == 0
    print(f"\n{path}  ({os.path.getsize(path) / 1e6:.0f} MB)")
    print(f"  routes      {bank.size}, per keyword " + ", ".join(
        f"{k} {n}" for k, n in zip(bank.keywords, torch.bincount(bank.keyword_id, minlength=len(bank.keywords)).tolist())))
    print(f"  hover       {hover.sum().item()}")
    print(f"  segments    median {q(bank.num_segments, 0.5):.0f}, p99 {q(bank.num_segments, 0.99):.0f}, max "
          f"{bank.num_segments.max().item()} (padded to {bank.max_segments})")
    d = bank.duration[~hover]
    print(f"  duration    median {q(d, 0.5):.1f} s, p95 {q(d, 0.95):.1f} s, max {d.max().item():.1f} s")
    print(f"  v_des       {bank.v_des.min().item():.2f} - {bank.v_des.max().item():.2f} m/s")

def list_banks(current):
    paths = sorted(glob.glob(os.path.join(BANK_DIR, "bank_*.pt")))
    if not paths:
        print(f"No cached banks in {BANK_DIR}")
    for p in paths:
        cfgs = torch.load(p, map_location="cpu").get("configs", {})
        mark = "  <- current configs" if p == current else ""
        print(f"{os.path.basename(p)}  {os.path.getsize(p) / 1e6:.0f} MB{mark}")
        for name, c in cfgs.items():
            print(f"    {name}: {c}")

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true", help="Rebuild even if a bank for the current configs is cached")
    parser.add_argument("--workers", type=int, default=None, help="Build processes (default TrajectoryBankConfig.num_workers)")
    parser.add_argument("--list", action="store_true", help="List cached banks")
    parser.add_argument("--clean", action="store_true", help="Delete cached banks that don't match the current configs")
    args = parser.parse_args()

    bank_cfg, route_cfg, sampler_cfg = TrajectoryBankConfig(), RouteConfig(), RouteSamplerConfig()
    if args.workers is not None: # not part of the cache key
        bank_cfg = dataclasses.replace(bank_cfg, num_workers=args.workers)
    path = TrajectoryBank.cache_path(bank_cfg, route_cfg, sampler_cfg)

    if args.list:
        list_banks(path)
        return
    if args.clean:
        stale = [p for p in glob.glob(os.path.join(BANK_DIR, "bank_*.pt")) if p != path]
        for p in stale:
            os.remove(p)
            print("removed", p)
        print(f"{len(stale)} stale bank(s) removed")
        return

    if os.path.isfile(path) and not args.force:
        print(f"Bank for the current configs is already cached (use --force to rebuild)")
    else:
        TrajectoryBank.build_and_save(bank_cfg, route_cfg, sampler_cfg)
    summary(TrajectoryBank.load(bank_cfg, route_cfg, sampler_cfg, verbose=False), path)

if __name__ == "__main__":
    main()
