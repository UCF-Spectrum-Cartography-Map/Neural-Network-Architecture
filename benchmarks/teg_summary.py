#!/usr/bin/env python3
"""Power and RAM summary from a tegrastats log. Works on the Orin Nano/NX (VDD_IN rail)
and the AGX Orin (no VDD_IN: sums VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0).
Usage: python3 teg_summary.py teg_fp16.log [more logs...]"""
import os
import re
import sys

paths = [p for p in sys.argv[1:] if os.path.exists(p)]
if not paths:
    sys.exit("No tegrastats logs found (did the benchmark runs finish? is tegrastats installed?)")
for path in paths:
    rails = {}
    power, ram = [], []
    for line in open(path):
        rails = {k: int(v) for k, v in re.findall(r"([A-Z][A-Z0-9_]+) (\d+)mW", line)}
        p = rails.get("VDD_IN") or sum(rails.get(k, 0) for k in ("VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0"))
        if p:
            power.append(p / 1000)
        ram += [int(x) for x in re.findall(r"RAM (\d+)/", line)]
    if not power:
        print(f"{path}: no power readings found (rail names: {sorted(rails)})")
        continue
    print(f"{path}: power mean {sum(power) / len(power):.1f} W, max {max(power):.1f} W, "
          f"peak RAM {max(ram) / 1024:.1f} GB ({len(power)} readings)")
