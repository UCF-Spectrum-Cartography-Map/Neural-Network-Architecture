#!/usr/bin/env python3
"""GNN radio-map reconstruction benchmark (accuracy + on-device cost).

Runs the ViG GNN from models/GNN/GNN_inference_run.ipynb on held-out parquet maps at FIXED
sampling densities, with the SAME saved sample masks every run, and writes:

  <out>/per_sample_<tag>.csv  one row per (map, density): dB errors, emitter error
  <out>/summary_<tag>.csv     per density: mean + 95% bootstrap CI
  <out>/run_info_<tag>.json   device, JetPack, torch version, settings, latency p50/p95/p99, RAM, power
  <out>/masks/*.npz           the sample masks (reused on the next run, so runs are comparable)
  <out>/tegrastats.log        power/RAM log while the timing runs (Jetson only)
<tag> is e.g. fp16_notx_uniform.

Metrics are computed on pixels the model had to infer: not sampled, and (by default) not
inside buildings. Emitter error = distance between the strongest predicted pixel (lowest path
loss) and the true tx_origin.

Example (on the Jetson, from the repo root):
    python3 benchmarks/benchmark_reconstruction.py --weights benchmarks/GNN_no_tx.pth \
        --test-dir data/gnn_inference_test --out-dir benchmarks/gnn_bench --fp16
"""
import argparse, glob, json, os, platform, resource, shutil, subprocess, time, zlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

H = W = 256
MIN_DB, MAX_DB = 30.0, 140.0  # must match training normalization


# ---------------------------------------------------------------- model (same as GNN_inference_run.ipynb)
def knn_graph(x, k):
    B, C, N, _ = x.shape
    x_sq = x.squeeze(-1).permute(0, 2, 1)
    dist = torch.cdist(x_sq, x_sq, p=2)
    _, idx = dist.topk(k + 1, dim=-1, largest=False)
    return idx[:, :, 1:]


class DynConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, k=9, dilation=1, act=None):
        super().__init__()
        self.k = k
        self.conv = nn.Conv2d(in_channels * 2, out_channels, 1)
        self.act = act

    def forward(self, x):
        B, C, N, _ = x.shape
        idx = knn_graph(x, self.k)
        x_flat = x.squeeze(-1).permute(0, 2, 1)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, C)
        neighbors = torch.gather(x_flat.unsqueeze(2).expand(-1, -1, self.k, -1), 1, idx_exp)
        center = x_flat.unsqueeze(2).expand_as(neighbors)
        edge = torch.cat([center, neighbors - center], dim=-1)
        agg = edge.max(dim=2).values.permute(0, 2, 1).unsqueeze(-1)
        out = self.conv(agg)
        return self.act(out) if self.act is not None else out


class GrapherModule(nn.Module):
    def __init__(self, in_channels, hidden_channels, k=9, dilation=1):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels))
        self.graph_conv = nn.Sequential(DynConv2d(in_channels, hidden_channels, k, dilation),
                                        nn.BatchNorm2d(hidden_channels), nn.GELU())
        self.fc2 = nn.Sequential(nn.Conv2d(hidden_channels, in_channels, 1), nn.BatchNorm2d(in_channels))

    def forward(self, x):
        return self.fc2(self.graph_conv(self.fc1(x))) + x


class FFNModule(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Conv2d(in_channels, hidden_channels, 1), nn.BatchNorm2d(hidden_channels), nn.GELU())
        self.fc2 = nn.Sequential(nn.Conv2d(hidden_channels, in_channels, 1), nn.BatchNorm2d(in_channels))

    def forward(self, x):
        return self.fc2(self.fc1(x)) + x


class ViGBlock(nn.Module):
    def __init__(self, channels, k=9):
        super().__init__()
        self.grapher = GrapherModule(channels, channels * 2, k)
        self.ffn = FFNModule(channels, channels * 4)

    def forward(self, x):
        return self.ffn(self.grapher(x))


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=8):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, E, n_h, n_w = x.shape
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x.transpose(1, 2).reshape(B, E, n_h, n_w)


class PatchUnembed(nn.Module):
    def __init__(self, embed_dim, out_channels, patch_size=8):
        super().__init__()
        self.proj = nn.ConvTranspose2d(embed_dim, out_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)


class ViGPatchModel(nn.Module):
    # DropPath is identity at eval time, so it is left out; state_dict keys are unchanged.
    def __init__(self, in_channels=3, out_channels=1, embed_dim=64, num_blocks=4, patch_size=8, k=9):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        self.blocks = nn.ModuleList([ViGBlock(embed_dim, k=k) for _ in range(num_blocks)])
        self.patch_unembed = PatchUnembed(embed_dim, out_channels, patch_size)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        _, _, oh, ow = x.shape
        ph = (self.patch_size - oh % self.patch_size) % self.patch_size
        pw = (self.patch_size - ow % self.patch_size) % self.patch_size
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        x = self.patch_embed(x)
        B, E, n_h, n_w = x.shape
        x = x.reshape(B, E, n_h * n_w, 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x.reshape(B, E, n_h, n_w))
        return self.patch_unembed(x)[:, :, :oh, :ow]


def load_model(path, device):
    model = ViGPatchModel()
    if path:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # older JetPack PyTorch builds
            state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    return model.to(device).eval()


# ---------------------------------------------------------------- data + masks
def load_map(path):
    row = pd.read_parquet(path, engine="pyarrow").iloc[0]
    env = np.asarray(row["building_mask"], dtype=np.float32).reshape(H, W)
    tx = np.asarray(row["tx_origin"], dtype=np.float32).reshape(H, W)
    gt = np.asarray(row["path_loss"], dtype=np.float32).reshape(H, W)
    return env, tx, gt


def make_mask(density, env, rng, kind, avoid_buildings):
    """Boolean HxW mask of sampled pixels. 'uniform' = random pixels (as in training);
    'walk' = pixels along random-walk paths, closer to what a driving rover collects."""
    n = max(1, int(round(density / 100 * H * W)))
    allowed = (env < 0.5) if avoid_buildings else np.ones((H, W), bool)
    mask = np.zeros((H, W), bool)
    if kind == "uniform":
        idx = np.flatnonzero(allowed)
        mask.flat[rng.choice(idx, size=min(n, idx.size), replace=False)] = True
        return mask
    free = np.argwhere(allowed)
    r, c = free[rng.integers(len(free))]
    steps = 0
    while mask.sum() < n and steps < 50 * n:
        mask[r, c] = True
        dr, dc = [(0, 1), (0, -1), (1, 0), (-1, 0)][rng.integers(4)]
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and allowed[nr, nc]:
            r, c = nr, nc
        if rng.random() < 0.002:  # occasionally start a new path
            r, c = free[rng.integers(len(free))]
        steps += 1
    return mask


def get_mask(mask_dir, fname, density, env, args):
    # Masks are saved and reused so every run (fp32/fp16, with/without tx, new weights) sees the same samples.
    path = os.path.join(mask_dir, f"{os.path.splitext(fname)[0]}_{args.mask_kind}_{density:g}.npz")
    if os.path.exists(path):
        return np.load(path)["mask"]
    seed = zlib.crc32(f"{fname}|{density}|{args.mask_kind}".encode())  # stable across runs and machines
    mask = make_mask(density, env, np.random.default_rng([args.seed, seed]), args.mask_kind, args.avoid_buildings)
    np.savez_compressed(path, mask=mask)
    return mask


def build_input(env, tx, gt, mask, use_tx):
    sparse = np.zeros((H, W), np.float32)
    sparse[mask] = np.clip((gt[mask] - MIN_DB) / (MAX_DB - MIN_DB), 0, 1)
    tx_ch = tx if use_tx else np.zeros_like(tx)
    return torch.from_numpy(np.stack([env, tx_ch, sparse])).float().unsqueeze(0)


# ---------------------------------------------------------------- timing / device info
def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def forward(model, x, fp16):
    if fp16 and x.is_cuda:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(x).float()
    return model(x)


def start_tegrastats(out_dir):
    if not shutil.which("tegrastats"):
        return None
    log = open(os.path.join(out_dir, "tegrastats.log"), "w")
    return subprocess.Popen(["tegrastats", "--interval", "100"], stdout=log, stderr=subprocess.DEVNULL), log


def parse_tegrastats(path):
    """Mean/max total input power (W) and peak used RAM (MB) from a tegrastats log."""
    import re
    ram, power = [], []
    for line in open(path):
        m = re.search(r"RAM (\d+)/(\d+)MB", line)
        if m:
            ram.append(int(m.group(1)))
        # Orin: 'VDD_IN 5432mW/5100mW' (instant/avg); fall back to summing rails if absent
        m = re.search(r"VDD_IN (\d+)mW", line)
        if m:
            power.append(int(m.group(1)) / 1000)
    return {"ram_used_peak_MB": max(ram) if ram else None,
            "power_mean_W": float(np.mean(power)) if power else None,
            "power_max_W": float(np.max(power)) if power else None}


def device_info(device):
    info = {"python": platform.python_version(), "torch": torch.__version__, "device": str(device)}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    for f, key in [("/etc/nv_tegra_release", "l4t")]:
        if os.path.exists(f):
            info[key] = open(f).readline().strip()
    if shutil.which("nvpmodel"):
        r = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True)
        info["nvpmodel"] = " ".join(r.stdout.split())
    return info


# ---------------------------------------------------------------- metrics
def score(pred_db, gt_db, env, tx, mask, args):
    eval_px = ~mask
    if args.exclude_buildings:
        eval_px &= env < 0.5
    err = pred_db[eval_px] - gt_db[eval_px]
    out = {"n_eval_px": int(eval_px.sum()),
           "rmse_db": float(np.sqrt(np.mean(err ** 2))),
           "mae_db": float(np.mean(np.abs(err))),
           "bias_db": float(np.mean(err)),
           "within_3db_pct": float(100 * np.mean(np.abs(err) <= 3)),
           "within_6db_pct": float(100 * np.mean(np.abs(err) <= 6))}
    # Emitter localization: strongest predicted pixel (lowest path loss) outside buildings vs true tx.
    if tx.max() > 0:
        ty, tx_c = np.unravel_index(np.argmax(tx), tx.shape)
        search = np.where(env < 0.5, pred_db, np.inf)
        py, px = np.unravel_index(np.argmin(search), search.shape)
        d_px = float(np.hypot(py - ty, px - tx_c))
        out.update({"emitter_err_px": d_px, "emitter_err_m": d_px * args.m_per_px,
                    "emitter_hit": float(d_px * args.m_per_px <= args.hit_radius_m)})
    return out


def bootstrap_ci(v, n=2000, seed=0):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="benchmarks/GNN_no_tx.pth", help="GNN .pth ('' = random init, for smoke tests)")
    ap.add_argument("--test-dir", default="data/gnn_inference_test", help="held-out .parquet maps (never used in training)")
    ap.add_argument("--out-dir", default="benchmarks/gnn_bench")
    ap.add_argument("--densities", type=float, nargs="+", default=[0.5, 1, 2, 5, 10], help="% of pixels sampled")
    ap.add_argument("--mask-kind", choices=["uniform", "walk"], default="uniform")
    ap.add_argument("--avoid-buildings", action="store_true", help="never place samples inside buildings")
    ap.add_argument("--use-tx", action="store_true", help="feed tx_origin (default: zeroed, like the real hunt)")
    ap.add_argument("--no-exclude-buildings", dest="exclude_buildings", action="store_false",
                    help="also score building pixels")
    ap.add_argument("--m-per-px", type=float, default=1.0, help="map resolution, meters per pixel")
    ap.add_argument("--hit-radius-m", type=float, default=5.0, help="emitter counts as found within this distance")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--fp16", action="store_true", help="autocast fp16 on CUDA")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--timed-iters", type=int, default=500, help="timed forward passes for the latency stats")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    mask_dir = os.path.join(args.out_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.test_dir, "*.parquet")))[:args.max_files]
    if not files:
        raise SystemExit(f"No .parquet files in {args.test_dir}")
    model = load_model(args.weights, device)
    tag = f"{'fp16' if args.fp16 else 'fp32'}_{'tx' if args.use_tx else 'notx'}_{args.mask_kind}"
    print(f"{len(files)} maps x {len(args.densities)} densities | {device} | {tag}")

    # ---- 1. accuracy: every (map, density) once, same masks every run
    rows = []
    for i, f in enumerate(files):
        env, tx, gt = load_map(f)
        name = os.path.basename(f)
        for d in args.densities:
            mask = get_mask(mask_dir, name, d, env, args)
            x = build_input(env, tx, gt, mask, args.use_tx).to(device)
            pred_db = forward(model, x, args.fp16).squeeze().cpu().numpy() * (MAX_DB - MIN_DB) + MIN_DB
            rows.append({"file": name, "density_pct": d, "n_samples": int(mask.sum()),
                         **score(pred_db, gt, env, tx, mask, args)})
        print(f"[{i + 1}/{len(files)}] {name}  rmse@{args.densities[-1]:g}% = {rows[-1]['rmse_db']:.2f} dB")
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(args.out_dir, f"per_sample_{tag}.csv"), index=False)

    # ---- 2. latency / memory / power: repeated runs on real maps (kNN cost depends on the input)
    xs = []
    for f in files[:min(20, len(files))]:
        env, tx, gt = load_map(f)
        mask = get_mask(mask_dir, os.path.basename(f), args.densities[len(args.densities) // 2], env, args)
        xs.append(build_input(env, tx, gt, mask, args.use_tx).to(device))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for j in range(args.warmup):
        forward(model, xs[j % len(xs)], args.fp16)
    sync(device)
    teg = start_tegrastats(args.out_dir)
    times = []
    for j in range(args.timed_iters):
        x = xs[j % len(xs)]
        sync(device); t0 = time.perf_counter()
        forward(model, x, args.fp16)
        sync(device); times.append((time.perf_counter() - t0) * 1000)
    if teg:
        teg[0].terminate(); teg[0].wait(); teg[1].close()
    t = np.array(times)
    lat = {"latency_ms_mean": float(t.mean()), "latency_ms_p50": float(np.percentile(t, 50)),
           "latency_ms_p95": float(np.percentile(t, 95)), "latency_ms_p99": float(np.percentile(t, 99)),
           "fps": float(1000 / t.mean()),
           "process_peak_rss_MB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    if device.type == "cuda":
        lat["torch_gpu_peak_MB"] = torch.cuda.max_memory_allocated() / 2**20
    if teg:
        lat.update(parse_tegrastats(os.path.join(args.out_dir, "tegrastats.log")))
        if lat.get("power_mean_W"):
            lat["energy_mJ_per_map"] = lat["power_mean_W"] * lat["latency_ms_mean"]

    # ---- 3. summary per density with 95% bootstrap CI over maps
    metrics = [m for m in ["rmse_db", "mae_db", "bias_db", "within_3db_pct", "within_6db_pct",
                           "emitter_err_m", "emitter_hit"] if m in per]
    summ = []
    for d, g in per.groupby("density_pct"):
        r = {"density_pct": d, "n_maps": len(g)}
        for m in metrics:
            lo, hi = bootstrap_ci(g[m], seed=args.seed)
            r[m] = g[m].mean(); r[f"{m}_ci_lo"] = lo; r[f"{m}_ci_hi"] = hi
        summ.append(r)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(args.out_dir, f"summary_{tag}.csv"), index=False)

    info = {**device_info(device), **vars(args), "tag": tag, "n_maps": len(files),
            "params": sum(p.numel() for p in model.parameters()), **lat}
    json.dump(info, open(os.path.join(args.out_dir, f"run_info_{tag}.json"), "w"), indent=2, default=str)

    pd.set_option("display.width", 160)
    cols = ["density_pct", "rmse_db", "rmse_db_ci_lo", "rmse_db_ci_hi", "within_6db_pct"] + \
           [c for c in ["emitter_err_m", "emitter_hit"] if c in summ]
    print("\nAccuracy (unobserved" + (", non-building" if args.exclude_buildings else "") + " pixels):")
    print(summ[cols].round(2).to_string(index=False))
    print("\nOn-device:", json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in lat.items()}))
    print(f"\nSaved to {args.out_dir}/ (tag {tag})")


if __name__ == "__main__":
    main()
