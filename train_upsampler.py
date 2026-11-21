"""
train_upsampler.py - train the upsampling decoder on GNN_no_tx + Sionna parquet, then score
baseline vs upsampled using Brittany's benchmark_reconstruction.py code (same model class,
same saved masks, same score()), so results drop straight into her table.

Colab, from the repo root (UCF-Spectrum-Cartography-Map/Neural-Network-Architecture):
  python train_upsampler.py --train-dir /content/dataset_256/~train --val-dir /content/dataset_256/val \
      --out "/content/drive/MyDrive/Senior Design/upsampler_run1"

Eval only:  python train_upsampler.py --eval-only --ckpt <out>/best.pth --out <out>

Path-shaped samples (SCRUM-104): --train-mask-kind walk|mix trains on the team's walk masks,
--init-ckpt <best.pth> starts from an earlier decoder, --mask-kind walk scores on walk masks.
Map scale comes from map_scale.py (per dataset); an unknown test folder needs --dataset or --m-per-px.
"""
import argparse, copy, glob, json, math, os, shutil, sys, time
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, "benchmarks")
import benchmark_reconstruction as br          # Brittany's: model, load_map, get_mask, score
import map_scale
from vig_upsampler import UpsampledViG, recon_loss, make_optimizer, blockiness

H = W = 256
DB_RANGE = br.MAX_DB - br.MIN_DB


def find_parquets(d, limit=None):
    files = sorted(glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True))
    return files[:limit] if limit else files


class Cache:
    """Maps in RAM: building uint8, tx index, normalized path loss float16 (~200 KB/map)."""
    def __init__(self, files, tag):
        self.files, self.env, self.tx, self.pl = files, [], [], []
        t0 = time.time()
        for i, f in enumerate(files):
            env, tx, gt = br.load_map(f)
            self.env.append((env > 0.5).astype(np.uint8))
            self.tx.append(int(np.argmax(tx)))
            self.pl.append(np.clip((gt - br.MIN_DB) / DB_RANGE, 0, 1).astype(np.float16))
            if (i + 1) % 1000 == 0:
                print(f"  cached {tag} {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)

    def __len__(self):
        return len(self.files)


def sample_pixels(env, density, rng, kind, avoid_buildings):
    """Flat indices of the sampled pixels for one training/validation input.
    uniform = scattered pixels (the original behaviour, same random stream as before);
    walk    = the benchmark's own path generator (br.make_mask), always outside buildings
              because a rover cannot drive through them;
    mix     = walk or uniform, 50/50 per map, so the model keeps working on scattered samples."""
    if kind == "mix":
        kind = "walk" if rng.random() < 0.5 else "uniform"
    if kind == "walk":
        return np.flatnonzero(br.make_mask(density, env, rng, "walk", True))
    pool = np.flatnonzero(env.ravel() == 0) if avoid_buildings else np.arange(H * W)
    n = max(1, int(round(density / 100 * H * W)))
    return rng.choice(pool, size=min(n, pool.size), replace=False)


def train_input(env, pl, density, rng, avoid_buildings, kind="uniform"):
    idx = sample_pixels(env, density, rng, kind, avoid_buildings)
    sparse = np.zeros(H * W, np.float32)
    sparse[idx] = np.maximum(pl.ravel()[idx].astype(np.float32), 1e-4)  # >0 so mask is exact
    return np.stack([env.astype(np.float32), np.zeros((H, W), np.float32), sparse.reshape(H, W)])


def batches(cache, bs, rng, avoid_buildings, kind="uniform"):
    order = rng.permutation(len(cache))
    for s in range(0, len(order), bs):
        xs, ys = [], []
        for i in order[s:s + bs]:
            d = math.exp(rng.uniform(math.log(0.5), math.log(10.0)))  # log-uniform 0.5-10%
            xs.append(train_input(cache.env[i], cache.pl[i], d, rng, avoid_buildings, kind))
            ys.append(cache.pl[i][None].astype(np.float32))
        yield torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


@torch.no_grad()
def val_rmse(model, cache, device, avoid_buildings, seed=7, kind="uniform"):
    """Quick val: mean per-map RMSE (dB) on unobserved non-building pixels, all densities.
    kind="mix" scores every map with both a uniform and a walk mask, so the checkpoint that is
    kept is the best on the two together."""
    if hasattr(model, "eval"):
        model.eval()
    errs = []
    for k in (("uniform", "walk") if kind == "mix" else (kind,)):
        for i in range(len(cache)):
            for d in (0.5, 2.0, 10.0):
                rng = np.random.default_rng([seed, i, int(d * 10)] + ([1] if k == "walk" else []))
                x = train_input(cache.env[i], cache.pl[i], d, rng, avoid_buildings, k)
                p = model(torch.from_numpy(x)[None].to(device)).float()[0, 0].cpu().numpy()
                region = (x[0] < 0.5) & (x[2] == 0)
                e = (p - cache.pl[i].astype(np.float32))[region] * DB_RANGE
                errs.append(math.sqrt(float(np.mean(e ** 2))))
    return float(np.mean(errs))


def load_state(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # older PyTorch builds
        return torch.load(path, map_location=device)


@torch.no_grad()
def benchmark(models, args, device):
    """Brittany's protocol: same test maps, same saved masks, same score()."""
    bargs = SimpleNamespace(mask_kind=args.mask_kind, seed=0, avoid_buildings=args.avoid_buildings,
                            exclude_buildings=True, m_per_px=args.m_per_px, hit_radius_m=args.hit_radius_m)
    mask_dir = os.path.join(args.out, "masks"); os.makedirs(mask_dir, exist_ok=True)
    files = find_parquets(args.test_dir)
    rows = []
    for f in files:
        env, tx, gt = br.load_map(f)
        name = os.path.basename(f)
        for d in args.densities:
            mask = br.get_mask(mask_dir, name, d, env, bargs)
            x = br.build_input(env, tx, gt, mask, use_tx=False).to(device)
            for mname, m in models.items():
                with torch.autocast("cuda", dtype=torch.float16, enabled=args.fp16 and device.type == "cuda"):
                    pred = m(x).float()
                pred_db = pred.squeeze().cpu().numpy() * DB_RANGE + br.MIN_DB
                rows.append({"model": mname, "file": name, "density_pct": d,
                             **br.score(pred_db, gt, env, tx, mask, bargs),
                             "blockiness": blockiness(pred_db, env < 0.5)})
    import pandas as pd
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(args.out, f"per_sample_compare_{args.mask_kind}.csv"), index=False)
    summ = []
    for (mname, d), g in per.groupby(["model", "density_pct"], sort=False):
        lo, hi = br.bootstrap_ci(g["rmse_db"])
        summ.append({"model": mname, "density_pct": d, "rmse_db": g["rmse_db"].mean(),
                     "rmse_ci_lo": lo, "rmse_ci_hi": hi, "within_6db_pct": g["within_6db_pct"].mean(),
                     "emitter_err_m": g["emitter_err_m"].mean(), "emitter_hit": g["emitter_hit"].mean(),
                     "blockiness": g["blockiness"].mean()})
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(args.out, f"summary_compare_{args.mask_kind}.csv"), index=False)
    return summ, files


@torch.no_grad()
def latency(m, device, fp16, n=100):
    x = torch.zeros(1, 3, H, W, device=device)
    x[:, 2] = (torch.rand(H, W, device=device) < 0.02) * torch.rand(H, W, device=device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
        for _ in range(20): m(x)
        if device.type == "cuda": torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n): m(x)
        if device.type == "cuda": torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t) / n


@torch.no_grad()
def save_panel(models, files, args, device, path, dens=2.0, n=3):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = min(n, len(files)); cols = 2 + len(models)
    fig, ax = plt.subplots(n, cols, figsize=(3.3 * cols, 3.3 * n), squeeze=False)
    bargs = SimpleNamespace(mask_kind=args.mask_kind, seed=0, avoid_buildings=args.avoid_buildings)
    for r in range(n):
        f = files[r * (len(files) // n)]
        env, tx, gt = br.load_map(f)
        mask = br.get_mask(os.path.join(args.out, "masks"), os.path.basename(f), dens, env, bargs)
        x = br.build_input(env, tx, gt, mask, use_tx=False).to(device)
        panels = [("ground truth", gt), (f"samples {dens:g}%", np.ma.masked_where(~mask, gt))]
        for mname, m in models.items():
            panels.append((mname, m(x).float().squeeze().cpu().numpy() * DB_RANGE + br.MIN_DB))
        ty, tx_c = np.unravel_index(np.argmax(tx), tx.shape)
        for c, (t, im) in enumerate(panels):
            ax[r, c].imshow(im, cmap="viridis_r", vmin=np.percentile(gt, 1), vmax=np.percentile(gt, 99))
            ax[r, c].plot(tx_c, ty, "r+", ms=12, mew=2); ax[r, c].set_title(t); ax[r, c].axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="benchmarks/GNN_no_tx.pth")
    ap.add_argument("--train-dir"); ap.add_argument("--val-dir")
    ap.add_argument("--test-dir", default="data/gnn_inference_test")
    ap.add_argument("--out", default="upsampler_out")
    ap.add_argument("--max-train", type=int, default=6000, help="maps cached in RAM (~200 KB each)")
    ap.add_argument("--max-val", type=int, default=150)
    ap.add_argument("--epochs1", type=int, default=8); ap.add_argument("--epochs2", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr-dec", type=float, default=1e-3); ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--dc", default="soft", choices=["soft", "hard", "none"])
    ap.add_argument("--no-gnn-head", action="store_true")
    ap.add_argument("--w-edge", type=float, default=0.1); ap.add_argument("--w-meas", type=float, default=0.2)
    ap.add_argument("--w-peak", type=float, default=1.0)
    ap.add_argument("--densities", type=float, nargs="+", default=[0.5, 1, 2, 5, 10])
    ap.add_argument("--mask-kind", choices=["uniform", "walk"], default="uniform", help="masks used for the final scoring")
    ap.add_argument("--train-mask-kind", choices=["uniform", "walk", "mix"], default="uniform",
                    help="masks used for training and validation (walk masks never enter buildings)")
    ap.add_argument("--avoid-buildings", action="store_true",
                    help="scoring masks and uniform training samples stay outside buildings")
    ap.add_argument("--dataset", choices=sorted(map_scale.DATASETS), help="scale of the test maps (default: from --test-dir)")
    ap.add_argument("--m-per-px", type=float, default=None, help="override the dataset scale")
    ap.add_argument("--hit-px", type=float, default=map_scale.HIT_PX, help="emitter hit radius in pixels")
    ap.add_argument("--hit-radius-m", type=float, default=None, help="override the hit radius in meters")
    ap.add_argument("--init-ckpt", help="start training from this UpsampledViG checkpoint (strict load)")
    ap.add_argument("--fp16", action="store_true", help="autocast fp16 for eval/latency (Jetson-like)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true"); ap.add_argument("--ckpt")
    args = ap.parse_args()

    args.m_per_px = map_scale.resolve_scale(dataset=args.dataset, path=args.test_dir, m_per_px=args.m_per_px)
    if args.hit_radius_m is None:
        args.hit_radius_m = map_scale.hit_radius_m(args.m_per_px, args.hit_px)
    print(f"scale: {args.m_per_px:g} m/px, emitter hit radius {args.hit_radius_m:g} m "
          f"({args.hit_radius_m / args.m_per_px:g} px)")
    os.makedirs(args.out, exist_ok=True)
    if not args.eval_only:  # keep the exact code and settings next to the checkpoint
        code_dir = os.path.join(args.out, "code"); os.makedirs(code_dir, exist_ok=True)
        here = os.path.dirname(os.path.abspath(__file__))
        for f in ("train_upsampler.py", "vig_upsampler.py", "map_scale.py"):
            if os.path.exists(os.path.join(here, f)):
                shutil.copy(os.path.join(here, f), code_dir)
        json.dump(vars(args), open(os.path.join(args.out, "run_config.json"), "w"), indent=1)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn = br.load_model(args.weights, device)                  # strict load, her class
    model = UpsampledViG(copy.deepcopy(gnn), feat_ch=gnn.patch_embed.proj.out_channels, vig_in_ch=3,
                         dc_mode=args.dc, init_from_gnn_head=not args.no_gnn_head).to(device)

    if args.eval_only:
        model.load_state_dict(load_state(args.ckpt, device))
    else:
        if args.init_ckpt:
            model.load_state_dict(load_state(args.init_ckpt, device))
            print(f"starting from {args.init_ckpt}")
        test_names = {os.path.basename(f) for f in find_parquets(args.test_dir)}
        rng = np.random.default_rng(args.seed)
        all_tr = find_parquets(args.train_dir)
        tr = [f for f in all_tr if os.path.basename(f) not in test_names]
        rng.shuffle(tr)
        all_va = find_parquets(args.val_dir) if args.val_dir else []
        va = [f for f in all_va if os.path.basename(f) not in test_names] if args.val_dir else tr[args.max_train:]
        removed = (len(all_tr) - len(tr)) + ((len(all_va) - len(va)) if args.val_dir else 0)
        print(f"found train {len(tr)}  val {len(va)}  (files removed because they share a name with a test map: {removed})")
        if not tr or not va:
            raise SystemExit("no training or validation maps found: check --train-dir / --val-dir")
        train = Cache(tr[:args.max_train], "train")
        val = Cache(va[:args.max_val], "val")
        base = val_rmse(gnn, val, device, args.avoid_buildings, kind=args.train_mask_kind)
        print(f"baseline GNN val rmse ({args.train_mask_kind} masks): {base:.2f} dB")
        steps = math.ceil(len(train) / args.bs)
        best, hist = float("inf"), []
        if args.init_ckpt:  # the run can only replace the starting checkpoint with something better on val
            best = val_rmse(model, val, device, args.avoid_buildings, kind=args.train_mask_kind)
            torch.save(model.state_dict(), os.path.join(args.out, "best.pth"))
            print(f"starting checkpoint val rmse ({args.train_mask_kind} masks): {best:.2f} dB")
        for stage, ep in ((1, args.epochs1), (2, args.epochs2)):
            if ep == 0: continue
            opt, sched = make_optimizer(model, stage, args.lr_dec, args.lr_backbone, ep, steps)
            for e in range(ep):
                model.train(); model.freeze_backbone(stage == 1)
                t0, tot, n = time.time(), 0.0, 0
                for x, y in batches(train, args.bs, rng, args.avoid_buildings, args.train_mask_kind):
                    x, y = x.to(device), y.to(device)
                    out, raw = model(x, return_raw=True)
                    loss = recon_loss(out, raw, y, x, args.w_edge, args.w_meas, args.w_peak)
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); sched.step()
                    tot += loss.item() * x.size(0); n += x.size(0)
                vr = val_rmse(model, val, device, args.avoid_buildings, kind=args.train_mask_kind)
                hist.append(dict(stage=stage, epoch=e + 1, loss=tot / n, val_rmse_db=vr,
                                 baseline_val_rmse_db=base, dc_w=torch.sigmoid(model.dc_logit).item()))
                flag = ""
                if vr < best:
                    best = vr; torch.save(model.state_dict(), os.path.join(args.out, "best.pth")); flag = " *best"
                print(f"S{stage} ep{e+1}/{ep}  loss {tot/n:.5f}  val {vr:.2f} dB (GNN {base:.2f})  "
                      f"{time.time()-t0:.0f}s{flag}", flush=True)
                json.dump(hist, open(os.path.join(args.out, "history.json"), "w"), indent=1)
        model.load_state_dict(load_state(os.path.join(args.out, "best.pth"), device))

    model.eval()
    models = {"GNN": gnn, "GNN+upsampler": model}
    summ, files = benchmark(models, args, device)
    piv = summ.pivot(index="density_pct", columns="model")
    import pandas as pd
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.precision", 2):
        print(f"\nAccuracy on {args.mask_kind} masks (unobserved, non-building pixels) - Brittany's score():")
        print(piv[["rmse_db", "within_6db_pct", "emitter_err_m", "emitter_hit", "blockiness"]])
    info = {}
    for mname, m in models.items():
        info[mname] = {"latency_ms": latency(m, device, args.fp16),
                       "params": sum(p.numel() for p in m.parameters())}
        print(f"{mname}: {info[mname]['latency_ms']:.1f} ms/map on {device}{' fp16' if args.fp16 else ''}, "
              f"{info[mname]['params']/1e6:.2f}M params")
    json.dump(info, open(os.path.join(args.out, "cost.json"), "w"), indent=1)
    save_panel(models, files, args, device, os.path.join(args.out, f"comparison_{args.mask_kind}.png"))
    print(f"\nsaved -> {args.out}/  summary_compare_{args.mask_kind}.csv  per_sample_compare_{args.mask_kind}.csv  "
          f"comparison_{args.mask_kind}.png")


if __name__ == "__main__":
    main()
