"""Turn a replay log (written by replay_node) into a shareable PNG figure.

Left: the test map's true path loss with buildings, the rover's sample path, every raw GNN
emitter estimate, the Kalman track, and the true emitter with the hit radius around it.
Right: emitter error over time for raw GNN and Kalman, with the hit radius as a dashed line.

Usage:  ros2 run gnn_reconstruction plot_replay ~/replay_logs/replay_<date>_<time>.csv [-o out.png]
   or:  python3 plot_replay.py <csv>     (needs matplotlib, pandas, pyarrow; no ROS needed)
With no csv argument it plots the newest log in ~/replay_logs/.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

H = W = 256
RAW, KALMAN = "#eb6834", "#2a78d6"   # orange = raw GNN, blue = Kalman (same in every figure)
INK, MUTED = "#0b0b0b", "#52514e"


def summary(e, hit):
    e = np.asarray(e, float)
    if not len(e):
        return "no estimates"
    return f"RMSE {np.sqrt(np.mean(e ** 2)):.1f} m, {100 * np.mean(e <= hit + 0.01):.0f}% within {hit:.1f} m"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", help="replay log .csv (default: newest in ~/replay_logs)")
    ap.add_argument("-o", "--out", help="output PNG (default: next to the csv)")
    args = ap.parse_args()

    path = args.csv
    if not path:
        logs = sorted(glob.glob(os.path.expanduser("~/replay_logs/replay_*.csv")))
        if not logs:
            raise SystemExit("No logs in ~/replay_logs - run the replay launch first.")
        path = logs[-1]
    meta = json.load(open(os.path.splitext(path)[0] + ".json"))
    log = pd.read_csv(path)
    out = args.out or os.path.splitext(path)[0] + ".png"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res, ox, oy, hit = meta["m_per_px"], meta["origin_x"], meta["origin_y"], meta["hit_radius_m"]
    tx, ty = meta["tx_xy"]
    row = pd.read_parquet(meta["map_file"], engine="pyarrow").iloc[0]
    env = np.asarray(row["building_mask"], np.float32).reshape(H, W)
    gt = np.asarray(row["path_loss"], np.float32).reshape(H, W)

    samples = log[log.kind == "sample"]
    raw, kf = log[log.kind == "raw"], log[log.kind == "kalman"]
    t_end = log.t_s.max()

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1, 1.15]})

    # ---- left: map + tracks (y up, like RViz)
    extent = [ox, ox + W * res, oy, oy + H * res]
    shown = np.ma.masked_where(env > 0.5, gt)
    im = ax.imshow(shown, origin="lower", extent=extent, cmap="Greys_r", interpolation="nearest")
    ax.imshow(np.ma.masked_where(env < 0.5, env), origin="lower", extent=extent, cmap="Greys",
              vmin=0, vmax=1.4, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("true path loss (dB), buildings dark gray", color=MUTED)
    ax.scatter(samples.x_m, samples.y_m, s=2, color="#8fd3f0", linewidth=0, alpha=0.7,
               label="RF samples (rover path)")
    ax.scatter(raw.x_m, raw.y_m, s=22, color=RAW, edgecolor="white", linewidth=0.6, zorder=3,
               label="raw GNN estimates")
    ax.plot(kf.x_m, kf.y_m, color=KALMAN, lw=2, zorder=4, label="Kalman track")
    if len(kf):
        ax.scatter(kf.x_m.iloc[-1:], kf.y_m.iloc[-1:], s=60, color=KALMAN, edgecolor="white",
                   linewidth=1.5, zorder=5, label="Kalman final")
    ax.add_patch(plt.Circle((tx, ty), hit, fill=False, color=INK, lw=1.2, ls="--", zorder=5))
    ax.scatter([tx], [ty], marker="*", s=220, color=INK, edgecolor="white", linewidth=1, zorder=6,
               label=f"true emitter (dashed = {hit:.1f} m)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(os.path.basename(meta["map_file"]), color=INK, loc="left")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # ---- right: error vs time
    ax2.plot(raw.t_s, raw.err_m, color=RAW, lw=0, marker="o", ms=4, alpha=0.8, label="raw GNN")
    ax2.plot(kf.t_s, kf.err_m, color=KALMAN, lw=2, label="Kalman")
    ax2.axhline(hit, color=INK, lw=1, ls="--")
    ax2.text(0.01 * t_end, hit, f"hit radius {hit:.1f} m", va="bottom", ha="left", color=MUTED, fontsize=9,
             bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax2.set_xlim(0, t_end); ax2.set_ylim(bottom=0)
    ax2.set_xlabel("time since start (s)"); ax2.set_ylabel("distance to true emitter (m)")
    ax2.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax2.set_title("Emitter error over time", color=INK, loc="left")
    ax2.legend(loc="upper right", fontsize=9)
    cover = samples.coverage_pct.max() if len(samples) else 0
    ax2.text(0.0, -0.17, f"Raw GNN: {summary(raw.err_m, hit)}\nKalman:  {summary(kf.err_m, hit)}\n"
             f"{meta['mode']} sampling, {cover:.1f}% of pixels sampled, {res} m/px",
             transform=ax2.transAxes, va="top", color=INK, fontsize=9, family="monospace")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    print(f"raw GNN: {summary(raw.err_m, hit)}")
    print(f"Kalman:  {summary(kf.err_m, hit)}")


if __name__ == "__main__":
    main()
