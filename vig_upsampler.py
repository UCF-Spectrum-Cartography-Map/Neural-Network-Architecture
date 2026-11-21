"""
vig_upsampler.py - ablation-winning decoder on the REAL ViGPatchModel (NNA repo).

  ViG (patch_embed -> 4 ViGBlocks -> norm)  -> (B, 64, 32, 32) node features
  InputSkipEncoder on the raw 256x256 input -> 256/128/64 skip features
  PixelShuffle decoder 32->64->128->256 with skips (A3) + soft data consistency (A7)
  Loss: unobserved non-building MSE + edge (A8) + measured-pixel (A9) + emitter-peak term

Input convention (models_radiomap.py):
  ch0 building_mask (0/1), ch1 tx_origin (zeros for no-tx), ch2 sparse path loss norm [0,1]
  target = path loss norm [0,1]  (LOW value = strong signal = near emitter)
Measured pixels = sparse > 0.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BLD_CH, TX_CH, SPARSE_CH = 0, 1, 2


def conv_block(cin, cout):
    g = math.gcd(8, cout)
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(g, cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(g, cout), nn.GELU(),
    )


class InputSkipEncoder(nn.Module):
    """Light conv pyramid on the raw input (+ explicit mask) -> features at 256, 128, 64."""
    def __init__(self, in_ch, chs=(16, 32, 64)):
        super().__init__()
        self.e256 = conv_block(in_ch, chs[0])
        self.e128 = conv_block(chs[0], chs[1])
        self.e64 = conv_block(chs[1], chs[2])

    def forward(self, x):
        s256 = self.e256(x)
        s128 = self.e128(F.avg_pool2d(s256, 2))
        s64 = self.e64(F.avg_pool2d(s128, 2))
        return s256, s128, s64


class PSUp(nn.Module):
    """x2 PixelShuffle upsample, concat skip, fuse."""
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.expand = nn.Conv2d(cin, cout * 4, 3, padding=1)
        self.ps = nn.PixelShuffle(2)
        self.fuse = conv_block(cout + cskip, cout)

    def forward(self, x, skip):
        return self.fuse(torch.cat([self.ps(self.expand(x)), skip], 1))


class SkipPSDecoder(nn.Module):
    def __init__(self, feat_ch, in_ch, skip_chs=(16, 32, 64), dec_chs=(128, 64, 32)):
        super().__init__()
        self.skips = InputSkipEncoder(in_ch, skip_chs)
        self.proj = conv_block(feat_ch, dec_chs[0])
        self.up64 = PSUp(dec_chs[0], skip_chs[2], dec_chs[0])
        self.up128 = PSUp(dec_chs[0], skip_chs[1], dec_chs[1])
        self.up256 = PSUp(dec_chs[1], skip_chs[0], dec_chs[2])
        self.head = nn.Conv2d(dec_chs[2], 1, 1)

    def forward(self, feat32, x):
        s256, s128, s64 = self.skips(x)
        h = self.proj(feat32)
        h = self.up64(h, s64)
        h = self.up128(h, s128)
        h = self.up256(h, s256)
        return self.head(h)


def vig_features(vig, x):
    """Run ViGPatchModel up to its norm -> (B, E, 32, 32). Skips the 8x ConvTranspose head.
    Works with benchmark_reconstruction.ViGPatchModel (use that one: its PatchEmbed has LayerNorm)."""
    ps = vig.patch_size
    ph, pw = (ps - x.shape[2] % ps) % ps, (ps - x.shape[3] % ps) % ps
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))
    x = vig.patch_embed(x)
    B, E, nh, nw = x.shape
    x = x.reshape(B, E, nh * nw, 1)
    for blk in vig.blocks:
        x = blk(x)
    return vig.norm(x.reshape(B, E, nh, nw))


class UpsampledViG(nn.Module):
    def __init__(self, vig, feat_ch=64, vig_in_ch=3, dc_mode="soft", init_from_gnn_head=True):
        super().__init__()
        self.vig = vig
        self.vig_in_ch = vig_in_ch  # 3 = (bld, tx, sparse); 2 = (bld, sparse) no-tx checkpoint
        # decoder always sees the full 3-ch input + explicit measurement mask
        self.decoder = SkipPSDecoder(feat_ch, 3 + 1)
        self.dc_mode = dc_mode
        self.dc_logit = nn.Parameter(torch.tensor(2.0))  # soft DC weight ~0.88 at init
        # residual on the GNN's own (bilinear-ish) output so stage 1 starts at the baseline
        self.use_gnn_head = init_from_gnn_head
        if init_from_gnn_head:
            nn.init.zeros_(self.decoder.head.weight)
            nn.init.zeros_(self.decoder.head.bias)

    def freeze_backbone(self, frozen=True):
        for p in self.vig.parameters():
            p.requires_grad = not frozen
        # keep BN stats fixed when frozen
        for m in self.vig.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval() if frozen else m.train()

    def forward(self, x, return_raw=False):
        mask = (x[:, SPARSE_CH:SPARSE_CH + 1] > 0).float()
        feat = vig_features(self.vig, x if self.vig_in_ch == 3 else x[:, [BLD_CH, SPARSE_CH]])
        raw = self.decoder(feat, torch.cat([x, mask], 1))
        if self.use_gnn_head:
            # start exactly from the pretrained GNN prediction; decoder learns the correction
            base = self.vig.patch_unembed(feat)[:, :, :x.shape[2], :x.shape[3]]
            raw = raw + base
        out = raw
        if self.dc_mode != "none":
            meas = x[:, SPARSE_CH:SPARSE_CH + 1]
            w = mask if self.dc_mode == "hard" else torch.sigmoid(self.dc_logit) * mask
            out = w * meas + (1 - w) * raw
        return (out, raw) if return_raw else out


def grad_loss(pred, gt, region):
    dxp, dxg = pred[..., :, 1:] - pred[..., :, :-1], gt[..., :, 1:] - gt[..., :, :-1]
    dyp, dyg = pred[..., 1:, :] - pred[..., :-1, :], gt[..., 1:, :] - gt[..., :-1, :]
    rx, ry = region[..., :, 1:] * region[..., :, :-1], region[..., 1:, :] * region[..., :-1, :]
    return ((dxp - dxg).abs() * rx).sum() / rx.sum().clamp(min=1) + \
           ((dyp - dyg).abs() * ry).sum() / ry.sum().clamp(min=1)


def recon_loss(out, raw, gt, x, w_edge=0.1, w_meas=0.2, w_peak=1.0, peak_q=0.02):
    """All in normalized path-loss units. Emitter = LOWEST path loss -> bottom quantile."""
    mask = (x[:, SPARSE_CH:SPARSE_CH + 1] > 0).float()
    bld = x[:, BLD_CH:BLD_CH + 1]
    valid = (1 - mask) * (1 - bld)
    se = (out - gt) ** 2
    main = (se * valid).sum() / valid.sum().clamp(min=1)
    gt_v = gt.masked_fill(bld > 0, float("inf")).flatten(1)
    thr = torch.quantile(gt_v.clamp(max=10.0), peak_q, dim=1).view(-1, 1, 1, 1)
    peak = (gt <= thr).float() * valid
    peak_l = (se * peak).sum() / peak.sum().clamp(min=1)
    meas_l = (((raw - gt) ** 2) * mask).sum() / mask.sum().clamp(min=1)
    return main + w_edge * grad_loss(out, gt, valid) + w_meas * meas_l + w_peak * peak_l


def make_optimizer(model, stage, lr_dec=1e-3, lr_backbone=2e-5, epochs=10, steps_per_epoch=100,
                   warmup_frac=0.05):
    """Stage 1: decoder only. Stage 2: + backbone at low LR. Cosine with warmup."""
    model.freeze_backbone(stage == 1)
    groups = [{"params": list(model.decoder.parameters()) + [model.dc_logit], "lr": lr_dec}]
    if stage == 2:
        groups.append({"params": model.vig.parameters(), "lr": lr_backbone})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    total = max(1, epochs * steps_per_epoch)
    warm = max(1, int(total * warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    return opt, sched


def blockiness(img, valid=None, patch=8):
    """8-px grid artifact score for a (H, W) map: mean |step| across patch boundaries divided by
    mean |step| inside patches. 1.0 = no grid visible (ground truth scores 1.00); the raw GNN
    scores 11-19 because every 8x8 block comes from one graph node. Steps that touch a pixel
    where valid is False (e.g. buildings) are ignored."""
    img = np.asarray(img, dtype=np.float64)
    valid = np.ones(img.shape, bool) if valid is None else np.asarray(valid, bool)
    on, off = [], []
    for ax in (0, 1):
        d = np.abs(np.diff(img, axis=ax))
        v = (valid[1:, :] & valid[:-1, :]) if ax == 0 else (valid[:, 1:] & valid[:, :-1])
        edge = (np.arange(d.shape[ax]) + 1) % patch == 0
        edge = np.broadcast_to(edge[:, None] if ax == 0 else edge[None, :], d.shape)
        on.append(d[edge & v]); off.append(d[~edge & v])
    on, off = np.concatenate(on), np.concatenate(off)
    if on.size == 0 or off.size == 0:
        return float("nan")
    return float(on.mean() / max(off.mean(), 1e-12))
