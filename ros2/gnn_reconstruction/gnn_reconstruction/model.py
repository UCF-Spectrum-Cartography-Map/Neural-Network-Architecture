"""ViG GNN used for radio-map reconstruction (same architecture as benchmarks/benchmark_reconstruction.py)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def load_torch_model(path, device):
    model = ViGPatchModel()
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older JetPack PyTorch builds
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
