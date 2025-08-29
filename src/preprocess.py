import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)


def to_onehot(y: np.ndarray, num_classes: int) -> torch.Tensor:
    y_t = torch.tensor(y, dtype=torch.long)
    return F.one_hot(y_t, num_classes=num_classes).float()


@torch.no_grad()
def morton_path_hw(H: int = 32, W: int = 32) -> torch.Tensor:
    def part1by1(n):
        n &= 0x0000FFFF
        n = (n | (n << 8)) & 0x00FF00FF
        n = (n | (n << 4)) & 0x0F0F0F0F
        n = (n | (n << 2)) & 0x33333333
        n = (n | (n << 1)) & 0x55555555
        return n
    coords = []
    for y in range(H):
        for x in range(W):
            z = (part1by1(x) | (part1by1(y) << 1))
            coords.append((z, y, x))
    coords.sort(key=lambda t: t[0])
    return torch.tensor([(y, x) for _, y, x in coords], dtype=torch.long)


class STRIPEEncoder(nn.Module):
    """
    STRIPE: Spatial-Temporal Random Image Patch Encoding
    - Multi-scale K, 2K, 4K patches sampled along a Morton path with configurable densities per scale.
    - Each patch is projected via a fixed ±1 hashing matrix to D dims.
    - Sequences across scales are interleaved in time.
    """
    def __init__(self, img_hw=32, K=3, scales=(1, 2, 4), densities=(1.0, 0.25, 0.0625), D=64, seed=0,
                 extra_stride: int = 1):
        super().__init__()
        self.img_hw = img_hw
        self.K = K
        self.scales = tuple(scales)
        self.densities = tuple(densities)
        assert len(self.scales) == len(self.densities)
        self.D = D
        self.extra_stride = max(1, int(extra_stride))
        set_seed(seed)
        # Morton path buffer
        self.register_buffer('path', morton_path_hw(img_hw, img_hw))  # (H*W, 2)
        # Precompute per-scale hashing matrices and center indices
        Hs = {}
        idxs = {}
        for s, dens in zip(self.scales, self.densities):
            Ks = K * s
            # Hash matrix D x Ks^2 with entries in {-1, +1} / sqrt(Ks^2)
            H_mat = torch.randint(0, 2, (D, Ks * Ks), dtype=torch.int8)
            H_mat = H_mat.float().mul_(2).sub_(1).div_(math.sqrt(Ks * Ks))
            Hs[str(s)] = nn.Parameter(H_mat, requires_grad=False)
            stride = max(1, int(round(1.0 / dens))) * self.extra_stride
            centers = self.path[::stride]
            idx = centers[:, 0] * img_hw + centers[:, 1]  # linear indices for unfold columns
            idxs[str(s)] = nn.Parameter(idx, requires_grad=False)
        self.H = nn.ParameterDict({str(s): Hs[str(s)] for s in self.scales})
        self.indices = nn.ParameterDict({str(s): idxs[str(s)] for s in self.scales})

    def forward(self, img_bchw: torch.Tensor) -> List[torch.Tensor]:
        # img_bchw: (B,1,H,W) with H=W=img_hw
        B, C, H, W = img_bchw.shape
        assert C == 1 and H == self.img_hw and W == self.img_hw
        outputs = []
        for b in range(B):
            per_scale = []
            for s in self.scales:
                Ks = self.K * s
                pad = Ks // 2
                patches = F.unfold(img_bchw[b:b + 1], kernel_size=Ks, padding=pad, stride=1)  # (1, Ks^2, H*W)
                patches = patches.squeeze(0)  # (Ks^2, H*W)
                idx = self.indices[str(s)]  # (#centers,)
                sel = patches.index_select(dim=1, index=idx).t()  # (#centers, Ks^2)
                feats = sel @ self.H[str(s)].t()  # (#centers, D)
                per_scale.append(feats)
            # Interleave scales
            lengths = [x.shape[0] for x in per_scale]
            maxL = max(lengths)
            padded = [F.pad(x, (0, 0, 0, maxL - x.shape[0])) for x in per_scale]
            stack = torch.stack(padded, dim=1)  # (maxL, S, D)
            mask = torch.zeros(maxL, len(per_scale), dtype=torch.bool)
            for i, L in enumerate(lengths):
                mask[:L, i] = True
            seq = stack.reshape(-1, self.D)[mask.reshape(-1)]  # (T, D)
            outputs.append(seq)
        return outputs


class RasterEncoder(nn.Module):
    """
    Raster-scan baseline: at each step, emit D-dim feature: u_t = pixel_t * h, h fixed random vector.
    """
    def __init__(self, img_hw=32, D=64, seed=0, extra_stride: int = 1):
        super().__init__()
        self.img_hw = img_hw
        self.D = D
        self.extra_stride = max(1, int(extra_stride))
        set_seed(seed)
        h = torch.randn(D) / math.sqrt(D)
        self.register_buffer('h', h)
        idxs = torch.arange(img_hw * img_hw)[::self.extra_stride]
        self.register_buffer('idxs', idxs)

    def forward(self, img_bchw: torch.Tensor) -> List[torch.Tensor]:
        B, C, H, W = img_bchw.shape
        assert C == 1 and H == self.img_hw and W == self.img_hw
        outs = []
        for b in range(B):
            flat = img_bchw[b].reshape(-1)
            flat = flat.index_select(0, self.idxs)
            u = flat.unsqueeze(1) * self.h.unsqueeze(0)  # (T, D)
            outs.append(u)
        return outs


class SITHLikeEncoder(nn.Module):
    """
    SITH-like static image sequence encoder with exponentially spaced IIR filters.
    """
    def __init__(self, img_hw=32, D=64, tau_min=1.5, tau_max=256, seed=0, extra_stride: int = 1):
        super().__init__()
        self.img_hw = img_hw
        self.D = D
        self.extra_stride = max(1, int(extra_stride))
        set_seed(seed)
        self.taus = torch.logspace(math.log10(tau_min), math.log10(tau_max), D)
        self.register_buffer('betas', torch.exp(-1.0 / self.taus))
        idxs = torch.arange(img_hw * img_hw)[::self.extra_stride]
        self.register_buffer('idxs', idxs)

    def forward(self, img_bchw: torch.Tensor) -> List[torch.Tensor]:
        B, C, H, W = img_bchw.shape
        assert C == 1 and H == self.img_hw and W == self.img_hw
        outs = []
        for b in range(B):
            flat = img_bchw[b].reshape(-1)
            flat = flat.index_select(0, self.idxs)  # (T,)
            T = flat.shape[0]
            s = torch.zeros(self.D, device=flat.device)
            U = torch.zeros(T, self.D, device=flat.device)
            for t in range(T):
                x_t = flat[t]
                s = self.betas * s + (1 - self.betas) * x_t
                U[t] = s
            outs.append(U)
        return outs


# Data loading

def get_datasets(name: str, root: str = './data'):
    name_l = name.lower()
    if name_l == 'mnist':
        tf = transforms.Compose([transforms.ToTensor(), transforms.Pad(2)])  # -> 32x32
        trainset = datasets.MNIST(root=root, train=True, download=True, transform=tf)
        testset = datasets.MNIST(root=root, train=False, download=True, transform=tf)
        C = 10
    elif name_l in ['fashion-mnist', 'fashion_mnist', 'fmnist']:
        tf = transforms.Compose([transforms.ToTensor(), transforms.Pad(2)])  # -> 32x32
        trainset = datasets.FashionMNIST(root=root, train=True, download=True, transform=tf)
        testset = datasets.FashionMNIST(root=root, train=False, download=True, transform=tf)
        C = 10
    elif name_l in ['cifar10', 'cifar10-gray', 'cifar-10-gray']:
        tf = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])  # 32x32
        trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=tf)
        testset = datasets.CIFAR10(root=root, train=False, download=True, transform=tf)
        C = 10
    else:
        raise ValueError('Unknown dataset name')
    return trainset, testset, C


def prepare_tensors(dataset, n_samples: int) -> Tuple[List[torch.Tensor], List[int]]:
    imgs = []
    labels = []
    for i in range(min(n_samples, len(dataset))):
        img, y = dataset[i]
        img = img.unsqueeze(0)  # (1,1,32,32)
        imgs.append(img)
        labels.append(int(y))
    return imgs, labels


# Robustness transforms

def add_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    return (x + sigma * torch.randn_like(x)).clamp(0, 1)


def translate_zeropad(x: torch.Tensor, max_px: int = 5) -> torch.Tensor:
    dy = np.random.randint(-max_px, max_px + 1)
    dx = np.random.randint(-max_px, max_px + 1)
    y = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
    if dy > 0:
        y[:, :, :dy, :] = 0
    elif dy < 0:
        y[:, :, dy:, :] = 0
    if dx > 0:
        y[:, :, :, :dx] = 0
    elif dx < 0:
        y[:, :, :, dx:] = 0
    return y
