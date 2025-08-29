import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import STRIPEEncoder


class FastfoodOp(nn.Module):
    """
    Fastfood/Hadamard structured linear operator for O(N log N) multiplication.
    Requires N to be power-of-two.
    """
    def __init__(self, N, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.N = N
        self.B = nn.Parameter(torch.randint(0, 2, (N,)).float().mul(2).sub(1), requires_grad=False)
        self.G = nn.Parameter(torch.randn(N), requires_grad=False)
        self.S = nn.Parameter(torch.rand(N).add(0.5), requires_grad=False)
        self.register_buffer('P', torch.randperm(N))

    def fwht(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N), N must be power-of-two
        B, N = x.shape
        h = 1
        y = x
        while h < N:
            y_reshaped = y.view(B, -1, 2 * h)
            a = y_reshaped[:, :, :h]
            b = y_reshaped[:, :, h:2 * h]
            y = torch.cat([a + b, a - b], dim=2).view(B, -1)
            h *= 2
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * self.B
        y = self.fwht(y)
        y = y.index_select(1, self.P)
        y = y * self.G
        y = self.fwht(y)
        y = y * self.S
        return y / math.sqrt(self.N)


class SRCReservoir(nn.Module):
    """
    Structured Echo-State Reservoir with Fastfood operator and astrocyte-style leak control.
    """
    def __init__(self, N, D, rho=0.9, alpha0=0.3, alpha_bounds=(0.05, 0.9), r_target=0.9, seed=0):
        super().__init__()
        assert (N & (N - 1)) == 0, "N must be power-of-two"
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.N, self.D = N, D
        self.alpha0 = alpha0
        self.amin, self.amax = alpha_bounds
        self.r_target = r_target
        self.W = FastfoodOp(N, seed=seed)
        self.W_in = nn.Linear(D, N, bias=False)
        with torch.no_grad():
            self.W_in.weight.normal_(0.0, 1.0 / math.sqrt(D))
        for p in self.W_in.parameters():
            p.requires_grad_(False)
        self.register_buffer('x', torch.zeros(1, N))
        self.register_buffer('r_x', torch.tensor([1e-3]))
        self.register_buffer('r_u', torch.tensor([1e-3]))
        self._scale_to_rho(rho)

    @torch.no_grad()
    def _scale_to_rho(self, rho: float):
        v = torch.randn(1, self.N)
        for _ in range(6):
            v = self.W(v)
            v = v / (v.norm() + 1e-8)
        s_est = self.W(v).norm()
        scale = (rho / (s_est + 1e-8)).item()
        self.W.S.mul_(scale)

    def reset_state(self, B=1, device=None):
        dev = device if device is not None else self.W_in.weight.device
        self.x = torch.zeros(B, self.N, device=dev)
        self.r_x = torch.full((B,), 1e-3, device=dev)
        self.r_u = torch.full((B,), 1e-3, device=dev)

    def step(self, u_t: torch.Tensor):
        # u_t: (B, D)
        drive = self.W_in(u_t)
        with torch.no_grad():
            self.r_u = 0.99 * self.r_u + 0.01 * (drive.pow(2).mean(dim=1).sqrt() + 1e-6)
        pre = self.W(self.x) + drive
        x_nl = torch.tanh(pre)
        with torch.no_grad():
            self.r_x = 0.99 * self.r_x + 0.01 * (x_nl.pow(2).mean(dim=1).sqrt() + 1e-6)
        ratio = self.r_x / (self.r_u + 1e-6)
        alpha = self.alpha0 * (self.r_target / (ratio + 1e-6))
        alpha = alpha.clamp(self.amin, self.amax).unsqueeze(1)
        self.x = (1 - alpha) * self.x + alpha * x_nl
        return self.x, pre

    def forward_sequence(self, U: torch.Tensor, T_tail: int = 512, pool: str = 'avg'):
        # U: (T, D) for B=1
        self.reset_state(B=1, device=U.device)
        T = U.shape[0]
        tail_states = []
        pres = []
        for t in range(T):
            x, pre = self.step(U[t:t + 1, :])
            if t >= T - T_tail:
                tail_states.append(x)
                pres.append(pre)
        tail = torch.cat(tail_states, dim=0)  # (T_tail, 1, N)
        if pool == 'avg':
            z = tail.mean(dim=0).squeeze(0)
        elif pool == 'flat':
            z = tail.transpose(0, 1).reshape(1, -1).squeeze(0)
        else:
            raise ValueError("pool must be 'avg' or 'flat'")
        return z, pres


class LinearRidgeReadout:
    """Closed-form ridge regression readout operating on NumPy arrays."""
    def __init__(self, lam=1e-3):
        self.lam = lam
        self.W = None  # (feat_dim, C)

    def fit(self, X: np.ndarray, y: List[int], C=10, device='cpu'):
        from .preprocess import to_onehot
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        Y = to_onehot(np.array(y), C).to(device)
        XT = X_t.t()
        A = XT @ X_t + self.lam * torch.eye(X_t.shape[1], device=device)
        B = XT @ Y
        W = torch.linalg.solve(A, B)  # (feat_dim, C)
        self.W = W.detach().cpu()
        print(f"[Ridge] Fit complete: feat_dim={X_t.shape[1]}, C={C}, lambda={self.lam}")

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.tensor(X, dtype=torch.float32)
        return (X_t @ self.W).numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(X)
        return logits.argmax(axis=1)


class STRIPE_SRC_Model(nn.Module):
    """End-to-end wrapper used for FGSM evaluation (encoder + reservoir + fixed linear readout)."""
    def __init__(self, img_hw, N, D, K=3, scales=(1, 2, 4), densities=(1, 0.25, 0.0625),
                 rho=0.9, alpha0=0.3, alpha_bounds=(0.05, 0.9), r_target=0.9, seed=0,
                 T_tail=512, C=10, extra_stride=1):
        super().__init__()
        self.encoder = STRIPEEncoder(img_hw=img_hw, K=K, scales=scales, densities=densities,
                                     D=D, seed=seed, extra_stride=extra_stride)
        self.src = SRCReservoir(N=N, D=D, rho=rho, alpha0=alpha0,
                                alpha_bounds=alpha_bounds, r_target=r_target, seed=seed)
        self.T_tail = T_tail
        self.readout = nn.Linear(N, C, bias=False)
        for p in self.readout.parameters():
            p.requires_grad_(False)

    def set_readout(self, W: torch.Tensor):
        with torch.no_grad():
            self.readout.weight.copy_(W.t())

    def forward(self, x: torch.Tensor):
        seqs = self.encoder(x)  # list length B; we assume B=1 in FGSM
        assert len(seqs) == 1
        z, _ = self.src.forward_sequence(seqs[0], T_tail=self.T_tail, pool='avg')
        logits = self.readout(z)
        return logits.unsqueeze(0)


@dataclass
class SRCConfig:
    N: int = 4096
    D: int = 32
    rho: float = 0.9
    alpha0: float = 0.3
    alpha_bounds: Tuple[float, float] = (0.05, 0.9)
    r_target: float = 0.9
    T_tail: int = 512
    K: int = 3
    scales: Tuple[int, ...] = (1, 2, 4)
    densities: Tuple[float, ...] = (1.0, 0.25, 0.0625)
    extra_stride: int = 1
    seed: int = 0
    lam: float = 1e-3


def collect_features_encoder_reservoir(encoder: nn.Module, reservoir: SRCReservoir,
                                       imgs: List[torch.Tensor], T_tail: int = 512, pool: str = 'avg') -> np.ndarray:
    feats = []
    for img in imgs:
        seqs = encoder(img)
        assert len(seqs) == 1
        z, _ = reservoir.forward_sequence(seqs[0], T_tail=T_tail, pool=pool)
        feats.append(z.cpu().numpy())
    return np.stack(feats, axis=0)


def make_encoder_for_ablation(img_hw: int, D: int, K: int, ablation: str, seed: int, extra_stride: int = 1) -> nn.Module:
    from .preprocess import STRIPEEncoder
    if ablation == 'no_multiscale':
        enc = STRIPEEncoder(img_hw=img_hw, K=K, scales=(1,), densities=(1.0,), D=D, seed=seed, extra_stride=extra_stride)
    elif ablation == 'random_path':
        enc = STRIPEEncoder(img_hw=img_hw, K=K, scales=(1, 2, 4), densities=(1.0, 0.25, 0.0625), D=D, seed=seed, extra_stride=extra_stride)
        with torch.no_grad():
            H = img_hw
            rand_idx = torch.randperm(H * H)
            rand_path = torch.stack([rand_idx // H, rand_idx % H], dim=1)
            enc.path.copy_(rand_path)
            # Recompute indices to reflect new path
            new_indices = {}
            for s, dens in zip(enc.scales, enc.densities):
                stride = max(1, int(round(1.0 / dens))) * enc.extra_stride
                centers = enc.path[::stride]
                idx = centers[:, 0] * img_hw + centers[:, 1]
                new_indices[str(s)] = nn.Parameter(idx, requires_grad=False)
            enc.indices = nn.ParameterDict(new_indices)
    elif ablation == 'none':
        enc = STRIPEEncoder(img_hw=img_hw, K=K, scales=(1, 2, 4), densities=(1.0, 0.25, 0.0625), D=D, seed=seed, extra_stride=extra_stride)
    else:
        raise ValueError("Ablation must be one of: 'none', 'no_multiscale', 'random_path'")
    return enc
