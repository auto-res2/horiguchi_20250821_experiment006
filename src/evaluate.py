import os
import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from .train import LinearRidgeReadout, SRCReservoir
from .preprocess import add_gaussian_noise, translate_zeropad, to_onehot
from .train import collect_features_encoder_reservoir

IMAGE_DIR = os.path.join('.research', 'iteration1', 'images')
os.makedirs(IMAGE_DIR, exist_ok=True)

# Improve PDF quality
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['savefig.dpi'] = 300


def run_clean_accuracy_latency(dataset_name: str,
                               encoders: Dict[str, torch.nn.Module],
                               reservoirs: Dict[str, SRCReservoir],
                               train_imgs: List[torch.Tensor], train_labels: List[int],
                               test_imgs: List[torch.Tensor], test_labels: List[int],
                               lam: float = 1e-3) -> Dict[str, Dict[str, float]]:
    import time
    results = {}
    for name in encoders.keys():
        encoder = encoders[name]
        reservoir = reservoirs[name]
        print(f"[Exp1] Collecting features for {name} ...")
        tic = time.perf_counter()
        X_train = collect_features_encoder_reservoir(encoder, reservoir, train_imgs,
                                                     T_tail=getattr(reservoir, 'T_tail', 512))
        toc = time.perf_counter()
        print(f"[Exp1] {name} train feature time: {toc - tic:.2f}s, shape={X_train.shape}")
        ridge = LinearRidgeReadout(lam=lam)
        ridge.fit(X_train, train_labels, C=10)
        # Eval + latency
        preds = []
        tic = time.perf_counter()
        for img in test_imgs:
            X_test = collect_features_encoder_reservoir(encoder, reservoir, [img],
                                                        T_tail=getattr(reservoir, 'T_tail', 512))
            pred = ridge.predict(X_test)[0]
            preds.append(pred)
        toc = time.perf_counter()
        acc = np.mean(np.array(preds) == np.array(test_labels))
        latency = (toc - tic) / max(1, len(test_imgs))
        results[name] = {"accuracy": float(acc), "latency_s": float(latency)}
        print(f"[Exp1] {name}: accuracy={acc*100:.2f}%, latency/sample={latency*1e3:.2f} ms")
    return results


def fgsm_attack(model: torch.nn.Module, img: torch.Tensor, label: int, eps: float = 0.15) -> torch.Tensor:
    x = img.clone().detach().requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(logits, torch.tensor([label]))
    loss.backward()
    x_adv = (x + eps * x.grad.sign()).clamp(0, 1).detach()
    return x_adv


def evaluate_robustness(model: torch.nn.Module, test_imgs: List[torch.Tensor], test_labels: List[int],
                        sigmas=(0.1, 0.3, 0.5), shift_px=5, eps=0.15) -> Dict[str, float]:
    # Clean
    correct_clean = 0
    for img, y in zip(test_imgs, test_labels):
        with torch.no_grad():
            pred = model(img).argmax(dim=1).item()
        correct_clean += int(pred == y)
    clean_acc = correct_clean / len(test_imgs)

    # Gaussian noise
    noise_accs = {}
    for s in sigmas:
        corr = 0
        for img, y in zip(test_imgs, test_labels):
            nimg = add_gaussian_noise(img, s)
            with torch.no_grad():
                pred = model(nimg).argmax(dim=1).item()
            corr += int(pred == y)
        noise_accs[f"gauss_{s}"] = corr / len(test_imgs)

    # Translation
    corr = 0
    for img, y in zip(test_imgs, test_labels):
        timg = translate_zeropad(img, shift_px)
        with torch.no_grad():
            pred = model(timg).argmax(dim=1).item()
        corr += int(pred == y)
    trans_acc = corr / len(test_imgs)

    # FGSM success rate
    n_correct = 0
    n_flip = 0
    for img, y in zip(test_imgs, test_labels):
        with torch.no_grad():
            pred = model(img).argmax(dim=1).item()
        if pred == y:
            n_correct += 1
            x_adv = fgsm_attack(model, img, y, eps=eps)
            with torch.no_grad():
                pred_adv = model(x_adv).argmax(dim=1).item()
            if pred_adv != y:
                n_flip += 1
    fgsm_succ = n_flip / max(1, n_correct)

    res = {"clean": clean_acc, **noise_accs, "translate_5px": trans_acc, "fgsm_succ": fgsm_succ}
    print("[Exp2] Robustness results:")
    for k, v in res.items():
        if k == 'fgsm_succ':
            print(f"  - {k}: {v*100:.2f}% (success rate)")
        else:
            print(f"  - {k}: {v*100:.2f}%")
    return res


@torch.no_grad()
def jacobian_spectral_radius(src: SRCReservoir, pres_list: List[torch.Tensor]) -> float:
    vals = []
    for pre in pres_list:
        d = 1.0 - torch.tanh(pre).pow(2)  # (1, N)
        d = d.squeeze(0)
        v = torch.randn(1, src.N, device=pre.device)
        for _ in range(10):
            v = src.W(v)
            v = v * d
            v = v / (v.norm() + 1e-8)
        val = (src.W(v) * d).norm().item()
        vals.append(val)
    return float(np.mean(vals)) if len(vals) > 0 else float('nan')


def plot_accuracy_bar(results: Dict[str, Dict[str, float]], title: str, filename: str):
    labels = list(results.keys())
    accs = [results[k]['accuracy'] * 100.0 for k in labels]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=accs, color='steelblue')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.ylim(0, 100)
    for i, a in enumerate(accs):
        plt.text(i, a + 1, f"{a:.1f}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    out = os.path.join(IMAGE_DIR, f"{filename}.pdf")
    plt.savefig(out, bbox_inches='tight')
    plt.close()


def plot_latency_bar(results: Dict[str, Dict[str, float]], title: str, filename: str):
    labels = list(results.keys())
    lats = [results[k]['latency_s'] * 1e3 for k in labels]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=lats, color='darkorange')
    plt.ylabel('Latency (ms/sample)')
    plt.title(title)
    for i, a in enumerate(lats):
        plt.text(i, a + max(lats) * 0.02, f"{a:.2f}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    out = os.path.join(IMAGE_DIR, f"{filename}.pdf")
    plt.savefig(out, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, filename: str):
    C = int(max(y_true.max(), y_pred.max()) + 1)
    cm = np.zeros((C, C), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    out = os.path.join(IMAGE_DIR, f"{filename}.pdf")
    plt.savefig(out, bbox_inches='tight')
    plt.close()


def plot_robustness_curves(robust_res: Dict[str, float], title: str, filename: str):
    # Gaussian
    ys = [robust_res.get(f"gauss_{s}", float('nan')) * 100 for s in [0.1, 0.3, 0.5]]
    plt.figure(figsize=(6, 4))
    plt.plot([0.1, 0.3, 0.5], ys, marker='o')
    plt.xlabel('Gaussian noise σ')
    plt.ylabel('Accuracy (%)')
    plt.title(title + ' (Gaussian)')
    plt.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    out1 = os.path.join(IMAGE_DIR, f"{filename}_gaussian.pdf")
    plt.savefig(out1, bbox_inches='tight')
    plt.close()

    # Translation
    plt.figure(figsize=(5, 4))
    trans = robust_res.get('translate_5px', float('nan')) * 100
    clean = robust_res.get('clean', float('nan')) * 100
    sns.barplot(x=['clean', 'shift_5px'], y=[clean, trans], palette=['#4C72B0', '#55A868'])
    plt.ylabel('Accuracy (%)')
    plt.title(title + ' (Translation)')
    plt.tight_layout()
    out2 = os.path.join(IMAGE_DIR, f"{filename}_translation.pdf")
    plt.savefig(out2, bbox_inches='tight')
    plt.close()

    # FGSM
    plt.figure(figsize=(4.5, 4))
    succ = robust_res.get('fgsm_succ', float('nan')) * 100
    sns.barplot(x=['FGSM ε=0.15'], y=[succ], color='#C44E52')
    plt.ylabel('Success rate (%)')
    plt.title(title + ' (FGSM)')
    plt.tight_layout()
    out3 = os.path.join(IMAGE_DIR, f"{filename}_fgsm.pdf")
    plt.savefig(out3, bbox_inches='tight')
    plt.close()


def plot_sr_vs_margin(sr_vals: List[float], margins: List[float], title: str, filename: str):
    plt.figure(figsize=(5, 4))
    plt.scatter(sr_vals, margins, s=12, alpha=0.6)
    plt.xlabel('Mean Jacobian spectral radius (proxy)')
    plt.ylabel('Logit margin (top - second)')
    plt.title(title)
    plt.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    out = os.path.join(IMAGE_DIR, f"{filename}.pdf")
    plt.savefig(out, bbox_inches='tight')
    plt.close()


def estimate_MACs_STRIPE(N: int, D: int, encoder) -> int:
    T = 0
    for s in encoder.scales:
        T += encoder.indices[str(s)].numel()
    kH = 2  # two FWHTs
    C_step = D * N + kH * N * int(math.log2(N)) + 5 * N  # rough estimate
    return int(T * C_step)
