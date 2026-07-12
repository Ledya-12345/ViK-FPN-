import numpy as np
import matplotlib.pyplot as plt
import os

def overlay_change(rgb, change, alpha=0.6):
    rgb = rgb.permute(1, 2, 0).numpy()
    out = rgb.copy()
    red = np.zeros_like(rgb)
    red[..., 0] = 1.0
    mask = change.numpy().astype(bool)
    out[mask] = (1 - alpha) * rgb[mask] + alpha * red[mask]
    return out

def save_result(t1, t2, gt, pred, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    ax[0].imshow(t1.permute(1, 2, 0)); ax[0].set_title("T1")
    ax[1].imshow(t2.permute(1, 2, 0)); ax[1].set_title("T2")
    ax[2].imshow(overlay_change(t2, gt)); ax[2].set_title("GT (Red)")
    ax[3].imshow(overlay_change(t2, pred)); ax[3].set_title("Prediction (Red)")
    for a in ax:
        a.axis("off")
    plt.savefig(path, dpi=300)
    plt.close()
