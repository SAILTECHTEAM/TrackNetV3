"""Visualization utilities.

Core is kept free of heavy ML runtime dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_heatmap_pred_sample(_x, _y, y_pred, _c, bg_mode: str = "", save_dir: str = ""):
    """Plot one training sample for heatmap prediction.

    Signature required by training/tools; implementation is intentionally minimal.
    """

    # Lazy import so users without matplotlib can still import core.
    import matplotlib.pyplot as plt
    import numpy as np

    save_path = Path(save_dir) if save_dir else None
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    # best-effort visualization: show max projection of predicted heatmap
    y_pred_np = np.asarray(y_pred)
    if y_pred_np.ndim >= 2:
        ax.imshow(y_pred_np.max(axis=0), cmap="hot")
    ax.set_title(f"bg_mode={bg_mode}")
    ax.axis("off")

    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / "heatmap_pred_sample.png", bbox_inches="tight")
    plt.close(fig)


def plot_traj_pred_sample(coor_gt, refine_coor, inpaint_mask, save_dir: str = ""):
    """Plot one trajectory refinement sample."""

    import matplotlib.pyplot as plt
    import numpy as np

    save_path = Path(save_dir) if save_dir else None

    coor_gt = np.asarray(coor_gt)
    refine_coor = np.asarray(refine_coor)
    inpaint_mask = np.asarray(inpaint_mask)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    if coor_gt.size:
        ax.plot(coor_gt[:, 0], coor_gt[:, 1], label="gt")
    if refine_coor.size:
        ax.plot(refine_coor[:, 0], refine_coor[:, 1], label="pred")
    if inpaint_mask.size:
        ax.plot(inpaint_mask, label="mask")
    ax.legend()

    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / "traj_pred_sample.png", bbox_inches="tight")
    plt.close(fig)


def write_to_tb(model_name: str, tb_writer: Any, losses: Any, val_res: Any, epoch: int):
    """Write metrics to an existing tensorboard writer.

    This function does not depend on tensorboard; it expects a writer-like object.
    """

    if tb_writer is None:
        return

    train_loss, val_loss = None, None
    try:
        train_loss, val_loss = losses
    except Exception:
        pass

    if train_loss is not None:
        tb_writer.add_scalar(f"{model_name}/loss/train", float(train_loss), epoch)
    if val_loss is not None:
        tb_writer.add_scalar(f"{model_name}/loss/val", float(val_loss), epoch)

    if isinstance(val_res, dict):
        for k, v in val_res.items():
            try:
                tb_writer.add_scalar(f"{model_name}/{k}", float(v), epoch)
            except Exception:
                continue


def plot_median_files(data_dir):
    """Plot median frames saved under a dataset directory."""

    import os

    import matplotlib.pyplot as plt
    import numpy as np

    medians = []
    for root, _dirs, files in os.walk(str(data_dir)):
        if "median.npz" in files:
            try:
                medians.append(np.load(os.path.join(root, "median.npz"))["median"])
            except Exception:
                continue

    if not medians:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.imshow(np.median(np.array(medians), axis=0).astype(np.uint8))
    ax.axis("off")
    plt.close(fig)
