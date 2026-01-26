import numpy as np
import torch

from tracknetv3.config.constants import HEIGHT, WIDTH


def _predict_from_network_outputs_fast(indices, y_pred=None, c_pred=None, img_scaler=(1, 1)):
    """
    Faster version:
    - c_pred: fully vectorized
    - y_pred: uses heatmap argmax to get (x,y) directly (no predict_location/to_img)
    """
    pred_dict = {"Frame": [], "X": [], "Y": [], "Visibility": []}

    if torch.is_tensor(indices):
        ind = indices.detach().cpu().numpy()
    else:
        ind = indices.numpy()
    frame_ids = ind[:, :, 1].astype(np.int64)

    flat_f = frame_ids.reshape(-1)
    keep = np.ones_like(flat_f, dtype=bool)
    keep[1:] = flat_f[1:] != flat_f[:-1]

    kept_frames = flat_f[keep]

    if c_pred is not None:
        if torch.is_tensor(c_pred):
            c = c_pred.detach().cpu().numpy()
        else:
            c = c_pred

        c_flat = c.reshape(-1, 2)[keep]

        xs = (c_flat[:, 0] * WIDTH * img_scaler[0]).astype(np.int32)
        ys = (c_flat[:, 1] * HEIGHT * img_scaler[1]).astype(np.int32)

    elif y_pred is not None:
        if torch.is_tensor(y_pred):
            y = y_pred.detach().float().cpu()
        else:
            y = torch.from_numpy(y_pred).float()

        if y.ndim == 4:
            B, T = ind.shape[0], ind.shape[1]
            if y.shape[0] == B and y.shape[1] == T:
                pass
            elif y.shape[1] == 1 and T == 1 and y.shape[0] == B:
                pass
            else:
                y = y.reshape(B, T, y.shape[-2], y.shape[-1])
        elif y.ndim == 5:
            y = y.squeeze(2)
        else:
            raise ValueError(f"Unexpected y_pred shape: {tuple(y.shape)}")

        y_flat = y.reshape(-1, y.shape[-2], y.shape[-1])[keep]

        _Hm, Wm = y_flat.shape[-2], y_flat.shape[-1]
        flat_idx = torch.argmax(y_flat.view(y_flat.shape[0], -1), dim=1).numpy()

        ys0 = (flat_idx // Wm).astype(np.int32)
        xs0 = (flat_idx % Wm).astype(np.int32)

        xs = (xs0.astype(np.float32) * img_scaler[0]).astype(np.int32)
        ys = (ys0.astype(np.float32) * img_scaler[1]).astype(np.int32)

    else:
        raise ValueError("Invalid input: both c_pred and y_pred are None")

    vis = ((xs != 0) | (ys != 0)).astype(np.int32)

    pred_dict["Frame"] = kept_frames.tolist()
    pred_dict["X"] = xs.tolist()
    pred_dict["Y"] = ys.tolist()
    pred_dict["Visibility"] = vis.tolist()
    return pred_dict
