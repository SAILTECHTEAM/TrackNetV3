"""Evaluation metrics for TrackNetV3."""

import math

import numpy as np
import torch

from tracknetv3.config.constants import HEIGHT, WIDTH
from tracknetv3.evaluation.predict import predict_location
from tracknetv3.utils.general import to_img, to_img_format

pred_types = ["TP", "TN", "FP1", "FP2", "FN"]
pred_types_map = {pred_type: i for i, pred_type in enumerate(pred_types)}


def evaluate(
    indices,
    y_true=None,
    y_pred=None,
    c_true=None,
    c_pred=None,
    tolerance=4.0,
    img_scaler=(1, 1),
    output_bbox=False,
    output_gt=False,
):
    """Predict and output the result of each frame.

    Args:
        indices (torch.Tensor) - Indices with shape (N, L, 2)
        y_true (torch.Tensor, optional) - Ground-truth heatmap sequences with shape (N, L, H, W)
        y_pred (torch.Tensor, optional) - Predicted heatmap sequences with shape (N, L, H, W)
        c_true (torch.Tensor, optional) - Ground-truth coordinate sequences with shape (N, L, 2)
        c_pred (torch.Tensor, optional) - Predicted coordinate sequences with shape (N, L, 2)
        tolerance (float) - Tolerance for FP1
        img_scaler (Tuple[float, float]) - Scaler of input image size to original image size
        output_bbox (bool) - Whether to output detection result

    Returns:
        pred_dict (Dict) - Prediction result
            Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Type':[], 'BBox': [], 'Confidence':[]}}
    """

    pred_dict = {
        "Frame": [],
        "X": [],
        "Y": [],
        "Visibility": [],
        "Type": [],
        "BBox": [],
        "Confidence": [],
        "X_GT": [],
        "Y_GT": [],
        "Visibility_GT": [],
    }

    # Pre-initialize variables to satisfy static analysis and ensure defined
    h_pred = None
    bbox_pred = (0, 0, 0, 0)
    conf = 0.0

    batch_size, seq_len = indices.shape[0], indices.shape[1]
    indices = (
        indices.detach().cpu().numpy().tolist()
        if torch.is_tensor(indices)
        else indices.numpy().tolist()
    )

    if y_true is not None and y_pred is not None:
        if c_true is not None or c_pred is not None:
            raise ValueError(
                "Invalid input: provide either heatmap tensors or coordinate tensors, not both"
            )
        y_true = y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else y_true
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_true = to_img_format(y_true)
        y_pred = to_img_format(y_pred)
        h_pred = y_pred > 0.5

    if c_true is not None and c_pred is not None:
        if y_true is not None or y_pred is not None:
            raise ValueError(
                "Invalid input: provide either coordinate tensors or heatmap tensors, not both"
            )
        if output_bbox:
            raise ValueError("Coordinate prediction cannot output detection")
        c_true = c_true.detach().cpu().numpy() if torch.is_tensor(c_true) else c_true
        c_pred = c_pred.detach().cpu().numpy() if torch.is_tensor(c_pred) else c_pred
        c_true[..., 0] = c_true[..., 0] * WIDTH
        c_true[..., 1] = c_true[..., 1] * HEIGHT
        c_pred[..., 0] = c_pred[..., 0] * WIDTH
        c_pred[..., 1] = c_pred[..., 1] * HEIGHT

    for n in range(batch_size):
        prev_d_i = [-1, -1]
        for f in range(seq_len):
            d_i = indices[n][f]
            if d_i != prev_d_i:
                if c_true is not None and c_pred is not None:
                    c_t = c_true[n][f]
                    c_p = c_pred[n][f]
                    cx_true, cy_true = int(c_t[0]), int(c_t[1])
                    cx_pred, cy_pred = int(c_p[0]), int(c_p[1])
                    vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                    if np.amax(c_p) == 0 and np.amax(c_t) == 0:
                        pred_dict["Type"].append(pred_types_map["TN"])
                    elif np.amax(c_p) > 0 and np.amax(c_t) == 0:
                        pred_dict["Type"].append(pred_types_map["FP2"])
                    elif np.amax(c_p) == 0 and np.amax(c_t) > 0:
                        pred_dict["Type"].append(pred_types_map["FN"])
                    elif np.amax(c_p) > 0 and np.amax(c_t) > 0:
                        dist = math.sqrt(pow(cx_pred - cx_true, 2) + pow(cy_pred - cy_true, 2))
                        if dist > tolerance:
                            pred_dict["Type"].append(pred_types_map["FP1"])
                        else:
                            pred_dict["Type"].append(pred_types_map["TP"])
                    else:
                        raise ValueError(f"Invalid input: {c_p}, {c_t}")
                elif y_true is not None and y_pred is not None:
                    y_t = y_true[n][f]
                    y_p = y_pred[n][f]
                    h_p = h_pred[n][f]
                    bbox_true = predict_location(to_img(y_t))
                    cx_true, cy_true = (
                        int(bbox_true[0] + bbox_true[2] / 2),
                        int(bbox_true[1] + bbox_true[3] / 2),
                    )
                    bbox_pred = predict_location(to_img(h_p))
                    cx_pred, cy_pred = (
                        int(bbox_pred[0] + bbox_pred[2] / 2),
                        int(bbox_pred[1] + bbox_pred[3] / 2),
                    )
                    if np.amax(bbox_pred) > 0:
                        conf = np.amax(
                            y_p[
                                bbox_pred[1] : bbox_pred[1] + bbox_pred[3],
                                bbox_pred[0] : bbox_pred[0] + bbox_pred[2],
                            ]
                        )
                    else:
                        conf = 0.0
                    vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                    if np.amax(h_p) == 0 and np.amax(y_t) == 0:
                        pred_dict["Type"].append(pred_types_map["TN"])
                    elif np.amax(h_p) > 0 and np.amax(y_t) == 0:
                        pred_dict["Type"].append(pred_types_map["FP2"])
                    elif np.amax(h_p) == 0 and np.amax(y_t) > 0:
                        pred_dict["Type"].append(pred_types_map["FN"])
                    elif np.amax(h_p) > 0 and np.amax(y_t) > 0:
                        dist = math.sqrt(pow(cx_pred - cx_true, 2) + pow(cy_pred - cy_true, 2))
                        if dist > tolerance:
                            pred_dict["Type"].append(pred_types_map["FP1"])
                        else:
                            pred_dict["Type"].append(pred_types_map["TP"])
                    else:
                        raise ValueError("Invalid input")
                else:
                    raise ValueError("Invalid input")
                pred_dict["Frame"].append(int(d_i[1]))
                pred_dict["X"].append(int(cx_pred * img_scaler[0]))
                pred_dict["Y"].append(int(cy_pred * img_scaler[1]))
                pred_dict["Visibility"].append(vis_pred)

                if output_bbox:
                    pred_dict["BBox"].append(
                        [
                            int(bbox_pred[0] * img_scaler[0]),
                            int(bbox_pred[1] * img_scaler[1]),
                            int(bbox_pred[2] * img_scaler[0]),
                            int(bbox_pred[3] * img_scaler[1]),
                        ]
                    )
                    pred_dict["Confidence"].append(float(conf))

                if output_gt:
                    vis_gt = 0 if cx_true == 0 and cy_true == 0 else 1
                    pred_dict["X_GT"].append(int(cx_true * img_scaler[0]))
                    pred_dict["Y_GT"].append(int(cy_true * img_scaler[1]))
                    pred_dict["Visibility_GT"].append(vis_gt)

                prev_d_i = d_i
            else:
                break

    if not output_bbox:
        del pred_dict["BBox"]
        del pred_dict["Confidence"]

    if not output_gt:
        del pred_dict["X_GT"]
        del pred_dict["Y_GT"]
        del pred_dict["Visibility_GT"]

    return pred_dict
