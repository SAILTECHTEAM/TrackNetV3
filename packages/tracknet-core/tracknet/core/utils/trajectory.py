"""Trajectory processing utilities."""

import numpy as np


def generate_inpaint_mask(pred_dict, th_h=30):
    """Generate inpaint mask form predicted trajectory.

    Args:
        pred_dict (Dict): Prediction result
            Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
        th_h (float): Height threshold (pixels) for y coordinate

    Returns:
        inpaint_mask (List): Inpaint mask
    """
    y = np.array(pred_dict["Y"])
    vis_pred = np.array(pred_dict["Visibility"])
    inpaint_mask = np.zeros_like(y)
    i = 0  # index that ball start to disappear
    j = 0  # index that ball start to appear
    threshold = th_h
    while j < len(vis_pred):
        while i < len(vis_pred) - 1 and vis_pred[i] == 1:
            i += 1
        j = i
        while j < len(vis_pred) - 1 and vis_pred[j] == 0:
            j += 1
        if j == i:
            break
        elif i == 0 and y[j] > threshold:
            # start from the first frame that ball disappear
            inpaint_mask[:j] = 1
        elif (i > 1 and y[i - 1] > threshold) and (j < len(vis_pred) and y[j] > threshold):
            inpaint_mask[i:j] = 1
        else:
            # ball is out of the field of camera view
            pass
        i = j

    return inpaint_mask.tolist()


def linear_interp(target, inpaint_mask):
    assert len(target) == len(inpaint_mask), "Length of target and inpaint_mask should be the same"
    target = np.array(target)
    inpaint_mask = np.array(inpaint_mask)
    i = 0  # index that ball start to disappear
    j = 0  # index that ball start to appear
    while j < len(inpaint_mask):
        while i < len(inpaint_mask) - 1 and inpaint_mask[i] == 0:
            i += 1
        j = i
        while j < len(inpaint_mask) - 1 and inpaint_mask[j] == 1:
            j += 1
        if j == i:
            break
        else:
            x = np.linspace(0, 1, len(inpaint_mask[i:j]))
            xp = [0, 1]
            if i == 0:
                fp = [target[j], target[j]]
            elif j == len(inpaint_mask) - 1:
                fp = [target[i - 1], target[i - 1]]
            else:
                fp = [target[i - 1], target[j]]
            target[i:j] = np.interp(x, xp, fp)
        i = j

    return target
