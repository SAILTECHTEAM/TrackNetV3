import math
import os

import numpy as np
import pandas as pd
import parse
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from tracknet.core.config.constants import HEIGHT, IMG_FORMAT, SIGMA, WIDTH
from tracknet.core.utils.general import get_match_median, get_rally_dirs

data_dir = "data"


class Shuttlecock_Trajectory_Dataset(Dataset):
    """Shuttlecock_Trajectory_Dataset
    Dataset description: https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw
    """

    def __init__(
        self,
        root_dir=data_dir,
        split="train",
        seq_len=8,
        sliding_step=1,
        data_mode="heatmap",
        bg_mode="",
        frame_alpha=-1,
        rally_dir=None,
        frame_arr=None,
        pred_dict=None,
        padding=False,
        debug=False,
        HEIGHT=HEIGHT,
        WIDTH=WIDTH,
        SIGMA=SIGMA,
        median=None,
    ):
        """Initialize the dataset

        Args:
            root_dir (str): File path of root directory of the dataset
            split (str): Split of dataset, 'train', 'test' or 'val'
            seq_len (int): Length of the input sequence
            sliding_step (int): Sliding step of the sliding window during the generation of input sequences
            data_mode (str): Data mode
                Choices:
                    - 'heatmap': Return TrackNet input data
                    - 'coordinate': Return InpaintNet input data
            bg_mode (str): Background mode
                Choices:
                    - '': Return original frame sequence
                    - 'subtract': Return difference frame sequence
                    - 'subtract_concat': Return frame sequence with RGB and difference frame channels
                    - 'concat': Return frame sequence with background as the first frame
            frame_alpha (float): Frame mixup alpha
            rally_dir (str): Rally directory
            frame_arr (numpy.ndarray): Frame sequence for TrackNet inference
            pred_dict (Dict): Prediction dictionary for InpaintNet inference
                Format: {'X': x_pred (List[int]),
                         'Y': y_pred (List[int]),
                         'Visibility': vis_pred (List[int]),
                         'Inpaint_Mask': inpaint_mask (List[int]),
                         'Img_scaler': img_scaler (Tuple[int]),
                         'Img_shape': img_shape (Tuple[int])}
            padding (bool): Padding the last frame if the frame sequence is shorter than the input sequence
            debug (bool): Debug mode
            HEIGHT (int): Height of the image for input.
            WIDTH (int): Width of the image for input.
            SIGMA (int): Sigma of the Gaussian heatmap which controls the label size.
            median (numpy.ndarray): Median image
        """

        assert split in ["train", "test", "val"], (
            f"Invalid split: {split}, should be train, test or val"
        )
        assert data_mode in ["heatmap", "coordinate"], (
            f"Invalid data_mode: {data_mode}, should be heatmap or coordinate"
        )
        assert bg_mode in ["", "subtract", "subtract_concat", "concat"], (
            f'Invalid bg_mode: {bg_mode}, should be "", subtract, subtract_concat or concat'
        )

        # Image size
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH

        # Gaussian heatmap parameters
        self.mag = 1
        self.sigma = SIGMA

        self.root_dir = root_dir
        self.split = split if rally_dir is None else self._get_split(rally_dir)
        self.seq_len = seq_len
        self.sliding_step = sliding_step
        self.data_mode = data_mode
        self.bg_mode = bg_mode
        self.frame_alpha = frame_alpha

        # Data for inference
        self.frame_arr = frame_arr
        self.pred_dict = pred_dict
        self.padding = padding and self.sliding_step == self.seq_len

        # Initialize the input data
        if self.frame_arr is not None:
            # For TrackNet inference
            self.data_dict, self.img_config = self._gen_input_from_frame_arr()
        elif self.pred_dict is not None:
            # For InpaintNet inference
            self.data_dict, self.img_config = self._gen_input_from_pred_dict()
        else:
            # For training and evaluation
            if debug:
                # Debug mode: Use single rally directory
                self.rally_dict = self._get_rally_dir_dict(debug)
                self.data_dict, self.img_config = self._gen_input_from_rally_dir_dict()
            else:
                # Normal mode: Load from or generate input
                # Determine file path
                file_name = os.path.join(
                    root_dir,
                    f"{split}_{seq_len}_{sliding_step}_{data_mode}_{bg_mode}.npz",
                )
                if not os.path.exists(file_name):
                    # Generate input sequences
                    self.rally_dict = get_rally_dirs(root_dir, split)
                    self._gen_input_to_file(file_name)
                # Load input sequences
                loaded = np.load(file_name)
                self.data_dict = {
                    "id": loaded["id"],
                    "coor": loaded["coor"],
                    "coor_pred": loaded["coor_pred"],
                    "vis": loaded["vis"],
                    "pred_vis": loaded["pred_vis"],
                    "inpaint_mask": loaded["inpaint_mask"],
                }

    def _get_rally_dir_dict(self, debug):
        """Get rally directory dictionary for debug mode."""
        rally_dir_dict = {}
        rally_dir = debug

        # Get match directories and rally ID
        file_format_str = os.path.join("{}", "frame", "{}")
        match_dir, rally_id = parse.parse(file_format_str, rally_dir)

        # Get ground truth or predicted csv file
        if self.data_mode == "heatmap":
            if "test" in rally_dir:
                csv_file = os.path.join(match_dir, "corrected_csv", f"{rally_id}_ball.csv")
            else:
                csv_file = os.path.join(match_dir, "csv", f"{rally_id}_ball.csv")
            label_df = pd.read_csv(csv_file, encoding="utf8").sort_values(by="Frame")
        else:
            csv_file = os.path.join(match_dir, "predicted_csv", f"{rally_id}_ball.csv")
            label_df = pd.read_csv(csv_file, encoding="utf8").sort_values(by="Frame")

        # Get image shape
        img_file = os.path.join(rally_dir, f"{label_df['Frame'].iloc[0]}.{IMG_FORMAT}")
        img = Image.open(img_file)
        w, h = img.size

        # Get median image if needed
        if self.bg_mode:
            median_file = os.path.join(match_dir, "median.npz")
            if os.path.exists(median_file):
                median = np.load(median_file)["median"]
            else:
                median_file = os.path.join(rally_dir, "median.npz")
                if os.path.exists(median_file):
                    median = np.load(median_file)["median"]
                else:
                    median = get_match_median(rally_dir, self.HEIGHT, self.WIDTH)
                    np.savez(median_file, median=median)
                    median = np.load(median_file)["median"]
        else:
            median = None

        rally_dir_dict["i2p"] = {self._get_rally_i(rally_dir): rally_dir}
        rally_dir_dict["img_shape"] = {self._get_rally_i(rally_dir): (w, h)}
        if median is not None:
            rally_dir_dict["median"] = {self._get_rally_i(rally_dir): median}

        return rally_dir_dict

    def _get_split(self, rally_dir):
        """Get the split of the given rally directory."""
        if "test" in rally_dir or rally_dir.startswith(os.path.join(self.root_dir, "test")):
            return "test"
        elif "val" in rally_dir or rally_dir.startswith(os.path.join(self.root_dir, "val")):
            return "val"
        else:
            return "train"

    def _get_rally_i(self, rally_dir):
        """Get the rally index from the given rally directory."""
        file_format_str = os.path.join("{}", "frame", "{}")
        match_dir, rally_id = parse.parse(file_format_str, rally_dir)
        return int(rally_id)

    def _gen_input_to_file(self, file_name):
        """Generate input sequences from all rally directories and save to file."""

        id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
        coor = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
        coor_pred = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
        vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)
        pred_vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)
        inpaint_mask = np.array([], dtype=np.float32).reshape(0, self.seq_len)

        # Generate input sequences from each rally
        for rally_dir in tqdm(self.rally_dict["i2p"].values()):
            data_dict = self._gen_input_from_rally_dir(rally_dir)
            id = np.concatenate((id, data_dict["id"]), axis=0)
            coor = np.concatenate((coor, data_dict["coor"]), axis=0)
            coor_pred = np.concatenate((coor_pred, data_dict["coor_pred"]), axis=0)
            vis = np.concatenate((vis, data_dict["vis"]), axis=0)
            pred_vis = np.concatenate((pred_vis, data_dict["pred_vis"]), axis=0)
            inpaint_mask = np.concatenate((inpaint_mask, data_dict["inpaint_mask"]), axis=0)

        np.savez(
            file_name,
            id=id,
            coor=coor,
            coor_pred=coor_pred,
            vis=vis,
            pred_vis=pred_vis,
            inpaint_mask=inpaint_mask,
        )

    def _gen_input_from_rally_dir(self, rally_dir):
        """Generate input sequences from a rally directory."""

        rally_i = self._get_rally_i(rally_dir)

        file_format_str = os.path.join("{}", "frame", "{}")
        match_dir, rally_id = parse.parse(file_format_str, rally_dir)

        if self.data_mode == "heatmap":
            # Read label csv file
            if "test" in rally_dir:
                csv_file = os.path.join(match_dir, "corrected_csv", f"{rally_id}_ball.csv")
            else:
                csv_file = os.path.join(match_dir, "csv", f"{rally_id}_ball.csv")

            assert os.path.exists(csv_file), f"{csv_file} does not exist."
            label_df = pd.read_csv(csv_file, encoding="utf8").sort_values(by="Frame").fillna(0)

            f_file = np.array(
                [os.path.join(rally_dir, f"{f_id}.{IMG_FORMAT}") for f_id in label_df["Frame"]]
            )
            x, y, v = (
                np.array(label_df["X"]),
                np.array(label_df["Y"]),
                np.array(label_df["Visibility"]),
            )

            id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
            frame_file = np.array([]).reshape(0, self.seq_len)
            coor = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)

            # Sliding on the frame sequence
            last_idx = -1
            for i in range(0, len(f_file), self.sliding_step):
                tmp_idx, tmp_frames, tmp_coor, tmp_vis = [], [], [], []
                # Construct a single input sequence
                for f in range(self.seq_len):
                    if i + f < len(f_file):
                        tmp_idx.append((rally_i, i + f))
                        tmp_frames.append(f_file[i + f])
                        tmp_coor.append((x[i + f], y[i + f]))
                        tmp_vis.append(v[i + f])
                        last_idx = i + f
                    else:
                        # Padding the last sequence if imcompleted
                        if self.padding:
                            tmp_idx.append((rally_i, last_idx))
                            tmp_frames.append(f_file[last_idx])
                            tmp_coor.append((x[last_idx], y[last_idx]))
                            tmp_vis.append(v[last_idx])
                        else:
                            break

                # Append the input sequence
                if len(tmp_frames) == self.seq_len:
                    assert len(tmp_frames) == len(tmp_coor) == len(tmp_vis), (
                        "Length of frames, coordinates and visibilities are not equal."
                    )
                    id = np.concatenate((id, [tmp_idx]), axis=0)
                    frame_file = np.concatenate((frame_file, [tmp_frames]), axis=0)
                    coor = np.concatenate((coor, [tmp_coor]), axis=0)
                    vis = np.concatenate((vis, [tmp_vis]), axis=0)

            return {"id": id, "frame_file": frame_file, "coor": coor, "vis": vis}
        else:
            # Read the predicted csv file
            pred_csv_file = os.path.join(match_dir, "predicted_csv", f"{rally_id}_ball.csv")
            assert os.path.exists(pred_csv_file), f"{pred_csv_file} does not exist."
            pred_df = pd.read_csv(pred_csv_file, encoding="utf8").sort_values(by="Frame").fillna(0)

            f_file = np.array(
                [os.path.join(rally_dir, f"{f_id}.{IMG_FORMAT}") for f_id in pred_df["Frame"]]
            )
            x, y, v = (
                np.array(pred_df["X_GT"]),
                np.array(pred_df["Y_GT"]),
                np.array(pred_df["Visibility_GT"]),
            )
            x_pred, y_pred, v_pred = (
                np.array(pred_df["X"]),
                np.array(pred_df["Y"]),
                np.array(pred_df["Visibility"]),
            )
            inpaint = np.array(pred_df["Inpaint_Mask"])

            id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
            coor = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            coor_pred = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
            vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)
            pred_vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)
            inpaint_mask = np.array([], dtype=np.float32).reshape(0, self.seq_len)

            # Sliding on the frame sequence
            last_idx = -1
            for i in range(0, len(f_file), self.sliding_step):
                tmp_idx, tmp_coor, tmp_coor_pred, tmp_vis, tmp_vis_pred, tmp_inpaint = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                # Construct a single input sequence
                for f in range(self.seq_len):
                    if i + f < len(f_file):
                        tmp_idx.append((rally_i, i + f))
                        tmp_coor.append((x[i + f], y[i + f]))
                        tmp_coor_pred.append((x_pred[i + f], y_pred[i + f]))
                        tmp_vis.append(v[i + f])
                        tmp_vis_pred.append(v_pred[i + f])
                        tmp_inpaint.append(inpaint[i + f])
                    else:
                        # Padding the last sequence if imcompleted
                        if self.padding:
                            tmp_idx.append((rally_i, last_idx))
                            tmp_coor.append((x[last_idx], y[last_idx]))
                            tmp_coor_pred.append((x_pred[last_idx], y_pred[last_idx]))
                            tmp_vis.append(v[last_idx])
                            tmp_vis_pred.append(v_pred[last_idx])
                            tmp_inpaint.append(inpaint[last_idx])
                        else:
                            break

                # Append the input sequence
                if len(tmp_idx) == self.seq_len:
                    assert (
                        len(tmp_idx)
                        == len(tmp_coor)
                        == len(tmp_coor_pred)
                        == len(tmp_vis)
                        == len(tmp_vis_pred)
                        == len(tmp_inpaint)
                    ), (
                        "Length of frames, coordinates, predicted coordinates,\
                            visibilities, predicted visibilities and inpaint masks are not equal."
                    )
                    id = np.concatenate((id, [tmp_idx]), axis=0)
                    coor = np.concatenate((coor, [tmp_coor]), axis=0)
                    coor_pred = np.concatenate((coor_pred, [tmp_coor_pred]), axis=0)
                    vis = np.concatenate((vis, [tmp_vis]), axis=0)
                    pred_vis = np.concatenate((pred_vis, [tmp_vis_pred]), axis=0)
                    inpaint_mask = np.concatenate((inpaint_mask, [tmp_inpaint]), axis=0)

            return {
                "id": id,
                "coor": coor,
                "coor_pred": coor_pred,
                "vis": vis,
                "pred_vis": pred_vis,
                "inpaint_mask": inpaint_mask,
            }

    def _gen_input_from_frame_arr(self):
        """Generate input sequences from a frame array."""

        # Calculate the image scaler
        h, w, _ = self.frame_arr[0].shape
        h_scaler, w_scaler = h / self.HEIGHT, w / self.WIDTH

        id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
        last_idx = -1
        for i in range(0, len(self.frame_arr), self.sliding_step):
            tmp_idx = []
            # Construct a single input sequence
            for f in range(self.seq_len):
                if i + f < len(self.frame_arr):
                    tmp_idx.append((0, i + f))
                    last_idx = i + f
                else:
                    # Padding the last sequence if imcompleted
                    if self.padding:
                        tmp_idx.append((0, last_idx))
                    else:
                        break
            if len(tmp_idx) == self.seq_len:
                # Append the input sequence
                id = np.concatenate((id, [tmp_idx]), axis=0)

        return {"id": id}, {"img_scaler": (w_scaler, h_scaler), "img_shape": (w, h)}

    def _gen_input_from_pred_dict(self):
        """Generate input sequences from a prediction dictionary."""
        id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
        coor_pred = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2)
        pred_vis = np.array([], dtype=np.float32).reshape(0, self.seq_len)
        inpaint_mask = np.array([], dtype=np.float32).reshape(0, self.seq_len)
        x_pred, y_pred, vis_pred = (
            self.pred_dict["X"],
            self.pred_dict["Y"],
            self.pred_dict["Visibility"],
        )
        inpaint = self.pred_dict["Inpaint_Mask"]
        assert len(x_pred) == len(y_pred) == len(vis_pred) == len(inpaint), (
            "Length of x_pred, y_pred, vis_pred and inpaint are not equal."
        )

        # Sliding on the frame sequence
        last_idx = -1
        for i in range(0, len(inpaint), self.sliding_step):
            tmp_idx, tmp_coor_pred, tmp_vis_pred, tmp_inpaint = [], [], [], []
            # Construct a single input sequence
            for f in range(self.seq_len):
                if i + f < len(inpaint):
                    tmp_idx.append((0, i + f))
                    tmp_coor_pred.append((x_pred[i + f], y_pred[i + f]))
                    tmp_vis_pred.append(vis_pred[i + f])
                    tmp_inpaint.append(inpaint[i + f])
                    last_idx = i + f
                else:
                    # Padding the last sequence if imcompleted
                    if self.padding:
                        tmp_idx.append((0, last_idx))
                        tmp_coor_pred.append((x_pred[last_idx], y_pred[last_idx]))
                        tmp_vis_pred.append(vis_pred[last_idx])
                        tmp_inpaint.append(inpaint[last_idx])
                    else:
                        break

            if len(tmp_idx) == self.seq_len:
                assert len(tmp_coor_pred) == len(tmp_inpaint), (
                    "Length of predicted coordinates and inpaint masks are not equal."
                )
                id = np.concatenate((id, [tmp_idx]), axis=0)
                coor_pred = np.concatenate((coor_pred, [tmp_coor_pred]), axis=0)
                pred_vis = np.concatenate((pred_vis, [tmp_vis_pred]), axis=0)
                inpaint_mask = np.concatenate((inpaint_mask, [tmp_inpaint]), axis=0)

        return (
            {"id": id, "coor_pred": coor_pred, "pred_vis": pred_vis, "inpaint_mask": inpaint_mask},
            {"img_scaler": self.pred_dict["Img_scaler"], "img_shape": self.pred_dict["Img_shape"]},
        )

    def _get_heatmap(self, cx, cy):
        """Generate a Gaussian heatmap centered at (cx, cy)."""
        if cx == cy == 0:
            return np.zeros((1, self.HEIGHT, self.WIDTH))
        x, y = np.meshgrid(
            np.linspace(1, self.WIDTH, self.WIDTH),
            np.linspace(1, self.HEIGHT, self.HEIGHT),
        )
        heatmap = ((y - (cy + 1)) ** 2) + ((x - (cx + 1)) ** 2)
        heatmap[heatmap <= self.sigma**2] = 1.0
        heatmap[heatmap > self.sigma**2] = 0.0
        heatmap = heatmap * self.mag
        return heatmap.reshape(1, self.HEIGHT, self.WIDTH)

    def __len__(self):
        """Return the number of data in the dataset."""
        return len(self.data_dict["id"])

    def __getitem__(self, idx):
        """Return the data of the given index.

        For training and evaluation:
            'heatmap': Return data_idx, frames, heatmaps, tmp_coor, tmp_vis
            'coordinate': Return data_idx, coor_pred, inpaint

        For inference:
            'heatmap': Return data_idx, frames
            'coordinate': Return data_idx, coor_pred, inpaint
        """
        if self.frame_arr is not None:
            data_idx = self.data_dict["id"][idx]  # (L,)
            imgs = self.frame_arr[data_idx[:, 1], ...]  # (L, H, W, 3)

            if self.bg_mode:
                median_img = self.median

            # Process the frame sequence
            frames = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)
            for i in range(self.seq_len):
                img = Image.fromarray(imgs[i])
                if self.bg_mode == "subtract":
                    img = Image.fromarray(np.sum(np.absolute(img - median_img), 2).astype("uint8"))
                    img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                    img = img.reshape(1, self.HEIGHT, self.WIDTH)
                elif self.bg_mode == "subtract_concat":
                    diff_img = Image.fromarray(
                        np.sum(np.absolute(img - median_img), 2).astype("uint8")
                    )
                    diff_img = np.array(diff_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    diff_img = diff_img.reshape(1, self.HEIGHT, self.WIDTH)
                    img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                    img = np.moveaxis(img, -1, 0)
                    img = np.concatenate((img, diff_img), axis=0)
                else:
                    img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                    img = np.moveaxis(img, -1, 0)

                frames = np.concatenate((frames, img), axis=0)

            if self.bg_mode == "concat":
                frames = np.concatenate((median_img, frames), axis=0)

            # Normalization
            frames /= 255.0

            return data_idx, frames

        elif self.pred_dict is not None:
            data_idx = self.data_dict["id"][idx]  # (L,)
            coor_pred = self.data_dict["coor_pred"][idx]  # (L, 2)
            inpaint = self.data_dict["inpaint_mask"][idx].reshape(-1, 1)  # (L, 1)
            w, h = self.img_config["img_shape"]

            # Normalization
            coor_pred[:, 0] = coor_pred[:, 0] / w
            coor_pred[:, 1] = coor_pred[:, 1] / h

            return data_idx, coor_pred, inpaint

        elif self.data_mode == "heatmap":
            if self.frame_alpha > 0:
                data_idx = self.data_dict["id"][idx]  # (L,)
                frame_file = self.data_dict["frame_file"][idx]  # (L,)
                coor = self.data_dict["coor"][idx]  # (L, 2)
                vis = self.data_dict["vis"][idx]  # (L,)
                w, h = self.img_config["img_shape"][data_idx[0][0]]
                w_scaler, h_scaler = self.img_config["img_scaler"][data_idx[0][0]]

                if self.bg_mode:
                    file_format_str = os.path.join("{}", "frame", "{}", "{}." + IMG_FORMAT)
                    match_dir, rally_id, _ = parse.parse(file_format_str, frame_file[0])
                    median_file = (
                        os.path.join(match_dir, "median.npz")
                        if os.path.exists(os.path.join(match_dir, "median.npz"))
                        else os.path.join(match_dir, "frame", rally_id, "median.npz")
                    )
                    assert os.path.exists(median_file), f"{median_file} does not exist."
                    median_img = np.load(median_file)["median"]

                # Frame mixup
                # Sample the mixing ratio
                lamb = np.random.beta(self.frame_alpha, self.frame_alpha)

                # Initialize the previous frame data
                prev_img = Image.open(frame_file[0])
                if self.bg_mode == "subtract":
                    prev_img = Image.fromarray(
                        np.sum(np.absolute(prev_img - median_img), 2).astype("uint8")
                    )
                    prev_img = np.array(prev_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    prev_img = prev_img.reshape(1, self.HEIGHT, self.WIDTH)
                elif self.bg_mode == "subtract_concat":
                    diff_img = Image.fromarray(
                        np.sum(np.absolute(prev_img - median_img), 2).astype("uint8")
                    )
                    diff_img = np.array(diff_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    diff_img = diff_img.reshape(1, self.HEIGHT, self.WIDTH)
                    prev_img = np.array(prev_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    prev_img = np.moveaxis(prev_img, -1, 0)
                    prev_img = np.concatenate((prev_img, diff_img), axis=0)
                else:
                    prev_img = np.array(prev_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    prev_img = np.moveaxis(prev_img, -1, 0)

                prev_coor = coor[0]
                prev_vis = vis[0]
                prev_heatmap = self._get_heatmap(
                    int(coor[0][0] / w_scaler), int(coor[0][1] / h_scaler)
                )

                # Keep the first dimension as the timestamp for resample
                if self.bg_mode == "subtract":
                    frames = prev_img.reshape(1, 1, self.HEIGHT, self.WIDTH)
                elif self.bg_mode == "subtract_concat":
                    frames = prev_img.reshape(1, 4, self.HEIGHT, self.WIDTH)
                else:
                    frames = prev_img.reshape(1, 3, self.HEIGHT, self.WIDTH)

                tmp_coor = prev_coor.reshape(1, -1)
                tmp_vis = prev_vis.reshape(1, -1)
                heatmaps = prev_heatmap

                # Read image and generate heatmap
                for i in range(1, self.seq_len):
                    cur_img = Image.open(frame_file[i])
                    if self.bg_mode == "subtract":
                        cur_img = Image.fromarray(
                            np.sum(np.absolute(cur_img - median_img), 2).astype("uint8")
                        )
                        cur_img = np.array(cur_img.resize(size=(self.WIDTH, self.HEIGHT)))
                        cur_img = cur_img.reshape(1, self.HEIGHT, self.WIDTH)
                    elif self.bg_mode == "subtract_concat":
                        diff_img = Image.fromarray(
                            np.sum(np.absolute(cur_img - median_img), 2).astype("uint8")
                        )
                        diff_img = np.array(diff_img.resize(size=(self.WIDTH, self.HEIGHT)))
                        diff_img = diff_img.reshape(1, self.HEIGHT, self.WIDTH)
                        cur_img = np.array(cur_img.resize(size=(self.WIDTH, self.HEIGHT)))
                        cur_img = np.moveaxis(cur_img, -1, 0)
                        cur_img = np.concatenate((cur_img, diff_img), axis=0)
                    else:
                        cur_img = np.array(cur_img.resize(size=(self.WIDTH, self.HEIGHT)))
                        cur_img = np.moveaxis(cur_img, -1, 0)

                    inter_img = prev_img * lamb + cur_img * (1 - lamb)

                    # Linear interpolation
                    if vis[i] == 0:
                        inter_coor = prev_coor
                        inter_vis = prev_vis
                        cur_heatmap = prev_heatmap
                        inter_heatmap = cur_heatmap
                    elif (
                        prev_vis == 0
                        or math.sqrt(
                            pow(prev_coor[0] - coor[i][0], 2) + pow(prev_coor[1] - coor[i][1], 2)
                        )
                        < 10
                    ):
                        inter_coor = coor[i]
                        inter_vis = vis[i]
                        cur_heatmap = self._get_heatmap(
                            int(inter_coor[0] / w_scaler), int(inter_coor[1] / h_scaler)
                        )
                        inter_heatmap = cur_heatmap
                    else:
                        inter_coor = coor[i]
                        inter_vis = vis[i]
                        cur_heatmap = self._get_heatmap(
                            int(coor[i][0] / w_scaler), int(coor[i][1] / h_scaler)
                        )
                        inter_heatmap = prev_heatmap * lamb + cur_heatmap * (1 - lamb)

                    tmp_coor = np.concatenate(
                        (tmp_coor, inter_coor.reshape(1, -1), coor[i].reshape(1, -1)),
                        axis=0,
                    )
                    tmp_vis = np.concatenate(
                        (
                            tmp_vis,
                            np.array([inter_vis]).reshape(1, -1),
                            np.array([vis[i]]).reshape(1, -1),
                        ),
                        axis=0,
                    )
                    frames = np.concatenate(
                        (frames, inter_img[None, :, :, :], cur_img[None, :, :, :]),
                        axis=0,
                    )
                    heatmaps = np.concatenate((heatmaps, inter_heatmap, cur_heatmap), axis=0)

                    prev_img, prev_heatmap, prev_coor, prev_vis = (
                        cur_img,
                        cur_heatmap,
                        coor[i],
                        vis[i],
                    )

                # Resample the input sequence
                rand_id = np.random.choice(len(frames), self.seq_len, replace=False)
                rand_id = np.sort(rand_id)
                tmp_coor = tmp_coor[rand_id]
                tmp_vis = tmp_vis[rand_id]
                frames = frames[rand_id]
                heatmaps = heatmaps[rand_id]

                if self.bg_mode == "concat":
                    median_img = Image.fromarray(median_img.astype("uint8"))
                    median_img = np.array(median_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    median_img = np.moveaxis(median_img, -1, 0)
                    frames = np.concatenate(
                        (median_img.reshape(1, 3, self.HEIGHT, self.WIDTH), frames),
                        axis=0,
                    )

                # Reshape to model input format
                frames = frames.reshape(-1, self.HEIGHT, self.WIDTH)

                # Normalization
                frames /= 255.0
                tmp_coor[:, 0] = tmp_coor[:, 0] / w
                tmp_coor[:, 1] = tmp_coor[:, 1] / h

                return data_idx, frames, heatmaps, tmp_coor, tmp_vis
            else:
                data_idx = self.data_dict["id"][idx]
                frame_file = self.data_dict["frame_file"][idx]
                coor = self.data_dict["coor"][idx]
                vis = self.data_dict["vis"][idx]
                w, h = self.img_config["img_shape"][data_idx[0][0]]
                w_scaler, h_scaler = self.img_config["img_scaler"][data_idx[0][0]]

                # Read median image
                if self.bg_mode:
                    file_format_str = os.path.join("{}", "frame", "{}", "{}." + IMG_FORMAT)
                    match_dir, rally_id, _ = parse.parse(file_format_str, frame_file[0])
                    median_file = (
                        os.path.join(match_dir, "median.npz")
                        if os.path.exists(os.path.join(match_dir, "median.npz"))
                        else os.path.join(match_dir, "frame", rally_id, "median.npz")
                    )
                    assert os.path.exists(median_file), f"{median_file} does not exist."
                    median_img = np.load(median_file)["median"]

                frames = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)
                heatmaps = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)

                # Read image and generate heatmap
                for i in range(self.seq_len):
                    img = Image.open(frame_file[i])
                    if self.bg_mode == "subtract":
                        img = Image.fromarray(
                            np.sum(np.absolute(img - median_img), 2).astype("uint8")
                        )
                        img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                        img = img.reshape(1, self.HEIGHT, self.WIDTH)
                    elif self.bg_mode == "subtract_concat":
                        diff_img = Image.fromarray(
                            np.sum(np.absolute(img - median_img), 2).astype("uint8")
                        )
                        diff_img = np.array(diff_img.resize(size=(self.WIDTH, self.HEIGHT)))
                        diff_img = diff_img.reshape(1, self.HEIGHT, self.WIDTH)
                        img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                        img = np.moveaxis(img, -1, 0)
                        img = np.concatenate((img, diff_img), axis=0)
                    else:
                        img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                        img = np.moveaxis(img, -1, 0)

                    heatmap = self._get_heatmap(
                        int(coor[i][0] / w_scaler), int(coor[i][1] / h_scaler)
                    )
                    frames = np.concatenate((frames, img), axis=0)
                    heatmaps = np.concatenate((heatmaps, heatmap), axis=0)

                if self.bg_mode == "concat":
                    median_img = Image.fromarray(median_img.astype("uint8"))
                    median_img = np.array(median_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    median_img = np.moveaxis(median_img, -1, 0)
                    frames = np.concatenate((median_img, frames), axis=0)

                # Normalization
                frames /= 255.0
                coor[:, 0] = coor[:, 0] / w
                coor[:, 1] = coor[:, 1] / h

                return data_idx, frames, heatmaps, coor, vis

        elif self.data_mode == "coordinate":
            data_idx = self.data_dict["id"][idx]  # (L,)
            coor = self.data_dict["coor"][idx]  # (L, 2)
            coor_pred = self.data_dict["coor_pred"][idx]  # (L, 2)
            vis = self.data_dict["vis"][idx]  # (L,)
            vis_pred = self.data_dict["pred_vis"][idx]  # (L,)
            inpaint = self.data_dict["inpaint_mask"][idx]  # (L,)
            w, h = self.img_config["img_shape"][data_idx[0][0]]

            # Normalization
            coor[:, 0] = coor[:, 0] / self.WIDTH
            coor[:, 1] = coor[:, 1] / self.HEIGHT
            coor_pred[:, 0] = coor_pred[:, 0] / self.WIDTH
            coor_pred[:, 1] = coor_pred[:, 1] / self.HEIGHT

            return (
                data_idx,
                coor_pred,
                coor,
                vis_pred.reshape(-1, 1),
                vis.reshape(-1, 1),
                inpaint.reshape(-1, 1),
            )
        else:
            raise NotImplementedError
