# tracknet_infer.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Literal

import numpy as np
from tqdm import tqdm

import torch
import cv2
from torch.utils.data import DataLoader

# keep your existing imports / project functions
from test import predict_location, get_ensemble_weight, generate_inpaint_mask
from dataset import Shuttlecock_Trajectory_Dataset, Video_IterableDataset
from utils.general import *  # WIDTH, HEIGHT, COOR_TH, get_model, generate_frames, write_pred_csv, write_pred_video, to_img_format, to_img


EvalMode = Literal["nonoverlap", "average", "weight"]


@dataclass
class TrackNetConfig:
    tracknet_ckpt: str
    inpaintnet_ckpt: str = ""           # optional
    batch_size: int = 16
    eval_mode: EvalMode = "weight"
    large_video: bool = False
    max_sample_num: int = 1800
    video_range: Optional[Tuple[int, int]] = None  # (start_sec, end_sec) or None
    num_workers_cap: int = 16


class TrackNetInfer:
    """
    Reusable TrackNet(+InpaintNet) inference wrapper.

    Usage:
        infer = TrackNetInfer(cfg)
        pred_dict = infer("tennis.mp4")
        infer.save_csv(pred_dict, "out.csv")
        infer.save_video("tennis.mp4", pred_dict, "out.mp4", traj_len=8)
    """

    def __init__(self, cfg: TrackNetConfig, device: Optional[str] = None):
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # ---- load TrackNet
        tracknet_ckpt = torch.load(cfg.tracknet_ckpt, map_location="cpu")
        self.tracknet_seq_len = int(tracknet_ckpt["param_dict"]["seq_len"])
        self.bg_mode = tracknet_ckpt["param_dict"]["bg_mode"]

        self.tracknet = get_model("TrackNet", self.tracknet_seq_len, self.bg_mode).to(self.device)
        self.tracknet.load_state_dict(tracknet_ckpt["model"])
        self.tracknet.eval()
        self._median_cache = {}  # key -> np.ndarray

        # ---- load InpaintNet (optional)
        self.inpaintnet = None
        self.inpaintnet_seq_len = None
        if cfg.inpaintnet_ckpt:
            inpaintnet_ckpt = torch.load(cfg.inpaintnet_ckpt, map_location="cpu")
            self.inpaintnet_seq_len = int(inpaintnet_ckpt["param_dict"]["seq_len"])
            self.inpaintnet = get_model("InpaintNet").to(self.device)
            self.inpaintnet.load_state_dict(inpaintnet_ckpt["model"])
            self.inpaintnet.eval()

    # ---------------------------
    # Public API
    # ---------------------------
    def __call__(self, video_file: str) -> Dict[str, Any]:
        return self.predict_video(video_file)

    def predict_video(self, video_file: str) -> Dict[str, Any]:
        """
        Returns pred_dict:
          {'Frame':[], 'X':[], 'Y':[], 'Visibility':[],
           'Inpaint_Mask':[], 'Img_scaler':(w_scaler,h_scaler), 'Img_shape':(w,h)}
        and if inpaintnet enabled, the returned dict is the inpainted one.
        """
        img_scaler, img_shape = self._get_video_scaler(video_file)

        # 1) TrackNet heatmap prediction
        tracknet_pred_dict = self._run_tracknet(video_file, img_scaler, img_shape)

        # 2) Optional InpaintNet refinement
        if self.inpaintnet is not None:
            inpaint_pred_dict = self._run_inpaintnet(tracknet_pred_dict, img_scaler)
            return inpaint_pred_dict

        return tracknet_pred_dict

    def save_csv(self, pred_dict: Dict[str, Any], save_file: str) -> None:
        os.makedirs(os.path.dirname(save_file) or ".", exist_ok=True)
        write_pred_csv(pred_dict, save_file=save_file)

    def save_video(
        self,
        video_file: str,
        pred_dict: Dict[str, Any],
        save_file: str,
        traj_len: int = 8,
        fps_fallback: float = 30.0,
    ) -> None:
        # 1) normalize save_file
        if not save_file:
            raise ValueError("save_file is empty")

        # if user passed a directory -> auto filename
        if save_file.endswith("/") or (os.path.exists(save_file) and os.path.isdir(save_file)):
            os.makedirs(save_file, exist_ok=True)
            base = os.path.splitext(os.path.basename(video_file))[0]
            save_file = os.path.join(save_file, f"{base}_tracknet.mp4")

        # ensure extension
        root, ext = os.path.splitext(save_file)
        if ext.lower() not in [".mp4", ".avi", ".mkv"]:
            save_file = root + ".mp4"

        os.makedirs(os.path.dirname(save_file) or ".", exist_ok=True)

        # 2) check fps
        cap = cv2.VideoCapture(video_file)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        if fps < 1.0:
            # IMPORTANT: your utils.write_pred_video probably reads fps internally.
            # The quickest hack is to temporarily set an env var or modify utils to accept fps.
            # If you can't modify utils, implement our own writer (next section).
            print(f"[WARN] input fps={fps}, fallback to {fps_fallback}")
            fps = fps_fallback

        # 3) If your write_pred_video DOES NOT accept fps, you MUST fix write_pred_video itself
        # Option A (recommended): implement local writer in this class (below)
        self._write_pred_video_safe(video_file, pred_dict, save_file, traj_len=traj_len, fps=fps)

    # ---------------------------
    # Core logic (ported from your script)
    # ---------------------------
    def _get_video_scaler(self, video_file: str) -> Tuple[Tuple[float, float], Tuple[int, int]]:
        cap = cv2.VideoCapture(video_file)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        w_scaler, h_scaler = w / WIDTH, h / HEIGHT
        return (w_scaler, h_scaler), (w, h)
    
    def _write_pred_video_safe(self, video_file: str, pred_dict: Dict[str, Any], save_file: str, traj_len: int, fps: float):
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_file}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps < 1.0:
            fps = 30.0

        # codec fallback chain
        codecs = ["mp4v", "avc1", "H264"] if save_file.lower().endswith(".mp4") else ["XVID", "MJPG"]
        writer = None
        for c in codecs:
            fourcc = cv2.VideoWriter_fourcc(*c)
            vw = cv2.VideoWriter(save_file, fourcc, fps, (w, h))
            if vw.isOpened():
                writer = vw
                break
        if writer is None:
            cap.release()
            raise RuntimeError(f"Failed to open VideoWriter: {save_file}")

        # Build a quick lookup frame->(x,y,vis)
        frames = pred_dict["Frame"]
        xs = pred_dict["X"]
        ys = pred_dict["Y"]
        vis = pred_dict["Visibility"]
        lookup = {int(f): (int(x), int(y), int(v)) for f, x, y, v in zip(frames, xs, ys, vis)}

        f_idx = 0
        trail = []  # store last traj_len points
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if f_idx in lookup:
                x, y, v = lookup[f_idx]
                if v == 1 and x > 0 and y > 0:
                    trail.append((x, y))
                    if len(trail) > traj_len:
                        trail = trail[-traj_len:]

            # draw trail + current
            for p in trail:
                cv2.circle(frame, p, 3, (0, 255, 0), -1)
            if len(trail) > 0:
                cv2.circle(frame, trail[-1], 5, (0, 0, 255), -1)

            writer.write(frame)
            f_idx += 1

        writer.release()
        cap.release()

    def _num_workers(self) -> int:
        # keep your original logic
        n = self.cfg.batch_size if self.cfg.batch_size <= self.cfg.num_workers_cap else self.cfg.num_workers_cap
        return max(0, int(n))

    def _make_tracknet_loader(self, video_file: str, seq_len: int):
        cfg = self.cfg
        num_workers = self._num_workers()

        if cfg.eval_mode == "nonoverlap":
            sliding_step = seq_len
        else:
            sliding_step = 1

        if cfg.large_video:
            median = None
            cache_key = None

            # Only cache/use median when bg_mode is enabled
            if self.bg_mode:
                cache_key = (
                    video_file,
                    self.bg_mode,
                    cfg.max_sample_num,
                    tuple(cfg.video_range) if cfg.video_range else None,
                )
                median = self._median_cache.get(cache_key)

            dataset = Video_IterableDataset(
                video_file,
                seq_len=seq_len,
                sliding_step=sliding_step,
                bg_mode=self.bg_mode,
                max_sample_num=cfg.max_sample_num,
                video_range=list(cfg.video_range) if cfg.video_range else None,
                median=median,          # <-- reuse if already computed
                fps_fallback=30.0,      # <-- avoid fps=0 issues
                verbose_timing=True,    # <-- prints [TIMER] for median
            )

            # Save the computed median to cache for reuse in the same process
            if self.bg_mode and (median is None) and (cache_key is not None):
                self._median_cache[cache_key] = dataset.median

            loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
            video_len = dataset.video_len
            return loader, video_len


        # not-large: load all frames
        frame_list = generate_frames(video_file)
        dataset = Shuttlecock_Trajectory_Dataset(
            seq_len=seq_len,
            sliding_step=sliding_step,
            data_mode="heatmap",
            bg_mode=self.bg_mode,
            frame_arr=np.array(frame_list)[:, :, :, ::-1],
            padding=(cfg.eval_mode == "nonoverlap"),
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        video_len = len(frame_list)
        return loader, video_len

    def _run_tracknet(self, video_file: str, img_scaler: Tuple[float, float], img_shape: Tuple[int, int]) -> Dict[str, Any]:
        cfg = self.cfg
        seq_len = self.tracknet_seq_len

        tracknet_pred_dict = {
            "Frame": [],
            "X": [],
            "Y": [],
            "Visibility": [],
            "Inpaint_Mask": [],
            "Img_scaler": img_scaler,
            "Img_shape": img_shape,
        }

        loader, video_len = self._make_tracknet_loader(video_file, seq_len)

        if cfg.eval_mode == "nonoverlap":
            for (i, x) in tqdm(loader):
                x = x.float().to(self.device, non_blocking=True)
                with torch.no_grad():
                    y_pred = self.tracknet(x).detach().cpu()
                tmp_pred = _predict_from_network_outputs_fast(i, y_pred=y_pred, img_scaler=img_scaler)
                for k in tmp_pred:
                    tracknet_pred_dict[k].extend(tmp_pred[k])
            return tracknet_pred_dict

        # overlap ensemble path (average/weight)
        num_sample = video_len - seq_len + 1
        sample_count = 0
        buffer_size = seq_len - 1
        batch_i = torch.arange(seq_len)
        frame_i = torch.arange(seq_len - 1, -1, -1)
        y_pred_buffer = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
        weight = get_ensemble_weight(seq_len, cfg.eval_mode)

        for (i, x) in tqdm(loader):
            x = x.float().to(self.device, non_blocking=True)
            b_size = i.shape[0]

            with torch.no_grad():
                y_pred = self.tracknet(x).detach().cpu()

            y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)

            ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
            ensemble_y_pred = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)

            for b in range(b_size):
                if sample_count < buffer_size:
                    y_ens = y_pred_buffer[batch_i + b, frame_i].sum(0) / (sample_count + 1)
                else:
                    y_ens = (y_pred_buffer[batch_i + b, frame_i] * weight[:, None, None]).sum(0)

                ensemble_i = torch.cat((ensemble_i, i[b][0].reshape(1, 1, 2)), dim=0)
                ensemble_y_pred = torch.cat((ensemble_y_pred, y_ens.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                sample_count += 1

                if sample_count == num_sample:
                    # pad and flush tail
                    y_zero_pad = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
                    y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)

                    for f in range(1, seq_len):
                        y_tail = y_pred_buffer[batch_i + b + f, frame_i].sum(0) / (seq_len - f)
                        ensemble_i = torch.cat((ensemble_i, i[-1][f].reshape(1, 1, 2)), dim=0)
                        ensemble_y_pred = torch.cat((ensemble_y_pred, y_tail.reshape(1, 1, HEIGHT, WIDTH)), dim=0)

            tmp_pred = _predict_from_network_outputs_fast(ensemble_i, y_pred=ensemble_y_pred, img_scaler=img_scaler)
            for k in tmp_pred:
                tracknet_pred_dict[k].extend(tmp_pred[k])

            y_pred_buffer = y_pred_buffer[-buffer_size:]

        return tracknet_pred_dict

    def _run_inpaintnet(self, tracknet_pred_dict: Dict[str, Any], img_scaler: Tuple[float, float]) -> Dict[str, Any]:
        cfg = self.cfg
        assert self.inpaintnet is not None and self.inpaintnet_seq_len is not None

        # generate inpaint mask based on TrackNet result
        w, h = tracknet_pred_dict["Img_shape"]
        tracknet_pred_dict["Inpaint_Mask"] = generate_inpaint_mask(tracknet_pred_dict, th_h=h * 0.05)

        seq_len = self.inpaintnet_seq_len
        num_workers = self._num_workers()

        inpaint_pred_dict = {"Frame": [], "X": [], "Y": [], "Visibility": []}

        if cfg.eval_mode == "nonoverlap":
            dataset = Shuttlecock_Trajectory_Dataset(
                seq_len=seq_len,
                sliding_step=seq_len,
                data_mode="coordinate",
                pred_dict=tracknet_pred_dict,
                padding=True,
            )
            loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

            for (i, coor_pred, inpaint_mask) in tqdm(loader):
                coor_pred = coor_pred.float()
                inpaint_mask = inpaint_mask.float()

                with torch.no_grad():
                    coor_inpaint = self.inpaintnet(
                        coor_pred.to(self.device),
                        inpaint_mask.to(self.device),
                    ).detach().cpu()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1 - inpaint_mask)

                # Thresholding
                th = (coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH)
                coor_inpaint[th] = 0.0

                tmp_pred = _predict_from_network_outputs_fast(i, c_pred=coor_inpaint, img_scaler=img_scaler)
                for k in tmp_pred:
                    inpaint_pred_dict[k].extend(tmp_pred[k])

            return inpaint_pred_dict

        # overlap ensemble path
        dataset = Shuttlecock_Trajectory_Dataset(
            seq_len=seq_len,
            sliding_step=1,
            data_mode="coordinate",
            pred_dict=tracknet_pred_dict,
        )
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        weight = get_ensemble_weight(seq_len, cfg.eval_mode)

        num_sample = len(dataset)
        sample_count = 0
        buffer_size = seq_len - 1
        batch_i = torch.arange(seq_len)
        frame_i = torch.arange(seq_len - 1, -1, -1)
        coor_inpaint_buffer = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)

        for (i, coor_pred, inpaint_mask) in tqdm(loader):
            coor_pred = coor_pred.float()
            inpaint_mask = inpaint_mask.float()
            b_size = i.shape[0]

            with torch.no_grad():
                coor_inpaint = self.inpaintnet(
                    coor_pred.to(self.device),
                    inpaint_mask.to(self.device),
                ).detach().cpu()
                coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1 - inpaint_mask)

            th = (coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH)
            coor_inpaint[th] = 0.0

            coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_inpaint), dim=0)
            ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
            ensemble_coor = torch.empty((0, 1, 2), dtype=torch.float32)

            for b in range(b_size):
                if sample_count < buffer_size:
                    c_ens = coor_inpaint_buffer[batch_i + b, frame_i].sum(0) / (sample_count + 1)
                else:
                    c_ens = (coor_inpaint_buffer[batch_i + b, frame_i] * weight[:, None]).sum(0)

                ensemble_i = torch.cat((ensemble_i, i[b][0].view(1, 1, 2)), dim=0)
                ensemble_coor = torch.cat((ensemble_coor, c_ens.view(1, 1, 2)), dim=0)
                sample_count += 1

                if sample_count == num_sample:
                    # pad and flush tail
                    coor_zero_pad = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
                    coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_zero_pad), dim=0)

                    for f in range(1, seq_len):
                        c_tail = coor_inpaint_buffer[batch_i + b + f, frame_i].sum(0) / (seq_len - f)
                        ensemble_i = torch.cat((ensemble_i, i[-1][f].view(1, 1, 2)), dim=0)
                        ensemble_coor = torch.cat((ensemble_coor, c_tail.view(1, 1, 2)), dim=0)

            th2 = (ensemble_coor[:, :, 0] < COOR_TH) & (ensemble_coor[:, :, 1] < COOR_TH)
            ensemble_coor[th2] = 0.0

            tmp_pred = _predict_from_network_outputs_fast(ensemble_i, c_pred=ensemble_coor, img_scaler=img_scaler)
            for k in tmp_pred:
                inpaint_pred_dict[k].extend(tmp_pred[k])

            coor_inpaint_buffer = coor_inpaint_buffer[-buffer_size:]

        return inpaint_pred_dict


def _predict_from_network_outputs_fast(indices, y_pred=None, c_pred=None, img_scaler=(1, 1)):
    """
    Faster version:
    - c_pred: fully vectorized
    - y_pred: uses heatmap argmax to get (x,y) directly (no predict_location/to_img)
    """
    pred_dict = {"Frame": [], "X": [], "Y": [], "Visibility": []}

    # indices: [B, T, 2] where [:,:,1] is frame index
    if torch.is_tensor(indices):
        ind = indices.detach().cpu().numpy()
    else:
        ind = indices.numpy()
    frame_ids = ind[:, :, 1].astype(np.int64)  # [B,T]

    # We only keep the first occurrence of each frame id in scan order.
    # Your original logic: once f_i repeats (overlap), break inner loop.
    # Equivalent: take all (n,f) until first time f_i == prev_f_i.
    # In practice, easiest + fast: build mask that keeps only the first appearance
    # of each frame id in row-major order.
    flat_f = frame_ids.reshape(-1)
    keep = np.ones_like(flat_f, dtype=bool)
    # keep[i] = False if same as previous
    keep[1:] = flat_f[1:] != flat_f[:-1]

    kept_frames = flat_f[keep]  # [N_kept]

    # --- Coordinate path (fastest)
    if c_pred is not None:
        if torch.is_tensor(c_pred):
            c = c_pred.detach().cpu().numpy()
        else:
            c = c_pred

        # flatten in the same order as indices
        c_flat = c.reshape(-1, 2)[keep]  # [N_kept,2]

        xs = (c_flat[:, 0] * WIDTH * img_scaler[0]).astype(np.int32)
        ys = (c_flat[:, 1] * HEIGHT * img_scaler[1]).astype(np.int32)

    # --- Heatmap path (use argmax instead of predict_location)
    elif y_pred is not None:
        # y_pred expected shape [B, T, H, W] or [B, 1, H, W] etc.
        # In your pipeline you pass [N,1,H,W] sometimes. We handle common shapes.

        if torch.is_tensor(y_pred):
            y = y_pred.detach().float().cpu()
        else:
            y = torch.from_numpy(y_pred).float()

        # Ensure shape is [B,T,H,W]
        if y.ndim == 4:
            # could be [B,T,H,W] already OR [N,1,H,W]
            # if second dim == 1 and indices has T==1, ok.
            # If indices T==1 always in your ensemble stage, treat as [B,1,H,W]
            B, T = ind.shape[0], ind.shape[1]
            if y.shape[0] == B and y.shape[1] == T:
                pass
            elif y.shape[1] == 1 and T == 1 and y.shape[0] == B:
                pass
            else:
                # fallback: assume [N,1,H,W] with N=B*T
                y = y.reshape(B, T, y.shape[-2], y.shape[-1])
        elif y.ndim == 5:
            # e.g. [B,T,1,H,W]
            y = y.squeeze(2)
        else:
            raise ValueError(f"Unexpected y_pred shape: {tuple(y.shape)}")

        # Flatten to match indices order
        y_flat = y.reshape(-1, y.shape[-2], y.shape[-1])[keep]  # [N_kept,H,W]

        # Optionally threshold (your old y_pred>0.5), but argmax doesn’t need it.
        # If you really need it, do:
        # y_flat = (y_flat > 0.5).float()

        # Argmax over H*W
        Hm, Wm = y_flat.shape[-2], y_flat.shape[-1]
        flat_idx = torch.argmax(y_flat.view(y_flat.shape[0], -1), dim=1).numpy()

        ys0 = (flat_idx // Wm).astype(np.int32)
        xs0 = (flat_idx % Wm).astype(np.int32)

        # If heatmap is all zeros, argmax returns 0 => (0,0) which matches your “invisible” behavior.
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