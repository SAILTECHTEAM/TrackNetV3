from collections import deque

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import IterableDataset

from tracknet.core.config.constants import HEIGHT, WIDTH


class Video_IterableDataset(IterableDataset):
    """Dataset for inference especially for large video."""

    def __init__(
        self,
        video_file,
        seq_len=8,
        sliding_step=1,
        bg_mode="",
        HEIGHT=HEIGHT,
        WIDTH=WIDTH,
        max_sample_num=1800,
        video_range=None,
        median=None,
        fps_fallback=30.0,  # NEW
        verbose_timing=True,  # NEW
    ):
        # Image size
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH

        self.video_file = video_file
        self.cap = cv2.VideoCapture(self.video_file)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_file}")

        self.video_len = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps < 1.0:
            # OpenCV sometimes returns 0 for fps (common). Use fallback.
            fps = float(fps_fallback)
        self.fps = fps  # keep as float for precise frame calc

        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.w_scaler, self.h_scaler = self.w / self.WIDTH, self.h / self.HEIGHT

        self.seq_len = seq_len
        self.sliding_step = sliding_step
        self.bg_mode = bg_mode

        self._verbose_timing = bool(verbose_timing)
        self._fps_fallback = float(fps_fallback)

        # Median/background image
        self.median = None
        if self.bg_mode:
            # If median passed in, reuse it (supports caching in TrackNetInfer)
            if median is not None:
                self.median = median
            else:
                self.median = self.__gen_median__(max_sample_num, video_range)

    def __iter__(self):
        """Return the data sequentially (fast + correct EOF behavior)."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        seq_len = self.seq_len
        step = self.sliding_step
        window = deque(maxlen=seq_len)

        start_f_id = 0
        end_f_id = 0  # next frame id to be read
        eof = False

        # Fill initial window
        while len(window) < seq_len:
            ok, frame = self.cap.read()
            if not ok:
                eof = True
                break
            window.append(frame)
            end_f_id += 1

        if len(window) == 0:
            self.cap.release()
            return

        while True:
            # Build indices for current window
            real_n = len(window)

            # If we hit EOF and window is short, pad ONCE (like original)
            padded_this_round = False
            if eof and real_n < seq_len:
                last = window[-1]
                while len(window) < seq_len:
                    window.append(last)
                padded_this_round = True
                real_n = seq_len

            # data_idx: (seq_len, 2), first col zeros, second col frame ids
            # end_f_id is "one past last read frame index"
            cur_start = start_f_id
            cur_end = min(start_f_id + seq_len, end_f_id)

            ids = np.arange(cur_start, cur_end, dtype=np.int64)
            if ids.size < seq_len:
                # pad indices with last valid id
                last_id = max(0, end_f_id - 1)
                ids = np.concatenate([ids, np.full((seq_len - ids.size,), last_id, dtype=np.int64)])

            data_idx = np.stack([np.zeros(seq_len, dtype=np.int64), ids], axis=1)

            # Frames: deque -> numpy, then BGR->RGB view
            frame_arr = np.asarray(list(window), dtype=np.uint8)[..., ::-1]
            frames = self.__process__(frame_arr)
            yield data_idx, frames

            # If this was the padded tail, stop (original yields tail once then exits)
            if padded_this_round:
                break

            # Advance window by 'step'
            pop_n = min(step, len(window))
            for _ in range(pop_n):
                window.popleft()
            start_f_id += step

            # Refill window
            while len(window) < seq_len:
                ok, frame = self.cap.read()
                if not ok:
                    eof = True
                    break
                window.append(frame)
                end_f_id += 1

            # If EOF and nothing left (shouldn't happen usually), stop
            if eof and len(window) == 0:
                break

        self.cap.release()

    def __gen_median__(self, max_sample_num, video_range):
        import time

        t0 = time.time()
        print("Generate median image (grab/retrieve)...")

        if video_range is None:
            start_frame, end_frame = 0, self.video_len
        else:
            start_frame = max(0, int(video_range[0] * self.fps))
            end_frame = min(int(video_range[1] * self.fps), self.video_len)

        if end_frame <= start_frame:
            raise ValueError("Invalid video_range")

        video_seg_len = end_frame - start_frame
        stride = max(1, video_seg_len // max_sample_num)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        f = start_frame

        while f < end_frame and len(frames) < max_sample_num:
            # Skip stride-1 frames cheaply
            for _ in range(stride - 1):
                if f >= end_frame:
                    break
                if not self.cap.grab():
                    break
                f += 1

            if f >= end_frame:
                break

            # Grab + retrieve sampled frame
            if not self.cap.grab():
                break
            ok, frame = self.cap.retrieve()
            if not ok:
                break
            frames.append(frame)
            f += 1

        if len(frames) == 0:
            raise RuntimeError("Failed to sample frames for median image.")

        median = np.median(frames, axis=0)[..., ::-1]

        if self.bg_mode == "concat":
            median_img = Image.fromarray(median.astype("uint8"))
            median_img = np.array(median_img.resize(size=(self.WIDTH, self.HEIGHT)))
            median = np.moveaxis(median_img, -1, 0)

        dt = time.time() - t0
        print("Median image generated.")
        if getattr(self, "_verbose_timing", True):
            print(
                f"[TIMER] median={dt:.2f}s | sampled={len(frames)} | stride={stride} "
                f"| frames=({start_frame}->{min(end_frame, f)}) | fps={self.fps:.2f}"
            )
        return median

    def __process__(self, imgs):
        """Fast processing: cv2.resize + preallocation (no PIL, no concat-in-loop)."""
        H, W = self.HEIGHT, self.WIDTH
        T = self.seq_len
        mode = self.bg_mode

        # Determine channels per frame
        if mode == "subtract":
            c_per = 1
        elif mode == "subtract_concat":
            c_per = 4  # 3 RGB + 1 diff
        else:
            c_per = 3  # RGB

        # For concat mode, prepend median channels (3) before the sequence
        extra_c = 3 if mode == "concat" else 0
        out = np.empty((extra_c + T * c_per, H, W), dtype=np.float32)

        # Prepare median if needed (median is RGB)
        median = self.median if mode else None

        # If concat mode: median already prepared in __gen_median__ as CHW float/uint8?
        # In your code: for bg_mode=='concat', median is stored as CHW uint8 resized already.
        # So just copy it into output.
        write_offset = 0
        if mode == "concat":
            # ensure float32
            out[0:3] = median.astype(np.float32)
            write_offset = 3

        for i in range(T):
            img = imgs[i]  # RGB uint8, shape (h0,w0,3)

            if mode == "subtract":
                # diff = sum(|img - median|) over channels -> uint8 gray
                # use int16 to avoid underflow
                diff = np.abs(img.astype(np.int16) - median.astype(np.int16)).sum(axis=2)
                diff = np.clip(diff, 0, 255).astype(np.uint8)

                diff_r = cv2.resize(diff, (W, H), interpolation=cv2.INTER_AREA)
                out[write_offset + i, :, :] = diff_r.astype(np.float32)

            elif mode == "subtract_concat":
                diff = np.abs(img.astype(np.int16) - median.astype(np.int16)).sum(axis=2)
                diff = np.clip(diff, 0, 255).astype(np.uint8)

                img_r = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)  # RGB
                diff_r = cv2.resize(diff, (W, H), interpolation=cv2.INTER_AREA)

                base = write_offset + i * 4
                # CHW
                out[base + 0] = img_r[:, :, 0].astype(np.float32)
                out[base + 1] = img_r[:, :, 1].astype(np.float32)
                out[base + 2] = img_r[:, :, 2].astype(np.float32)
                out[base + 3] = diff_r.astype(np.float32)

            else:
                img_r = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                base = write_offset + i * 3
                out[base + 0] = img_r[:, :, 0].astype(np.float32)
                out[base + 1] = img_r[:, :, 1].astype(np.float32)
                out[base + 2] = img_r[:, :, 2].astype(np.float32)

        # Normalize in-place
        out *= 1.0 / 255.0
        return out
