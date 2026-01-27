import numpy as np
import onnxruntime as ort
from tracknet.core.utils.trajectory import generate_inpaint_mask


class InpaintInference:
    def __init__(self, model_path: str, seq_len: int = 16, batch_size: int = 4):
        self.sess = ort.InferenceSession(model_path)
        self.seq_len = seq_len
        self.batch_size = batch_size

    def __call__(self, pred_dict: dict):
        return self.predict(pred_dict)

    def predict(self, pred_dict: dict):
        # Generate inpaint mask
        w, h = pred_dict["Img_shape"]
        pred_dict["Inpaint_Mask"] = generate_inpaint_mask(pred_dict, th_h=h * 0.05)

        inpaint_mask = np.array(pred_dict["Inpaint_Mask"])
        if not np.any(inpaint_mask):
            return pred_dict

        xs = np.array(pred_dict["X"], dtype=np.float32)
        ys = np.array(pred_dict["Y"], dtype=np.float32)
        vs = np.array(pred_dict["Visibility"], dtype=np.float32)

        # Normalization
        xs_norm = xs / w
        ys_norm = ys / h
        coords = np.stack([xs_norm, ys_norm], axis=-1)  # (N, 2)
        mask = inpaint_mask.astype(np.float32).reshape(-1, 1)  # (N, 1)

        total_frames = len(xs)
        inpainted_coords = coords.copy()

        # Sliding window inference
        # We need to handle batching and padding if necessary
        # The model expects [batch_size, seq_len, 2] and [batch_size, seq_len, 1]

        # Simple non-overlapping or overlapping windowing with ensemble?
        # PT demo uses both non-overlap and ensemble (weight).
        # For simplicity and to match the model's fixed batch size of 4,
        # let's implement a straightforward windowed inference.

        # We'll use a sliding window of seq_len with step 1 and average the results,
        # similar to how PT's ensemble mode works.

        padded_coords = np.pad(coords, ((self.seq_len - 1, self.seq_len - 1), (0, 0)), mode="edge")
        padded_mask = np.pad(mask, ((self.seq_len - 1, self.seq_len - 1), (0, 0)), mode="edge")

        num_windows = total_frames + self.seq_len - 1
        acc_coords = np.zeros((total_frames, 2), dtype=np.float32)
        acc_counts = np.zeros((total_frames, 1), dtype=np.float32)

        windows_coords = []
        windows_mask = []
        window_indices = []

        for i in range(num_windows):
            win_c = padded_coords[i : i + self.seq_len]
            win_m = padded_mask[i : i + self.seq_len]
            windows_coords.append(win_c)
            windows_mask.append(win_m)
            window_indices.append(i)

            if len(windows_coords) == self.batch_size or i == num_windows - 1:
                # Pad batch if needed
                actual_batch_size = len(windows_coords)
                if actual_batch_size < self.batch_size:
                    for _ in range(self.batch_size - actual_batch_size):
                        windows_coords.append(np.zeros((self.seq_len, 2), dtype=np.float32))
                        windows_mask.append(np.zeros((self.seq_len, 1), dtype=np.float32))

                batch_c = np.stack(windows_coords, axis=0)
                batch_m = np.stack(windows_mask, axis=0)

                outputs = self.sess.run(None, {"coords": batch_c, "mask": batch_m})
                batch_out = outputs[0]  # (batch_size, seq_len, 2)

                for b in range(actual_batch_size):
                    win_idx = window_indices[b]
                    for t in range(self.seq_len):
                        frame_idx = win_idx + t - (self.seq_len - 1)
                        if 0 <= frame_idx < total_frames:
                            acc_coords[frame_idx] += batch_out[b, t]
                            acc_counts[frame_idx] += 1

                windows_coords = []
                windows_mask = []
                window_indices = []

        inpainted_coords_norm = acc_coords / np.maximum(acc_counts, 1)

        # Blend: inpainted = inpaint_output * mask + original * (1 - mask)
        final_coords_norm = inpainted_coords_norm * mask + coords * (1 - mask)

        # Denormalization
        final_xs = (final_coords_norm[:, 0] * w).astype(int)
        final_ys = (final_coords_norm[:, 1] * h).astype(int)

        # Update visibility: if it was inpainted, it should be visible now?
        # PT demo: coor_inpaint[th] = 0.0 where COOR_TH is 0.01.
        # It doesn't explicitly change Visibility to 1, but usually inpainting is to find missing balls.
        # In tracknet-pt/tracknet/pt/inference/offline.py:472
        # It uses _predict_from_network_outputs_fast which sets visibility.
        # Let's see how it determines visibility from coordinates.

        # In PT, visibility is 1 if x > 0 and y > 0 after denorm.
        new_vs = vs.copy()
        for i in range(total_frames):
            if inpaint_mask[i] == 1:
                if final_xs[i] > 0 and final_ys[i] > 0:
                    new_vs[i] = 1
                else:
                    new_vs[i] = 0

        return {
            "Frame": pred_dict["Frame"],
            "X": final_xs.tolist(),
            "Y": final_ys.tolist(),
            "Visibility": new_vs.astype(int).tolist(),
            "Inpaint_Mask": inpaint_mask.tolist(),
            "Img_scaler": pred_dict["Img_scaler"],
            "Img_shape": pred_dict["Img_shape"],
        }
