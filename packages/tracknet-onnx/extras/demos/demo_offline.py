import argparse
import os
import time
from typing import Any

import cv2
from tracknet.core.utils.general import write_pred_csv, write_pred_video
from tracknet.onnx.inference.streaming_onnx import StreamingInferenceONNX


def results_to_pred_dict(results: list[dict[str, Any]], total_frames: int, img_scaler, img_shape):
    xs = [0] * total_frames
    ys = [0] * total_frames
    vs = [0] * total_frames
    for r in results:
        f = int(r["Frame"])
        if 0 <= f < total_frames:
            xs[f] = int(r["X"])
            ys[f] = int(r["Y"])
            vs[f] = int(r["Visibility"])
    return {
        "Frame": list(range(total_frames)),
        "X": xs,
        "Y": ys,
        "Visibility": vs,
        "Inpaint_Mask": [0] * total_frames,
        "Img_scaler": img_scaler,
        "Img_shape": img_shape,
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run TrackNet ONNX offline inference on a video.")
    p.add_argument("video", nargs="?", default="./test_video/1.mp4", help="Input video path")
    p.add_argument("--model", required=True, help="Path to TrackNet ONNX model")
    p.add_argument("--seq-len", type=int, default=8, help="Sequence length for TrackNet")
    p.add_argument(
        "--bg-mode", default="", help="Background mode (subtract, concat, subtract_concat)"
    )
    p.add_argument(
        "--eval-mode",
        default="weight",
        choices=["weight", "average", "nonoverlap"],
        help="Evaluation mode",
    )
    p.add_argument(
        "--median-warmup", type=int, default=0, help="Number of frames for median warmup"
    )
    p.add_argument("--out-csv", default="pred_result/tennis_ball_onnx.csv")
    p.add_argument("--out-video", default="pred_result/tennis_ball_onnx.mp4")
    p.add_argument("--traj-len", type=int, default=8)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {args.video}")
        return 1

    _total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    infer = StreamingInferenceONNX(
        model_path=args.model,
        seq_len=args.seq_len,
        bg_mode=args.bg_mode,
        eval_mode=args.eval_mode,
        median_warmup=args.median_warmup,
    )

    results = []
    fid = 0
    start_time = time.perf_counter()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        out = infer.push(frame_bgr, frame_id=fid)
        if out is not None:
            results.append(out)
        fid += 1

    results.extend(infer.flush())
    cap.release()

    end_time = time.perf_counter()
    duration = end_time - start_time
    print(
        f"[INFO] Video processing time: {duration:.3f}s for {fid} frames ({fid / max(1e-6, duration):.2f} FPS)"
    )

    print("\n=== Timing (excluding cap.read) ===")
    s = infer.stats
    print(f"[TrackNet ONNX] calls={s['push_calls']} outputs={s['outputs']}")
    print(
        f"  preprocess: {s['t_preprocess']:.3f}s  ({s['t_preprocess'] / max(1, s['push_calls']):.6f}s/call)"
    )
    print(
        f"  forward:    {s['t_forward']:.3f}s  ({s['t_forward'] / max(1, s['push_calls']):.6f}s/call)"
    )
    print(f"  post:       {s['t_post']:.3f}s  ({s['t_post'] / max(1, s['push_calls']):.6f}s/call)")
    print(f"  median:     {s['t_median']:.3f}s")

    pred_dict = results_to_pred_dict(results, fid, infer.img_scaler, infer.img_shape)

    if os.path.dirname(args.out_csv):
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    write_pred_csv(pred_dict, args.out_csv)

    if os.path.dirname(args.out_video):
        os.makedirs(os.path.dirname(args.out_video), exist_ok=True)
    write_pred_video(args.video, pred_dict, args.out_video, traj_len=args.traj_len)

    print(f"[DONE] Total frames={fid}, outputs={len(results)}")
    print(f"[SAVED] {args.out_csv} and {args.out_video}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
