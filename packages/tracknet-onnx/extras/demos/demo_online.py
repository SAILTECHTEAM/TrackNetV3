import argparse
import os
import time
from collections import deque
from typing import Any

import cv2
from tracknet.core.utils.general import draw_traj, write_pred_csv, write_pred_video
from tracknet.onnx.inference import InpaintModule, TrackNetModule


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
    p = argparse.ArgumentParser(
        description="Run TrackNet ONNX online/streaming inference on a video."
    )
    p.add_argument("video", nargs="?", default="./test_video/1.mp4", help="Input video path")
    p.add_argument("--model", required=True, help="Path to TrackNet ONNX model")
    p.add_argument("--inpaint", help="Path to InpaintNet ONNX model")
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
    p.add_argument("--show", action="store_true", help="Display the results in real-time")
    p.add_argument("--out-csv", default=None, help="Output CSV path")
    p.add_argument("--out-video", default=None, help="Output video path")
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

    infer = TrackNetModule(
        model_path=args.model,
        seq_len=args.seq_len,
        bg_mode=args.bg_mode,
        eval_mode=args.eval_mode,
        median_warmup=args.median_warmup,
    )

    inpaint_mod = None
    if args.inpaint:
        print(f"[INFO] Using InpaintNet: {args.inpaint}")
        inpaint_mod = InpaintModule(args.inpaint)

    results = []
    fid = 0
    traj = deque(maxlen=args.traj_len)

    # We need to keep track of frames for display because of the pipeline delay (seq_len)
    frame_buffer = {}

    print("Starting online inference... Press 'q' to quit if --show is enabled.")
    start_time = time.perf_counter()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if args.show or args.out_video:
            frame_buffer[fid] = frame_bgr.copy()

        out = infer.push(frame_bgr, frame_id=fid)

        if out is not None:
            if inpaint_mod is not None:
                out = inpaint_mod.push(out, img_scaler=infer.img_scaler)

        if out is not None:
            results.append(out)
            ofid = out["Frame"]
            ox, oy, ov = out["X"], out["Y"], out["Visibility"]

            # Print detection
            if ov:
                print(f"Frame {ofid}: ({ox}, {oy})")

            if args.show:
                # Handle visualization of the delayed output
                if ofid in frame_buffer:
                    disp_frame = frame_buffer[ofid]
                    traj.appendleft([ox, oy] if ov else None)
                    vis_frame = draw_traj(disp_frame, traj, color="yellow")
                    cv2.imshow("TrackNet ONNX Online Demo", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    # Clean up old frames from buffer
                    for k in list(frame_buffer.keys()):
                        if k <= ofid:
                            del frame_buffer[k]

        fid += 1

    print("\nFlushing remaining frames...")
    flush_results = infer.flush()
    if inpaint_mod is not None:
        # For each flushed track result, push to inpaint if possible
        # but wait, infer.flush() returns a list of results.
        # PT's InpaintModule.flush() is different.
        # In this ONNX InpaintModule, push() takes one pred.
        inpainted_flush = []
        for res in flush_results:
            out_i = inpaint_mod.push(res)
            if out_i is not None:
                inpainted_flush.append(out_i)
        inpainted_flush.extend(inpaint_mod.flush())
        flush_results = inpainted_flush

    results.extend(flush_results)

    for out in flush_results:
        ofid = out["Frame"]
        ox, oy, ov = out["X"], out["Y"], out["Visibility"]
        if ov:
            print(f"Frame {ofid}: ({ox}, {oy})")

        if args.show and ofid in frame_buffer:
            disp_frame = frame_buffer[ofid]
            traj.appendleft([ox, oy] if ov else None)
            vis_frame = draw_traj(disp_frame, traj, color="yellow")
            cv2.imshow("TrackNet ONNX Online Demo", vis_frame)
            cv2.waitKey(1)

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    end_time = time.perf_counter()
    duration = end_time - start_time
    print(
        f"\n[INFO] Video processing time: {duration:.3f}s for {fid} frames ({fid / max(1e-6, duration):.2f} FPS)"
    )

    print("\n=== Timing ===")
    s = infer.stats
    print(f"[TrackNet ONNX] calls={s['push_calls']} outputs={s['outputs']}")
    print(f"  preprocess: {s['t_preprocess']:.3f}s")
    print(f"  forward:    {s['t_forward']:.3f}s")
    print(f"  post:       {s['t_post']:.3f}s")
    print(f"  median:     {s['t_median']:.3f}s")

    # Optional saving
    if args.out_csv or args.out_video:
        pred_dict = results_to_pred_dict(results, fid, infer.img_scaler, infer.img_shape)

        if args.out_csv:
            if os.path.dirname(args.out_csv):
                os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
            write_pred_csv(pred_dict, args.out_csv)
            print(f"[SAVED] {args.out_csv}")

        if args.out_video:
            if os.path.dirname(args.out_video):
                os.makedirs(os.path.dirname(args.out_video), exist_ok=True)
            write_pred_video(args.video, pred_dict, args.out_video, traj_len=args.traj_len)
            print(f"[SAVED] {args.out_video}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
