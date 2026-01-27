import argparse
import os
import time
from typing import List

import cv2
from tracknet.pt.inference.config import TrackNetConfig
from tracknet.pt.inference.offline import TrackNetInfer
from tracknet.pt.inference.streaming import InpaintModule, TrackNetModule


# (no self.stats here)


def results_to_pred_dict(results, total_frames, img_scaler, img_shape):
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
        "Inpaint_Mask": [],
        "Img_scaler": img_scaler,
        "Img_shape": img_shape,
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run TrackNet online (streaming) inference on a video.")
    p.add_argument("video", nargs="?", default="./test_video/1.mp4", help="Input video path")
    p.add_argument("--tracknet-ckpt", default="./ckpts/TrackNet_best.pt")
    p.add_argument("--inpaintnet-ckpt", default="./ckpts/InpaintNet_best.pt")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-mode", default="weight")
    p.add_argument("--large-video", action="store_true", default=True)
    p.add_argument("--max-sample-num", type=int, default=600)
    p.add_argument("--out-csv", default="pred_result/tennis_ball_stream.csv")
    p.add_argument("--out-video-dir", default="pred_result/")
    p.add_argument("--traj-len", type=int, default=8)
    return p


def main(argv: List[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    cfg = TrackNetConfig(
        tracknet_ckpt=args.tracknet_ckpt,
        inpaintnet_ckpt=args.inpaintnet_ckpt,
        eval_mode=args.eval_mode,
        batch_size=args.batch_size,
        large_video=args.large_video,
        max_sample_num=args.max_sample_num,
    )

    infer = TrackNetInfer(cfg)

    video_path = args.video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    median_warmup = 50 if infer.bg_mode else 0

    track_mod = TrackNetModule(
        tracknet=infer.tracknet,
        seq_len=infer.tracknet_seq_len,
        bg_mode=infer.bg_mode,
        device=infer.device,
        eval_mode=cfg.eval_mode,
        median=None,
        median_warmup=median_warmup,
    )

    inpaint_mod = None
    results = []
    fid = 0

    start = time.perf_counter()
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        out_t = track_mod.push(frame_bgr, frame_id=fid)

        if (
            inpaint_mod is None
            and infer.inpaintnet is not None
            and track_mod.img_scaler is not None
        ):
            inpaint_mod = InpaintModule(
                inpaintnet=infer.inpaintnet,
                seq_len=int(infer.inpaintnet_seq_len),
                device=infer.device,
                img_scaler=track_mod.img_scaler,
            )

        if out_t is not None:
            if inpaint_mod is not None:
                out_i = inpaint_mod.push(out_t)
                if out_i is not None:
                    results.append(out_i)
            else:
                results.append(out_t)

        fid += 1

    cap.release()
    end = time.perf_counter()
    print(
        f"[INFO] Video processing time: {end - start:.3f}s for {fid} frames ({fid / (end - start):.2f} FPS)"
    )

    results.extend(track_mod.flush())
    if inpaint_mod is not None:
        results.extend(inpaint_mod.flush())

    print(
        f"[DONE] total_frames={fid}, outputs={len(results)} (bg_mode={infer.bg_mode}, median_warmup={median_warmup})"
    )

    pred_dict = results_to_pred_dict(results, fid, track_mod.img_scaler, track_mod.img_shape)

    print("\n=== Timing (excluding cap.read) ===")
    s = track_mod.stats
    print(f"[TrackNet] calls={s['push_calls']} outputs={s['outputs']}")
    print(
        f"  preprocess: {s['t_preprocess']:.3f}s  ({s['t_preprocess'] / max(1, s['push_calls']):.6f}s/call)"
    )
    print(
        f"  forward:    {s['t_forward']:.3f}s  ({s['t_forward'] / max(1, s['push_calls']):.6f}s/call)"
    )
    print(f"  post:       {s['t_post']:.3f}s  ({s['t_post'] / max(1, s['push_calls']):.6f}s/call)")
    print(f"  median:     {s['t_median']:.3f}s")

    if inpaint_mod is not None:
        si = inpaint_mod.stats
        print(f"[Inpaint] calls={si['push_calls']} outputs={si['outputs']}")
        print(
            f"  forward:    {si['t_forward']:.3f}s  ({si['t_forward'] / max(1, si['push_calls']):.6f}s/call)"
        )
        print(
            f"  post:       {si['t_post']:.3f}s  ({si['t_post'] / max(1, si['push_calls']):.6f}s/call)"
        )

    os.makedirs(os.path.dirname(args.out_csv) or "pred_result", exist_ok=True)
    infer.save_csv(pred_dict, args.out_csv)
    infer.save_video(video_path, pred_dict, args.out_video_dir, traj_len=args.traj_len)
    print(f"[SAVED] {args.out_csv} and video(s) to {args.out_video_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
