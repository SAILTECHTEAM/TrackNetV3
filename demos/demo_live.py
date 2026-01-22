import os
import cv2
import time
import argparse
from collections import deque

from tracknetv3.inference import (
    TrackNetInfer,
    TrackNetConfig,
    TrackNetModule,
    InpaintModule,
)


def open_writer_mp4(path, w, h, fps):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if fps < 1:
        fps = 30.0

    for c in ["mp4v", "avc1", "H264"]:
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*c), fps, (w, h))
        if vw.isOpened():
            return vw
    raise RuntimeError(f"Cannot open VideoWriter: {path}")


def draw_trail(frame, trail, traj_len):
    if not trail:
        return
    for p in trail[-traj_len:]:
        cv2.circle(frame, p, 3, (0, 255, 0), -1)
    cv2.circle(frame, trail[-1], 5, (0, 0, 255), -1)


if __name__ == "__main__":
    # -------------------------------
    # Arguments (ONLY here)
    # -------------------------------
    parser = argparse.ArgumentParser("TrackNet Live Demo")
    parser.add_argument(
        "--src", required=True, help="RTSP URL / video file / webcam index (0)"
    )
    parser.add_argument("--out_dir", default="pred_result")
    parser.add_argument("--csv", default="tracknet_live.csv")
    parser.add_argument("--mp4", default="tracknet_live.mp4")
    parser.add_argument("--traj_len", type=int, default=8)
    parser.add_argument("--median_warmup", type=int, default=50)
    parser.add_argument("--fps_fallback", type=float, default=30.0)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--drop_if_lagging", action="store_true")
    args = parser.parse_args()

    # -------------------------------
    # Model load
    # -------------------------------
    cfg = TrackNetConfig(
        tracknet_ckpt="./ckpts/TrackNet_best.pt",
        inpaintnet_ckpt="./ckpts/InpaintNet_best.pt",
        eval_mode="weight",
        batch_size=16,
        large_video=True,
    )
    infer = TrackNetInfer(cfg)

    # -------------------------------
    # Open source (RTSP / cam / file)
    # -------------------------------
    src = int(args.src) if args.src.isdigit() else args.src
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.src}")

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps < 1:
        fps = args.fps_fallback

    # -------------------------------
    # Streaming modules
    # -------------------------------
    median_warmup = args.median_warmup if infer.bg_mode else 0
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
    trail = deque(maxlen=max(1, args.traj_len))

    # -------------------------------
    # Outputs
    # -------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, args.csv)
    mp4_path = os.path.join(args.out_dir, args.mp4)

    csv_f = open(csv_path, "w")
    csv_f.write("Frame,X,Y,Visibility\n")

    vw = None

    # -------------------------------
    # Main loop
    # -------------------------------
    t0 = time.perf_counter()
    fid = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        out_t = track_mod.push(frame, frame_id=fid)

        if inpaint_mod is None and infer.inpaintnet and track_mod.img_scaler:
            inpaint_mod = InpaintModule(
                inpaintnet=infer.inpaintnet,
                seq_len=infer.inpaintnet_seq_len,
                device=infer.device,
                img_scaler=track_mod.img_scaler,
            )

        out = None
        if out_t:
            out = out_t
            if inpaint_mod:
                out_i = inpaint_mod.push(out_t)
                if out_i:
                    out = out_i

        if out:
            if out["Visibility"] == 1:
                trail.append((out["X"], out["Y"]))
            csv_f.write(f"{out['Frame']},{out['X']},{out['Y']},{out['Visibility']}\n")

        if vw is None:
            h, w = frame.shape[:2]
            vw = open_writer_mp4(mp4_path, w, h, fps)

        draw_trail(frame, trail, args.traj_len)
        vw.write(frame)

        if args.show:
            cv2.imshow("TrackNet Live", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        fid += 1
        if args.max_frames and fid >= args.max_frames:
            break

    # -------------------------------
    # Cleanup & flush
    # -------------------------------
    cap.release()
    vw.release()
    csv_f.close()
    cv2.destroyAllWindows()

    dt = time.perf_counter() - t0
    print(f"\n[DONE] frames={fid}  time={dt:.2f}s  FPS={fid / dt:.2f}")

    print("\n=== Module timing (excluding cap.read) ===")
    s = track_mod.stats
    print(f"[TrackNet] calls={s['push_calls']} outputs={s['outputs']}")
    print(f" preprocess: {s['t_preprocess']:.3f}s")
    print(f" forward:    {s['t_forward']:.3f}s")
    print(f" post:       {s['t_post']:.3f}s")
    print(f" median:     {s['t_median']:.3f}s")

    if inpaint_mod:
        si = inpaint_mod.stats
        print(f"[Inpaint] calls={si['push_calls']} outputs={si['outputs']}")
        print(f" forward: {si['t_forward']:.3f}s")
        print(f" post:    {si['t_post']:.3f}s")

    print(f"\n[SAVED] {csv_path}")
    print(f"[SAVED] {mp4_path}")
