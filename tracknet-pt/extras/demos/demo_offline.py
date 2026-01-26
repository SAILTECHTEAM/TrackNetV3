import argparse
import time

from tracknet.pt.inference.config import TrackNetConfig
from tracknet.pt.inference.offline import TrackNetInfer


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run TrackNet offline inference on a video.")
    p.add_argument("video", nargs="?", default="./test_video/1.mp4", help="Input video path")
    p.add_argument("--tracknet-ckpt", default="./ckpts/TrackNet_best.pt")
    p.add_argument("--inpaintnet-ckpt", default="./ckpts/InpaintNet_best.pt")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-mode", default="weight")
    p.add_argument("--large-video", action="store_true", default=True)
    p.add_argument("--max-sample-num", type=int, default=600)
    p.add_argument("--out-csv", default="pred_result/tennis_ball.csv")
    p.add_argument("--out-video-dir", default="./test_video/1_out/")
    p.add_argument("--traj-len", type=int, default=8)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    cfg = TrackNetConfig(
        tracknet_ckpt=args.tracknet_ckpt,
        inpaintnet_ckpt=args.inpaintnet_ckpt,
        batch_size=args.batch_size,
        eval_mode=args.eval_mode,
        large_video=args.large_video,
        max_sample_num=args.max_sample_num,
    )

    t0 = time.time()
    infer = TrackNetInfer(cfg)
    print("init seconds:", time.time() - t0)

    t1 = time.time()
    pred = infer(args.video)
    print("predict seconds:", time.time() - t1)
    infer.save_csv(pred, args.out_csv)
    infer.save_video(args.video, pred, args.out_video_dir, traj_len=args.traj_len)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
