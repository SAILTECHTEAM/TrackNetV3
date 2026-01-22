from tracknetv3.inference import TrackNetInfer, TrackNetConfig
import time


if __name__ == "__main__":
    cfg = TrackNetConfig(
        tracknet_ckpt="./ckpts/TrackNet_best.pt",
        inpaintnet_ckpt="./ckpts/InpaintNet_best.pt",  # or ""
        batch_size=16,
        eval_mode="weight",
        large_video=True,
        max_sample_num=600,
    )

    t0 = time.time()
    infer = TrackNetInfer(cfg)
    print("init seconds:", time.time() - t0)

    t1 = time.time()
    pred = infer("./test_video/1.mp4")
    print("predict seconds:", time.time() - t1)
    infer.save_csv(pred, "pred_result/tennis_ball.csv")
    infer.save_video("./test_video/1.mp4", pred, "./test_video/1_out/", traj_len=8)
    print("done")
