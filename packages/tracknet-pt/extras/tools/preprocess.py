import argparse
import os
import shutil

import parse
from tracknet.core.utils.general import (
    generate_data_frames,
    get_match_median,
    get_num_frames,
    list_dirs,
)
from tracknet.core.utils.visualize import plot_median_files
from tracknet.pt.datasets.shuttlecock import data_dir


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prepare TrackNet dataset: generate frames, medians, and validation split."
    )
    p.add_argument(
        "--data-dir",
        default=data_dir,
        help="Dataset root directory (default: tracknet.pt.datasets.shuttlecock.data_dir)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    root_dir = args.data_dir

    # Replace csv to corrected csv in test set
    if os.path.exists("corrected_test_label"):
        match_dirs = list_dirs(os.path.join(root_dir, "test"))
        match_dirs = sorted(match_dirs, key=lambda s: int(s.split("match")[-1]))
        for match_dir in match_dirs:
            file_format_str = os.path.join("{}", "test", "{}")
            _, match_dir = parse.parse(file_format_str, match_dir)
            if not os.path.exists(os.path.join(root_dir, "test", match_dir, "corrected_csv")):
                shutil.copytree(
                    os.path.join("corrected_test_label", match_dir, "corrected_csv"),
                    os.path.join(root_dir, "test", match_dir, "corrected_csv"),
                )
                shutil.copy(
                    os.path.join("corrected_test_label", "drop_frame.json"),
                    os.path.join(root_dir, "drop_frame.json"),
                )

    # Generate frames from videos
    for split in ["train", "test"]:
        split_frame_count = 0
        match_dirs = list_dirs(os.path.join(root_dir, split))
        for match_dir in match_dirs:
            match_frame_count = 0
            file_format_str = os.path.join("{}", "match{}")
            _, match_id = parse.parse(file_format_str, match_dir)
            video_files = list_dirs(os.path.join(match_dir, "video"))
            for video_file in video_files:
                generate_data_frames(video_file)
                file_format_str = os.path.join("{}", "video", "{}.mp4")
                _, video_name = parse.parse(file_format_str, video_file)
                rally_dir = os.path.join(match_dir, "frame", video_name)
                video_frame_count = get_num_frames(rally_dir)
                print(
                    f"[{split} / match{match_id} / {video_name}]\tvideo frames: {video_frame_count}"
                )
                match_frame_count += video_frame_count
            get_match_median(match_dir)
            print(f"[{split} / match{match_id}]:\ttotal frames: {match_frame_count}")
            split_frame_count += match_frame_count

        print(f"[{split}]:\ttotal frames: {split_frame_count}")

    # Form validation set
    if not os.path.exists(os.path.join(root_dir, "val")):
        match_dirs = list_dirs(os.path.join(root_dir, "train"))
        match_dirs = sorted(match_dirs, key=lambda s: int(s.split("match")[-1]))
        for match_dir in match_dirs:
            # Pick last rally in each match as validation set
            video_files = list_dirs(os.path.join(match_dir, "video"))
            file_format_str = os.path.join("{}", "train", "{}", "video", "{}.mp4")
            _, match_dir, rally_id = parse.parse(file_format_str, video_files[-1])
            os.makedirs(os.path.join(root_dir, "val", match_dir, "csv"), exist_ok=True)
            os.makedirs(os.path.join(root_dir, "val", match_dir, "video"), exist_ok=True)
            shutil.move(
                os.path.join(root_dir, "train", match_dir, "csv", f"{rally_id}_ball.csv"),
                os.path.join(root_dir, "val", match_dir, "csv", f"{rally_id}_ball.csv"),
            )
            shutil.move(
                os.path.join(root_dir, "train", match_dir, "video", f"{rally_id}.mp4"),
                os.path.join(root_dir, "val", match_dir, "video", f"{rally_id}.mp4"),
            )
            shutil.move(
                os.path.join(root_dir, "train", match_dir, "frame", rally_id),
                os.path.join(root_dir, "val", match_dir, "frame", rally_id),
            )
            shutil.copy(
                os.path.join(root_dir, "train", match_dir, "median.npz"),
                os.path.join(root_dir, "val", match_dir, "median.npz"),
            )

    # Plot median frames, save at <data_dir>/median
    plot_median_files(root_dir)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
