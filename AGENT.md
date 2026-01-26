# AGENT

1. Project info
   - Project: TrackNetV3 — video object tracking model & utilities (inference, export, demos).
   - Key files/folders:
     - tracknetv3/                - Python package (implementation)
     - onnx/                      - exported ONNX models and data
     - demos/                     - demo scripts (demos/demo_offline.py, demo_live.py, demo_online.py)
     - tools/                     - helper scripts (tools/export_*.py, preprocess.py, etc.)
     - ckpts/                     - model checkpoints (ckpts/TrackNet_best.pt)
     - main.py                    - top-level entrypoint
     - pyproject.toml, uv.lock    - project metadata & lockfile

2. Using uv (focus: uv run)
   - Rule: always run work inside uv for reproducible environments.
   - Run a defined task: uv run <task-name>
     - Examples (if tasks are configured):
       - uv run ruff check / uv run ruff format
       - uv run demo_offline
   - If a task isn't defined, run the script inside uv:
     - uv shell
     - python main.py <args>
   - Useful commands:
     - uv add           # install deps
     - uv shell         # open environment shell
