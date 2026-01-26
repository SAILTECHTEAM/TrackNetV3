# TrackNetV3 Modularization Plan - Draft

## User Requirements

### Primary Objective
Separate the current TrackNetV3 project into independent modules to allow selective installation and reduce dependency overhead for different use cases.

### Target Modules
1. **tracknet-core**: Common utilities with minimal dependencies
2. **tracknet-onnx**: ONNX Runtime-based inference only
3. **tracknet-pt**: PyTorch-based inference and training (merged from pt-inference + training)

### Key Constraints
- **Naming**: Use `tracknet-*` prefix for all modules
- **Publishing**: No PyPI publishing required
- **Backward Compatibility**: Not required - can break existing import paths
- **Demos & Tools**: Include as extras, not separate modules
- **Dependencies**: ONNX module must NOT require PyTorch

### User's Uncertainties
- **Core Module Scope**: User indicated uncertainty about what should go into core - needs recommendation based on exploration

## Context & Investigation Findings

### Current Project Structure

```
TrackNetV3/
├── tracknetv3/                    # Main package
│   ├── config/
│   │   └── constants.py          # Image dims, thresholds, formats
│   ├── models/
│   │   ├── tracknet.py           # TrackNet PyTorch model
│   │   ├── inpaintnet.py         # InpaintNet PyTorch model
│   │   └── blocks.py             # Model building blocks
│   ├── inference/
│   │   ├── streaming.py          # PyTorch streaming
│   │   ├── streaming_onnx.py     # ONNX Runtime streaming
│   │   ├── offline.py            # Batch inference (PyTorch)
│   │   └── helpers.py            # Inference utilities
│   ├── datasets/
│   │   ├── shuttlecock.py        # Training dataset
│   │   └── video_iterable.py     # Video dataset
│   ├── evaluation/
│   │   ├── ensemble.py           # Temporal ensemble
│   │   ├── metrics.py            # Evaluation metrics
│   │   └── predict.py            # Prediction utilities
│   └── utils/
│       ├── general.py            # Video/image utilities
│       ├── trajectory.py         # Trajectory processing
│       └── metric.py             # Metric functions
├── scripts/
│   ├── train.py                  # Training script
│   └── test.py                   # Testing script
├── tools/                        # Utility tools
│   ├── export_tracknet_onnx.py
│   ├── export_inpaintnet_onnx.py
│   ├── preprocess.py
│   ├── error_analysis.py
│   ├── correct_label.py
│   └── generate_mask_data.py
├── demos/
│   ├── demo_offline.py
│   ├── demo_online.py
│   └── demo_live.py
└── pyproject.toml                # Single dependency file
```

### Current Dependencies (from pyproject.toml)

**Heavy ML Libraries:**
- `torch>=2.0.0` - Required for all current uses
- `onnxruntime-gpu>=1.19.0` - Required for ONNX inference

**Lightweight Libraries (Potential Core Dependencies):**
- `numpy>=1.21.0`
- `opencv-python>=4.5.0`
- `pillow>=8.0.0`
- `pandas>=1.3.0`
- `parse>=1.0.0`
- `scipy>=1.7.0`
- `matplotlib>=3.4.0`
- `pyyaml>=6.0`
- `tqdm>=4.62.0`
- `pytube>=12.0.0`
- `tqdm>=4.62.0`

### Dependency Analysis by Component

#### **PyTorch-Dependent Components:**
- `tracknetv3/models/` - All files
- `tracknetv3/inference/streaming.py` - PyTorch streaming
- `tracknetv3/inference/offline.py` - Batch inference
- `tracknetv3/inference/helpers.py` - Inference utilities
- `tracknetv3/inference/config.py` - Inference config
- `tracknetv3/datasets/shuttlecock.py` - Training dataset
- `tracknetv3/datasets/video_iterable.py` - Video dataset
- `tracknetv3/evaluation/metrics.py` - WBCELoss, eval functions
- `tracknetv3/evaluation/ensemble.py` - Uses torch.nn.functional.softmax
- `tracknetv3/utils/metric.py` - Metric calculations
- `scripts/train.py` - Training script
- `scripts/test.py` - Testing script

#### **ONNX-Dependent Components:**
- `tracknetv3/inference/streaming_onnx.py` - ONNX Runtime streaming
- `tools/export_tracknet_onnx.py` - Export tool
- `tools/export_inpaintnet_onnx.py` - Export tool

#### **ML-Independent Components (Potential Core):**
- `tracknetv3/config/constants.py` - Configuration constants (no deps)
- `tracknetv3/utils/general.py` - Video/image utilities (cv2, numpy, pandas, PIL)
- `tracknetv3/utils/trajectory.py` - Trajectory processing (numpy)
- `tracknetv3/evaluation/predict.py` - Prediction utilities (cv2, numpy)

### Issues Identified

1. **Missing Module**: `tracknetv3/utils/visualize.py` is imported in `scripts/train.py` and `tools/preprocess.py` but doesn't exist
   - Functions needed with signatures (based on usage patterns):
     ```python
     def plot_heatmap_pred_sample(x, y, y_pred, c, bg_mode="", save_dir=""):
         """
         Plot heatmap prediction sample during training.

         Args:
             x (np.ndarray): Input images, shape (L, H, W, C)
             y (np.ndarray): Ground truth heatmaps, shape (L, H, W)
             y_pred (np.ndarray): Predicted heatmaps, shape (L, H, W)
             c (np.ndarray): Coordinates
             bg_mode (str): Background mode for visualization
             save_dir (str): Directory to save the plot
         """

     def plot_traj_pred_sample(coor_gt, refine_coor, inpaint_mask, save_dir=""):
         """
         Plot trajectory prediction sample during training.

         Args:
             coor_gt (np.ndarray): Ground truth coordinates
             refine_coor (np.ndarray): Refined/predicted coordinates
             inpaint_mask (np.ndarray): Inpainting mask
             save_dir (str): Directory to save the plot
         """

     def write_to_tb(model_name, tb_writer, losses, val_res, epoch):
         """
         Write training metrics to TensorBoard.

         Args:
             model_name (str): Model name (TrackNet or InpaintNet)
             tb_writer (SummaryWriter): TensorBoard writer instance
             losses (tuple): Tuple of (train_loss, val_loss)
             val_res (dict): Validation results dictionary
             epoch (int): Current epoch number
         """

     def plot_median_files(data_dir):
         """
         Plot median frames for preprocessing.

         Args:
             data_dir (str): Directory containing the data
         """
     ```

2. **Code Duplication**: `get_ensemble_weight()` appears in both:
   - `tracknetv3/evaluation/ensemble.py`
   - `scripts/test.py`
   - **Action**: Remove from `scripts/test.py` and import from `evaluation/ensemble.py`

3. **New Modules to Create**: Per user requirements, need to create:
   - `core/config/configs.py` - Configuration classes
   - `core/utils/preprocessing.py` - Preprocessing utilities
   - `core/evaluation/helpers.py` - Prediction helpers

### Current Usage Patterns

From demo files and scripts:
- **Inference**: Uses `tracknetv3.inference.streaming` or `tracknetv3.inference.streaming_onnx`
- **Training**: Uses `scripts/train.py` which imports from models, datasets, utils, evaluation
- **Evaluation**: Uses `tracknetv3.evaluation.metrics` and `tracknetv3.evaluation.predict`
- **Data Processing**: Uses `tracknetv3.utils.general` and `tracknetv3.utils.trajectory`

## Decisions & Rationale

### Decision 1: Core Module Scope

**Recommendation**: Include the following in `tracknet-core`:

```
tracknet-core/
├── config/
│   ├── constants.py              # No external dependencies
│   └── [new] configs.py          # Configuration classes
├── utils/
│   ├── general.py                # Video/image processing
│   ├── trajectory.py             # Trajectory manipulation
│   ├── visualize.py              # Visualization utilities (needs creation)
│   └── [new] preprocessing.py    # Preprocessing utilities
├── evaluation/
│   ├── predict.py                # Basic prediction logic
│   └── [new] helpers.py          # Prediction helpers
```

**Rationale**:
- **constants.py**: Pure Python, zero dependencies
- **configs.py**: Configuration classes for settings management (new, per user request)
- **general.py**: Essential for data loading and preprocessing in both inference and training
- **trajectory.py**: Used by both inference and training for trajectory smoothing
- **visualize.py**: Needed by training and tools, should be available across modules
- **preprocessing.py**: Preprocessing utilities (new, per user request)
- **predict.py**: Basic prediction logic that doesn't require models
- **helpers.py**: Prediction helpers (new, per user request)

**Dependencies for core**: `numpy`, `opencv-python`, `pillow`, `pandas`, `parse`, `matplotlib`, `scipy`, `pyyaml`, `tqdm`, `pytube`

**Excluded from core**:
- `utils/metric.py` - Contains ML-specific metric calculations
- `evaluation/ensemble.py` - Uses torch (keeps torch dependency per user preference)
- `evaluation/metrics.py` - Contains WBCELoss and ML-specific metrics

### Decision 2: ONNX Module Scope

**Recommendation**: Minimal ONNX-specific code

```
tracknet-onnx/
├── inference/
│   └── streaming_onnx.py         # ONNX Runtime inference
└── __init__.py
```

**Rationale**: Keep ONNX module as lightweight as possible - just the inference engine

**Dependencies**: `tracknet-core>=0.1.0`, `onnxruntime-gpu>=1.19.0`

**Excluded**:
- Export tools - these need PyTorch to export, so they go to tracknet-pt
- ONNX models - these are runtime artifacts, not code
- Ensemble - **MUST stay in PT module** due to torch dependency (decision confirmed by user, not optional)

### Decision 3: PT Module Scope (Merged Training)

**Recommendation**: Combine PyTorch inference and training

```
tracknet-pt/
├── models/
│   ├── tracknet.py               # TrackNet model
│   ├── inpaintnet.py             # InpaintNet model
│   └── blocks.py                 # Building blocks
├── inference/
│   ├── streaming.py              # PyTorch streaming
│   ├── offline.py                # Batch inference
│   └── helpers.py                # Inference utilities
├── datasets/
│   ├── shuttlecock.py            # Training dataset
│   └── video_iterable.py         # Video dataset
├── evaluation/
│   ├── ensemble.py               # Temporal ensemble (keeps torch dependency)
│   ├── metrics.py                # ML-specific metrics
│   └── predict.py                # (re-export from core if needed)
├── utils/
│   └── metric.py                 # ML-specific metrics
├── scripts/
│   ├── train.py                  # Training script
│   └── test.py                   # Testing script
└── [extras]
    ├── tools/                    # All tools
    └── demos/                    # All demos
```

**Rationale**:
- Training and PyTorch inference share the same dependencies (torch, tensorboard)
- Users who want training will also want PyTorch inference for development
- Simplifies dependency management
- All tools and demos are extras that can be optionally installed
- Ensemble keeps torch dependency as per user preference

**Dependencies**: `tracknet-core>=0.1.0`, `torch>=2.0.0`, `tensorboard>=2.20.0`

### Decision 4: Directory Structure Layout

**Recommendation**: Multi-level monorepo-style with separate subdirectories and uv workspaces

```
TrackNetV3/
├── pyproject.toml                # Root workspace config (uv workspace)
├── tracknet-core/                # Core module
│   ├── tracknet/
│   │   ├── config/
│   │   ├── utils/
│   │   └── evaluation/
│   ├── pyproject.toml            # Module-specific config
│   └── README.md
├── tracknet-onnx/                # ONNX module
│   ├── tracknet/
│   │   └── inference/
│   ├── pyproject.toml            # Module-specific config
│   └── README.md
├── tracknet-pt/                  # PT module (merged)
│   ├── tracknet/
│   │   ├── models/
│   │   ├── inference/
│   │   ├── datasets/
│   │   ├── evaluation/
│   │   ├── utils/
│   │   └── scripts/
│   ├── pyproject.toml            # Module-specific config
│   ├── extras/
│   │   ├── tools/
│   │   └── demos/
│   └── README.md
├── README.md                     # Main project README
```

**Rationale**:
- Clear separation of installable units
- Each module has its own pyproject.toml for uv
- Root pyproject.toml manages uv workspace
- Multi-level pyproject structure for better dependency management
- Each module can be installed independently with uv
- No PyPI publishing required - local development focus

### Decision 5: Import Path Changes

**Current**: `from tracknetv3.inference.streaming import ...`
**Proposed** (Deep explicit paths, per user preference):
- Core: `from tracknet.core.config.constants import IMG_DIM`
- ONNX: `from tracknet.onnx.inference.streaming_onnx import StreamingInferenceONNX`
- PT: `from tracknet.pt.models.tracknet import TrackNet`

**Rationale**:
- Explicit imports make dependencies clear
- No package-level exports needed in __init__.py files
- Easier to understand what's being imported
- No backward compatibility needed

**Import Strategy for Workspace Development**:
During migration, verify imports using sys.path manipulation with this order:
1. Always add core to sys.path first: `sys.path.insert(0, '/workspace/TrackNetV3/tracknet-core')`
2. Then add the module being tested: `sys.path.insert(0, '/workspace/TrackNetV3/tracknet-onnx')` or `sys.path.insert(0, '/workspace/TrackNetV3/tracknet-pt')`
3. This ensures core is always available before dependent modules
4. This approach works for migration verification; final development workflow may use uv sync

### Decision 6: Code Reorganization Strategy

**Refactor Priority** (Sequential execution order):
1. **Fix Missing Module**: Create `utils/visualize.py` first
2. **Remove Duplication**: Remove `get_ensemble_weight()` from `scripts/test.py`
3. **Create Directory Structure**: Set up new module directories
4. **Create Core Module**: Move/create core files (Task 1 - MUST complete first)
5. **Create ONNX Module**: Move ONNX files (Task 2 - depends on Task 1)
6. **Create PT Module**: Move PT files (Task 3 - depends on Task 1)
7. **Update Imports**: Fix all import statements (Task 4 - after all moves)
8. **Create pyproject.toml**: Set up multi-level configs (Task 5)
9. **Verify**: Test each module works independently (Task 6)

**Critical Execution Dependency**: Task 1 (core module) MUST complete before Tasks 2 and 3, because both ONNX and PT modules depend on core. Do not attempt parallel execution of module creation.

## Assumptions & Implications

### Assumption 1: Visualization Module Content
**Assumption**: The `utils/visualize.py` module should contain visualization functions for training and tools
**Implication**: Need to implement:
- `plot_heatmap_pred_sample()` - Plot heatmap predictions
- `plot_traj_pred_sample()` - Plot trajectory predictions
- `write_to_tb()` - Write to TensorBoard
- `plot_median_files()` - Plot median files (for preprocessing)

**If Incorrect**: May need to adjust based on actual usage patterns in training scripts

### Assumption 2: Ensemble Refactor Feasibility
**Assumption**: `torch.nn.functional.softmax` can be replaced with numpy's softmax or exponential normalization
**Implication**: `ensemble.py` can be moved to core module

**If Incorrect**: `ensemble.py` may need to stay in pt module, limiting core functionality

### Assumption 3: Tools and Demos as Extras
**Assumption**: Tools and demos should be installable extras rather than separate modules
**Implication**: They will be included in tracknet-pt with optional installation

**If Incorrect**: May need separate packages or different distribution strategy

### Assumption 4: Shared Code Between PT and ONNX
**Assumption**: PT and ONNX inference may share some code (e.g., configuration, data loading)
**Implication**: Need to identify shared code and keep it in core

**If Incorrect**: May need duplicate code or refactoring to minimize duplication

### Assumption 5: Tools and Demos Import Strategy
**Assumption**: Tools and demos will have their imports updated during the general import refactoring (Task 4), not during the file move task (Task 3).
**Implication**: Tools and demos may have broken imports until Task 4 completes, but they will be fixed in the comprehensive import update pass.

**If Incorrect**: Tools and demos may be left with broken imports and need special handling.

### Assumption 6: Development Workflow
**Assumption**: During migration, developers will use sys.path manipulation for import verification rather than formal package installation with uv. The workspace structure allows direct module access without installation.
**Implication**: Import verification commands use sys.path.insert() to add modules to Python path. This is acceptable for migration but may change for final development workflow.

**If Incorrect**: May need to update import verification to use uv sync or other installation methods.

### Verification Strategy Note
**Important**: When verifying that old `tracknetv3` imports have been removed, use grep with directory filters to avoid false positives:
```bash
# Only search within new module directories, not root or old locations
grep -R "tracknetv3" tracknet-core/ tracknet-onnx/ tracknet-pt/ tracknet-pt/extras/
```
This prevents false positives from unmigrated files or directories outside the module structure.

**Tools and Demos Verification**: After Task 4 (import refactoring), verify that tools and demos import correctly by testing a representative sample:
```bash
# Test a tool script
python tracknet-pt/extras/tools/preprocess.py --help

# Test a demo script
python tracknet-pt/extras/demos/demo_offline.py --help
```
This ensures tools and demos were properly updated during the general import refactoring.

## Open Questions

**Resolved**: No automated tests in current project, so no test organization changes needed.

## Proposed File Migration Map

### From `tracknetv3/` to `tracknet-core/tracknet/core/`
```
config/constants.py → core/config/constants.py
[create] config/configs.py → core/config/configs.py
utils/general.py → core/utils/general.py
utils/trajectory.py → core/utils/trajectory.py
[create] utils/preprocessing.py → core/utils/preprocessing.py
[create] utils/visualize.py → core/utils/visualize.py
evaluation/predict.py → core/evaluation/predict.py
[create] evaluation/helpers.py → core/evaluation/helpers.py
```

### From `tracknetv3/` to `tracknet-onnx/tracknet/onnx/`
```
inference/streaming_onnx.py → onnx/inference/streaming_onnx.py
```

### From `tracknetv3/` to `tracknet-pt/tracknet/pt/`
```
models/tracknet.py → pt/models/tracknet.py
models/inpaintnet.py → pt/models/inpaintnet.py
models/blocks.py → pt/models/blocks.py
inference/streaming.py → pt/inference/streaming.py
inference/offline.py → pt/inference/offline.py
inference/helpers.py → pt/inference/helpers.py
inference/config.py → pt/inference/config.py
datasets/shuttlecock.py → pt/datasets/shuttlecock.py
datasets/video_iterable.py → pt/datasets/video_iterable.py
evaluation/ensemble.py → pt/evaluation/ensemble.py (keeps torch dependency)
evaluation/metrics.py → pt/evaluation/metrics.py
utils/metric.py → pt/utils/metric.py
scripts/train.py → pt/scripts/train.py
scripts/test.py → pt/scripts/test.py
```

### From root to `tracknet-pt/[extras]/`
```
tools/ → pt/[extras]/tools/
demos/ → pt/[extras]/demos/
```

## Next Steps

1. **Clarify Testing**: Get user input on testing organization
2. **Create Execution Plan**: Convert this into an executable plan using plan-writer
3. **Review Plan**: Have plan-reviewer validate the plan
4. **Finalize**: Present approved plan to user
