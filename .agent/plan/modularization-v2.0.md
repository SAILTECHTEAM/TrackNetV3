# TrackNetV3 Modularization Plan v2.0
*Derived from the project draft at /workspace/TrackNetV3/.agent/draft/modularization-v1.0.md.*

## Context

### Source Document
- **Draft**: /workspace/TrackNetV3/.agent/draft/modularization-v1.0.md
- **Objective**: Separate the current TrackNetV3 project into three installable modules (tracknet-core, tracknet-onnx, tracknet-pt) to reduce dependency overhead and enable selective installation.

### Summary of Requirements
- **Core Deliverables**:
  - Create modules: tracknet-core, tracknet-onnx, tracknet-pt (with a multi-level pyproject.toml workspace using uv).
  - Ensure ensemble remains in tracknet-pt (torch dependency).
  - Implement deep explicit imports (no package-level exports).
  - Create missing modules/files inside core: visualize.py (with complete function signatures), configs.py, preprocessing.py, helpers.py.
  - Remove duplicated function get_ensemble_weight() from scripts/test.py and import it from pt/evaluation/ensemble.py.
  - Put tools and demos as extras under tracknet-pt/extras/.
- **Key Constraints**:
  - No PyPI publishing.
  - Backward compatibility NOT required (imports may change).
  - ONNX module must NOT require PyTorch.
  - Ensemble MUST stay in PT module (torch dependency).
  - Sequential execution required: Task 1 → Tasks 2 & 3 → Task 4 → Task 5 → Task 6.
- **Out of Scope**:
  - Publishing to PyPI.
  - Maintaining old import compatibility.

---

## Work Objectives & Verification Strategy

### Definition of Done
The project is complete when:
- [ ] tracknet-core/, tracknet-onnx/, and tracknet-pt/ directories exist with the file structure defined in the Draft.
- [ ] All missing modules (visualize.py with signatures, configs.py, preprocessing.py, helpers.py) are present in tracknet-core and referenced imports updated.
- [ ] get_ensemble_weight exists only in tracknet-pt/evaluation/ensemble.py and scripts/test.py imports it from there.
- [ ] Deep explicit import style is used across modules (no package-level exports).
- [ ] Root and per-module pyproject.toml files exist and indicate uv workspace membership.
- [ ] Verification commands (grep, example python import checks, tools/demos --help) pass as defined below.

### Verification Approach
Based on the Draft's stated testing strategy: Manual QA with targeted command checks.

Decision:
- **Test Infrastructure Exists**: NO (Draft: no automated tests present)
- **Testing Mandate**: Manual QA Only (Draft states no automated tests; manual verification commands provided)
- **Framework/Tools**: grep (with directory filters), python import checks via sys.path insertion, running representative tool/demo scripts with --help

If Manual QA Only:
- Each task includes explicit, step-by-step manual verification procedures (commands to run and expected outputs).
- Evidence required: terminal output of verification commands; for tools/demos, run --help and confirm exit code 0 and usage printed.

---

## Architecture & Design
Derived from the Draft's requirements and constraints.

High-level overview:
- tracknet-core: common utilities and config (no heavy ML deps)
- tracknet-onnx: ONNX Runtime inference only (no torch)
- tracknet-pt: PyTorch models, training, ensemble, tools/demos extras

Component textual diagram:

tracknet-core/
  - config/ (constants.py, configs.py)
  - utils/ (general.py, trajectory.py, preprocessing.py, visualize.py)
  - evaluation/ (predict.py, helpers.py)

tracknet-onnx/
  - inference/ (streaming_onnx.py)

tracknet-pt/
  - models/, inference/, datasets/, evaluation/ (ensemble.py, metrics.py), utils/, scripts/, extras/{tools,demos}

Key architectural decisions (from Draft):
1. Ensemble must remain in tracknet-pt to keep torch dependency and avoid bringing torch into core or onnx modules (Decision: ensemble stays in PT). Rationale: ensemble uses torch.nn.functional.softmax and other torch-specific operations.
2. Deep explicit imports (no package-level exports). Rationale: makes dependencies explicit and avoids ambiguous package re-exports; simplifies reasoning about module requirements.
3. Multi-level pyproject.toml with uv workspaces. Rationale: enable per-module installability during local development without PyPI publishing.

---

## Task Breakdown

```
0 => 1 => (2 & 3) => 4 => 5 => 6
```

### Parallelization Guide
| Task Group | Tasks | Rationale for Parallelism |
|------------|-------|---------------------------|
| Group A    | 2, 3  | After core (Task 1) is complete, ONNX and PT moves can proceed in parallel because they both depend only on core being present. However, per Critical Requirements the plan enforces sequential Task 1 then 2 and 3—2 and 3 may run in parallel only after Task 1 finishes. |

### Dependency Mapping
| Task | Depends On | Reason for Dependency |
|------|------------|-----------------------|
| 1    | -          | Create core first; both ONNX and PT rely on core files |
| 2    | 1          | ONNX needs core for shared utils/config |
| 3    | 1          | PT needs core for shared utils/config; ensemble must stay in PT |
| 4    | 2,3        | Import refactor requires new module layout in place |
| 5    | 4          | Create pyproject.toml workspace after imports are updated and verified locally |
| 6    | 5          | Final verification across workspaces and extras

---

## TODOs
Each TODO is a single, atomic unit of work combining implementation and verification.

### Task 0: [Preparation — Repository snapshot]

Objective: Create a snapshot branch (or copy) for migration so changes are isolated.

Implementation Steps:
1. Create a new branch: git checkout -b modularization/v2.0-snapshot
2. Ensure working tree is clean and commit any outstanding changes.

Parallelizable: NO

References (CRITICAL):
> These repository actions are preparatory and drawn from the Draft context (root pyproject, existing tracknetv3/ layout).
- **Source Draft**: [/workspace/TrackNetV3/.agent/draft/modularization-v1.0.md] - Use this as the authoritative map of files to move and transform.

Acceptance Criteria (Manual Verification):
- [ ] Git branch exists: git branch --list modularization/v2.0-snapshot
- [ ] git status shows no unstaged changes

Commit Guidance:
- Commit Suggested: YES
- Message: feat(mod): snapshot repo for modularization v2.0
- Files: none (branch only)

---

### Task 1: [Create tracknet-core module and missing core files] (MUST complete first)

Objective: Create tracknet-core/ with the required directory structure and implement the missing modules listed in the Draft. This task must finish before any ONNX/PT moves.

Implementation Steps:
1. Create directory tracknet-core/tracknet/ with subdirectories: config/, utils/, evaluation/.
2. Move or recreate the following files into tracknet-core as per Draft mapping:
   - config/constants.py (move from tracknetv3/config/constants.py)
   - Create config/configs.py (new)
   - utils/general.py (move)
   - utils/trajectory.py (move)
   - Create utils/preprocessing.py (new)
   - Create utils/visualize.py (new) — include the complete function signatures described in the Draft:
     - plot_heatmap_pred_sample(x, y, y_pred, c, bg_mode="", save_dir="")
     - plot_traj_pred_sample(coor_gt, refine_coor, inpaint_mask, save_dir="")
     - write_to_tb(model_name, tb_writer, losses, val_res, epoch)
     - plot_median_files(data_dir)
   - evaluation/predict.py (move)
   - Create evaluation/helpers.py (new)
3. Ensure these core files do not import torch or onnxruntime; they must be free of heavy ML dependencies per Draft.
4. Commit the new core module layout.

Parallelizable: NO (must be completed before Tasks 2 & 3)

References (CRITICAL):
> Extract paths and behavior from the Draft migration map and usage notes. Explain WHAT to extract and WHY.
- **File Mapping**: [Draft lines: migration map section] — Follow the From `tracknetv3/` to `tracknet-core/` mapping exactly to relocate code and create missing files.
  - config/constants.py → core/config/constants.py - extract constants used across modules (why: shared config)
  - [create] config/configs.py → core/config/configs.py - implement configuration classes (why: centralize settings)
  - utils/general.py → core/utils/general.py - video/image utilities (why: used by inference and preprocessing)
  - utils/trajectory.py → core/utils/trajectory.py - trajectory processing (why: shared logic)
  - [create] utils/preprocessing.py → core/utils/preprocessing.py - preprocessing utilities (why: used by tools and demos)
  - [create] utils/visualize.py → core/utils/visualize.py - visualization functions with signatures provided in Draft (why: referenced by scripts/tools)
  - evaluation/predict.py → core/evaluation/predict.py - basic prediction logic (why: shared by ONNX/PT)
  - [create] evaluation/helpers.py → core/evaluation/helpers.py - helpers for prediction (why: support predict.py and inference modules)

Acceptance Criteria (Manual Verification):
- [ ] Directory structure exists: ls tracknet-core/tracknet/{config,utils,evaluation}
- [ ] Missing files exist: tracknet-core/tracknet/config/configs.py, tracknet-core/tracknet/utils/preprocessing.py, tracknet-core/tracknet/utils/visualize.py, tracknet-core/tracknet/evaluation/helpers.py
- [ ] Visualize signatures present: grep -n "def plot_heatmap_pred_sample" tracknet-core/ -R
- [ ] No torch or onnxruntime imports in core: grep -R "import torch\|onnxruntime" tracknet-core/ || true (expected: no matches)

Evidence Capture:
- [ ] Save terminal outputs for ls and grep checks.

Commit Guidance:
- Commit Suggested: YES
- Message: feat(core): create tracknet-core with missing modules and visualization signatures
- Files: list of files added under tracknet-core/

---

### Task 2: [Create tracknet-onnx module] (Depends on Task 1)

Objective: Create tracknet-onnx/ containing ONNX runtime inference code and depend on tracknet-core.

Implementation Steps:
1. Create directory tracknet-onnx/tracknet/inference/.
2. Move inference/streaming_onnx.py into tracknet-onnx/tracknet/inference/streaming_onnx.py per Draft mapping.
3. Ensure imports inside streaming_onnx.py reference core via deep explicit imports, e.g., from tracknet.core.utils.general import ... (update import paths to match new layout).
4. Confirm ONNX module has no torch imports and depends on tracknet-core only.
5. Commit the onnx module changes.

Parallelizable: YES (with Task 3) but only after Task 1 completes

References (CRITICAL):
- **File Mapping**: inference/streaming_onnx.py → onnx/inference/streaming_onnx.py — follow Draft mapping (why: keep ONNX lightweight and separate torch)
- **Import Pattern**: Decision 5 (Deep explicit paths) — ensure streaming_onnx.py uses core imports (why: explicit dependencies and to avoid package-level exports)

Acceptance Criteria (Manual Verification):
- [ ] File exists: ls tracknet-onnx/tracknet/inference/streaming_onnx.py
- [ ] streaming_onnx.py imports core via explicit path: grep -n "tracknet.core" tracknet-onnx/ -R
- [ ] No torch imports in tracknet-onnx/: grep -R "import torch" tracknet-onnx/ || true (expected: no matches)

Evidence Capture:
- [ ] Save outputs of ls and grep commands.

Commit Guidance:
- Commit Suggested: YES
- Message: feat(onnx): add streaming_onnx inference module referencing core
- Files: tracknet-onnx/tracknet/inference/streaming_onnx.py

---

### Task 3: [Create tracknet-pt module] (Depends on Task 1)

Objective: Create tracknet-pt/ and move PyTorch-related code (models, inference, datasets, evaluation including ensemble) into it. Ensure ensemble stays here.

Implementation Steps:
1. Create tracknet-pt/tracknet/ with subdirectories: models/, inference/, datasets/, evaluation/, utils/, scripts/, extras/{tools,demos}.
2. Move files as per Draft mapping (models/tracknet.py, inpaintnet.py, blocks.py; inference/streaming.py, offline.py, helpers.py, config.py; datasets/shuttlecock.py, video_iterable.py; evaluation/ensemble.py, metrics.py; utils/metric.py; scripts/train.py, test.py; tools/ and demos/ to extras/).
3. Remove duplicated get_ensemble_weight() from scripts/test.py and update scripts/test.py to import get_ensemble_weight from tracknet.pt.evaluation.ensemble (deep import). Ensure only one authoritative implementation remains in pt/evaluation/ensemble.py.
4. Ensure ensemble.py uses torch and remains inside PT module (do not move to core).
5. Update internal imports in PT files to import core utilities via deep explicit imports (e.g., from tracknet.core.utils.general import ...).
6. Commit the PT module changes.

Parallelizable: YES (with Task 2) but only after Task 1 completes

References (CRITICAL):
- **File Mapping**: Follow Draft mapping from tracknetv3/ to tracknet-pt/ (see Draft migration map). Extract the exact files and rationale:
  - models/* → pt/models/ (why: keep torch models together)
  - inference/* → pt/inference/ (why: torch inference relies on models and utils)
  - evaluation/ensemble.py → pt/evaluation/ensemble.py (why: ensemble uses torch; must remain in PT)
  - tools/ and demos/ → pt/extras/ (why: extras not separate modules)
  - scripts/test.py: remove local get_ensemble_weight duplicate and import from pt/evaluation/ensemble.py (why: remove duplication, single source of truth)

Acceptance Criteria (Manual Verification):
- [ ] Directories and files exist under tracknet-pt/ per mapping: ls tracknet-pt/tracknet/
- [ ] grep for get_ensemble_weight shows only one definition located in tracknet-pt/tracknet/evaluation/ensemble.py:
  - grep -R "def get_ensemble_weight" -n . | grep tracknet-pt/
- [ ] scripts/test.py imports get_ensemble_weight from tracknet.pt.evaluation.ensemble: grep -n "get_ensemble_weight" tracknet-pt/ -R
- [ ] ensemble.py contains torch imports and remains within pt: grep -n "import torch" tracknet-pt/ -R

Evidence Capture:
- [ ] Save outputs from ls and grep commands showing single definition and correct imports.

Commit Guidance:
- Commit Suggested: YES (group with Task 2 if both completed together)
- Message: feat(pt): create tracknet-pt layout, move torch code, centralize ensemble
- Files: list of files moved into tracknet-pt/

---

### Task 4: [Update imports across repository to deep explicit paths]

Objective: Replace old tracknetv3 imports with deep explicit imports referencing the new module layout. No package-level exports.

Implementation Steps:
1. Identify all occurrences of "tracknetv3" in the codebase within the new module directories using the Draft-recommended grep filter:
   - grep -R "tracknetv3" tracknet-core/ tracknet-onnx/ tracknet-pt/ tracknet-pt/extras/
2. For each occurrence, update import to the deep explicit path (examples from Draft):
   - from tracknet.core.config.constants import IMG_DIM
   - from tracknet.onnx.inference.streaming_onnx import StreamingInferenceONNX
   - from tracknet.pt.models.tracknet import TrackNet
3. Ensure no __init__.py package-level re-exports are added; prefer explicit module imports.
4. Commit import changes.

Parallelizable: NO (depends on Tasks 2 & 3 completion)

References (CRITICAL):
- **Import Strategy**: Draft Decision 5 and Import Strategy section — use explicit deep imports and the sys.path verification order during local tests (why: to ensure core is found before dependent modules during migration).
- **Grep Guidance**: Draft verification note — use filtered grep to avoid false positives (why: to only search within migrated code directories).

Acceptance Criteria (Manual Verification):
- [ ] No occurrences of "tracknetv3" remain in new module directories:
  - grep -R "tracknetv3" tracknet-core/ tracknet-onnx/ tracknet-pt/ tracknet-pt/extras/ || true (expected: no matches)
- [ ] Representative import checks succeed when using sys.path insertion order from Draft. Example manual check (run in shell):
  - python - <<'PY'
import sys
sys.path.insert(0, '/workspace/TrackNetV3/tracknet-core')
sys.path.insert(0, '/workspace/TrackNetV3/tracknet-pt')
from tracknet.pt.models.tracknet import TrackNet
print('import ok')
PY
  - Expected: prints "import ok" and exits 0

Evidence Capture:
- [ ] Save grep outputs and the python import check terminal output.

Commit Guidance:
- Commit Suggested: YES
- Message: refactor(imports): replace tracknetv3 imports with deep explicit module imports
- Files: list of modified files with imports updated

---

### Task 5: [Create multi-level pyproject.toml with uv workspaces]

Objective: Add root and per-module pyproject.toml files configured for local workspace development with uv (multi-level pyproject structure) as defined in Draft.

Implementation Steps:
1. Create/modify root pyproject.toml to declare uv workspace members: tracknet-core, tracknet-onnx, tracknet-pt (follow the Draft statement about uv workspaces).
2. Create per-module pyproject.toml files under each module with minimal metadata and dependencies as described in Draft:
   - tracknet-core/pyproject.toml (core deps: numpy, opencv-python, pillow, pandas, parse, matplotlib, scipy, pyyaml, tqdm, pytube)
   - tracknet-onnx/pyproject.toml (depends on tracknet-core, onnxruntime-gpu)
   - tracknet-pt/pyproject.toml (depends on tracknet-core, torch, tensorboard; extras point to extras/ directory)
3. Commit pyproject.toml files.

Parallelizable: NO (run after imports updated so pyproject references match code layout)

References (CRITICAL):
- **Draft Sections**: Decision 4 (Directory Structure Layout) and Decision 3/1 (module dependencies and excluded items) — extract required dependency lists and workspace membership from Draft (why: ensure pyproject reflects the planned module responsibilities).

Acceptance Criteria (Manual Verification):
- [ ] Files exist: ls pyproject.toml tracknet-core/pyproject.toml tracknet-onnx/pyproject.toml tracknet-pt/pyproject.toml
- [ ] Root pyproject mentions workspace members: grep -n "tracknet-core\|tracknet-onnx\|tracknet-pt" pyproject.toml
- [ ] Each module's pyproject contains at least one dependency string referenced in the Draft (e.g., 'torch' in tracknet-pt): grep -n "torch\|onnxruntime\|numpy" tracknet-*/pyproject.toml

Evidence Capture:
- [ ] Save outputs of ls and grep checks.

Commit Guidance:
- Commit Suggested: YES
- Message: chore(pyproject): add multi-level pyproject.toml for uv workspaces
- Files: pyproject.toml, tracknet-*/pyproject.toml

---

### Task 6: [Verification pass — grep, import checks, tools/demos run] (Final)

Objective: Run the full verification sequence as defined in the Draft to ensure modularization completed, imports updated, duplicates removed, and extras working.

Implementation Steps (Manual Verification commands):
1. Verify no remaining "tracknetv3" imports within migrated directories:
   - grep -R "tracknetv3" tracknet-core/ tracknet-onnx/ tracknet-pt/ tracknet-pt/extras/ || true
   - Expected: no matches
2. Verify single definition of get_ensemble_weight:
   - grep -R "def get_ensemble_weight" -n . | grep tracknet-pt/ || true
   - Expected: single match in tracknet-pt/tracknet/evaluation/ensemble.py
3. Run representative import checks using sys.path insertion (order from Draft):
   - python - <<'PY'
import sys
sys.path.insert(0, '/workspace/TrackNetV3/tracknet-core')
sys.path.insert(0, '/workspace/TrackNetV3/tracknet-pt')
from tracknet.pt.models.tracknet import TrackNet
print('pt import ok')
PY
   - python - <<'PY'
import sys
sys.path.insert(0, '/workspace/TrackNetV3/tracknet-core')
sys.path.insert(0, '/workspace/TrackNetV3/tracknet-onnx')
from tracknet.onnx.inference.streaming_onnx import StreamingInferenceONNX
print('onnx import ok')
PY
4. Test extras (tools & demos) as per Draft examples:
   - python tracknet-pt/extras/tools/preprocess.py --help
   - python tracknet-pt/extras/demos/demo_offline.py --help
   - Expected: scripts exit 0 and print usage/help
5. Confirm pyproject workspace references (root):
   - grep -n "tracknet-core\|tracknet-onnx\|tracknet-pt" pyproject.toml

Acceptance Criteria:
- [ ] grep for "tracknetv3" within module directories returns no matches
- [ ] Single definition of get_ensemble_weight in tracknet-pt/evaluation/ensemble.py
- [ ] Import checks print "pt import ok" and "onnx import ok"
- [ ] Tools and demos --help run and show usage
- [ ] Root pyproject references modules

Evidence Capture:
- [ ] Save all terminal outputs for grep, import checks, and tools/demos runs. Collect them into a verification log file: .agent/plan/modularization-v2.0-verification.txt

Commit Guidance:
- Commit Suggested: NO (verification artifacts only), but record findings in PR or issue

---

## Success Criteria & Final Verification

### Final Integration Test
```bash
# Check no old package references remain
grep -R "tracknetv3" tracknet-core/ tracknet-onnx/ tracknet-pt/ tracknet-pt/extras/ # Expected: no output
```

### Project Completion Checklist
- [ ] All "Core Deliverables" implemented and verified.
- [ ] All constraints respected (ensemble in PT, ONNX without torch, no PyPI publishing, no backward compatibility guarantees).
- [ ] No code duplication remains for get_ensemble_weight.
- [ ] All verification steps defined above passed and evidence captured.
