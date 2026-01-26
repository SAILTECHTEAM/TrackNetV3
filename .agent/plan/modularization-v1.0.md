# TrackNetV3 Modularization v1.0 Plan
*Derived from the project name in the Draft.*

## Context

### Source Document
- **Draft**: /workspace/TrackNetV3/.agent/draft/modularization-v1.0.md
- **Objective**: Separate the current TrackNetV3 project into three installable modules (tracknet-core, tracknet-onnx, tracknet-pt) to allow selective installation and reduce dependency overhead.

### Summary of Requirements
- **Core Deliverables**:
  - Create module skeletons and pyproject.toml files for: 
    - /workspace/TrackNetV3/tracknet-core/
    - /workspace/TrackNetV3/tracknet-onnx/
    - /workspace/TrackNetV3/tracknet-pt/
  - Move/implement files according to the Draft's migration map (see "Proposed File Migration Map" in Draft) with exact file paths listed in each TODO.
  - Create missing modules (new files):
    - /workspace/TrackNetV3/tracknet-core/tracknet/core/config/configs.py
    - /workspace/TrackNetV3/tracknet-core/tracknet/core/utils/preprocessing.py
    - /workspace/TrackNetV3/tracknet-core/tracknet/core/utils/visualize.py
    - /workspace/TrackNetV3/tracknet-core/tracknet/core/evaluation/helpers.py
  - Remove duplication of get_ensemble_weight(): delete the duplicate in scripts/test.py and import the implementation from pt/evaluation/ensemble.py
  - Keep evaluation/ensemble.py in the PT module (tracknet-pt) to preserve torch dependency.
  - Set up multi-level pyproject.toml structure with uv workspaces (root + module pyproject files).
  - Update all imports to use deep explicit paths (no package-level exports).

- **Key Constraints**:
  - Module names must use the `tracknet-*` prefix.
  - No PyPI publishing is required.
  - Backward compatibility of import paths is NOT required; breaking changes acceptable.
  - Deep explicit imports must be used (no package-level re-exports in __init__.py).
  - ONNX module must not depend on PyTorch.
  - Tools and demos belong as extras in tracknet-pt, not separate modules.

- **Out of Scope**:
  - Writing implementation code for algorithms (this plan specifies file creation and refactors but not code bodies).
  - Publishing to PyPI or configuring CI for publishing.
  - Creating automated test suites: Draft states there are no automated tests; this plan assumes Manual QA as the verification strategy (explicitly stated below).

---

## Work Objectives & Verification Strategy

### Definition of Done
The project is complete when:
- [ ] The three module directories exist with their module-level pyproject.toml files: /workspace/TrackNetV3/tracknet-core/, /workspace/TrackNetV3/tracknet-onnx/, /workspace/TrackNetV3/tracknet-pt/
- [ ] All files listed in the Draft’s Proposed File Migration Map have been moved/created at the target paths and are importable using the new deep explicit paths (see "Verification Approach").
- [ ] The duplicate get_ensemble_weight() has been removed from scripts/test.py and scripts/test.py imports it from tracknet.pt.evaluation.ensemble.

### Verification Approach
Based on the Draft's statement that there are no automated tests and the absence of a stated testing strategy, the conservative, explicit approach is Manual QA.

**Decision**:
- **Test Infrastructure Exists**: NO (Draft: "No automated tests in current project")
- **Testing Mandate**: Manual QA Only (inferred per Exception Handling rule when Draft is silent)
- **Framework/Tools**: Python (import/run), simple shell commands (ls, python -c), runpy for scripts, and manual inspection of file contents.

If Manual QA Only:
- Each implementation task MUST include explicit import/run verification commands.
- Evidence Required: terminal command outputs (captured), and for any demo/scripts: recorded run output and exit code 0.

---

## Architecture & Design
Derived from the Draft's requirements and constraints.

High-level overview:
- Three sibling packages inside the repository root, each containing a 'tracknet' package subtree with a designated subpackage name that encodes the module role: tracknet/core, tracknet/onnx, tracknet/pt.
- Runtime dependency separation:
  - tracknet-core: lightweight libraries only (numpy, cv2, pillow, pandas, etc.)
  - tracknet-onnx: depends on tracknet-core and onnxruntime-gpu
  - tracknet-pt: depends on tracknet-core and torch + tensorboard

Textual component diagram:

tracknet-core (minimal deps)
  ├─ tracknet/core/config/
  ├─ tracknet/core/utils/
  └─ tracknet/core/evaluation/

tracknet-onnx (runtime ONNX)
  └─ tracknet/onnx/inference/streaming_onnx.py

tracknet-pt (torch)
  ├─ tracknet/pt/models/
  ├─ tracknet/pt/inference/
  ├─ tracknet/pt/evaluation/ (ensemble stays here)
  └─ extras/ (tools, demos)

Key architectural decisions (from Draft):
1. Keep core minimal and dependency-light; include utilities required by both PT and ONNX (constants, general utils, trajectory, visualize, preprocessing, predict, config classes).
   - Rationale: reduces install size for users who need only inference runtime.
2. ONNX package contains only runtime inference (streaming_onnx.py) and must not import torch.
   - Rationale: avoid heavy torch dependency for ONNX users.
3. PT package includes models, training scripts, export tools, and ensemble implementation (keeps torch dependency).
   - Rationale: training and PyTorch inference share dependencies; ensemble uses torch.nn.functional and is kept with PT.

---

## Task Breakdown

```
0 -> 1 -> 4 -> 5 -> 6
0 -> 2 -> 4 -> 5 -> 6
0 -> 3 -> 4 -> 5 -> 6
```

Explanation: Task 0 (setup) must run first. Tasks 1 (core creation), 2 (onnx creation), and 3 (pt creation) can start after setup in parallel (subject to file move dependencies). Task 4 (remove duplication + import updates) depends on files being in place. Task 5 (pyproject TOML adjustments) can be done alongside file moves but must be present before per-module import verification. Task 6 (final verification/manual QA) runs after all changes.

### Parallelization Guide
| Task Group | Tasks | Rationale for Parallelism |
|------------|-------|---------------------------|
| Group A    | 1, 2, 3 | Creating module skeletons and moving files are independent once root workspace exists. |

### Dependency Mapping
| Task | Depends On | Reason for Dependency |
|------|------------|----------------------|
| 1 (core files) | 0 | Root module directories must exist before files are written. |
| 2 (onnx module) | 0,1 | ONNX code will import core; core must be present for import verification. |
| 3 (pt module) | 0,1 | PT will import from core; core must be present for import verification. |
| 4 (imports & dedupe) | 1,2,3 | Import updates and duplication removal require target files to be in-place. |
| 5 (pyproject) | 0 | Workspace configuration should exist before module-level verification. |
| 6 (final QA) | 1-5 | All prior changes must be complete to verify module independence.

---

## TODOs
Each TODO is an atomic work unit combining implementation and verification.

### Task 0: Setup/Foundation

**Objective**: Create repository workspace structure and placeholder pyproject files so module creation can proceed.

**Implementation Steps**:
1. Create directories:
   - /workspace/TrackNetV3/tracknet-core/
   - /workspace/TrackNetV3/tracknet-onnx/
   - /workspace/TrackNetV3/tracknet-pt/
2. In each module directory create a minimal package layout and README.md and an empty tracknet/ subdirectory for the package tree. Example paths to create:
   - /workspace/TrackNetV3/tracknet-core/tracknet/
   - /workspace/TrackNetV3/tracknet-onnx/tracknet/
   - /workspace/TrackNetV3/tracknet-pt/tracknet/
3. Create root pyproject.toml at /workspace/TrackNetV3/pyproject.toml and module pyproject.toml placeholders at:
   - /workspace/TrackNetV3/tracknet-core/pyproject.toml
   - /workspace/TrackNetV3/tracknet-onnx/pyproject.toml
   - /workspace/TrackNetV3/tracknet-pt/pyproject.toml
   (Each file should declare the package directory as part of the uv workspace — exact TOML contents are implementation detail; include workspace members: "tracknet-core", "tracknet-onnx", "tracknet-pt")

**Parallelizable**: YES (core/onnx/pt directories created in parallel)

**References (CRITICAL)**:
> Exhaustively list reference points from the Draft or implied project context. Explain WHAT to extract and WHY it's relevant.
- **Pattern Reference**: [Draft:Lines 36-67] - Current project structure. Use this to map current file locations (source paths) to new module locations.
  - WHY: The Draft lists the existing files and their paths which must be moved.
- **Configuration Reference**: [Draft:Lines 40-49, 236-267] - Proposed module layout and multi-level pyproject usage.
  - WHY: Informs exact directories to create and that each module must have its own pyproject.toml and that a root pyproject should define the workspace.

**Acceptance Criteria**:
- [ ] Directories exist: run `ls -la /workspace/TrackNetV3` and verify tracknet-core, tracknet-onnx, tracknet-pt are present.
- [ ] Files exist: `test -f /workspace/TrackNetV3/pyproject.toml && echo ok` prints `ok`.
- [ ] Evidence Capture: save `ls -la` output to `/workspace/TrackNetV3/.agent/plan/evidence/task0_ls.txt`.

**Commit Guidance**:
- **Commit Suggested**: YES
- **Message**: feat(workspace): add module directories and placeholder pyproject files
- **Files**: pyproject.toml (root), tracknet-core/, tracknet-onnx/, tracknet-pt/ (initial skeleton)

---

### Task 1: Create/Move Core Module Files

**Objective**: Build the tracknet-core package and add/create the minimal modules specified in the Draft.

**Implementation Steps**:
1. Create target directories (if not already created by Task 0):
   - /workspace/TrackNetV3/tracknet-core/tracknet/core/config/
   - /workspace/TrackNetV3/tracknet-core/tracknet/core/utils/
   - /workspace/TrackNetV3/tracknet-core/tracknet/core/evaluation/
2. Move existing file:
   - FROM: /workspace/TrackNetV3/tracknetv3/config/constants.py
   - TO:   /workspace/TrackNetV3/tracknet-core/tracknet/core/config/constants.py
   (Use a git mv or file copy + delete depending on your workflow.)
3. Create new files (stubs that the developer will implement):
   - /workspace/TrackNetV3/tracknet-core/tracknet/core/config/configs.py
   - /workspace/TrackNetV3/tracknet-core/tracknet/core/utils/preprocessing.py
   - /workspace/TrackNetV3/tracknet-core/tracknet/core/utils/visualize.py
   - /workspace/TrackNetV3/tracknet-core/tracknet/core/evaluation/helpers.py
   For each new file, include module-level docstring and function signatures only (developer implements bodies).
4. Ensure each package directory has an __init__.py to make them importable (explicit exports are NOT required; keep them empty or with minimal metadata).

**Parallelizable**: NO for the move+create sequence per-file (move must precede imports verification). File creation across subpackages may be done in parallel.

**References (CRITICAL)**:
- **Source File**: [/workspace/TrackNetV3/tracknetv3/config/constants.py] - Existing constants file to move into core.
  - WHY: constants.py is pure-Python and listed in Draft as core candidate (Draft Lines ~109-116 and migration map lines ~41-51).
- **Pattern Reference**: [Draft:Lines 44-57, 69-75] - Core module recommended tree and dependency list.
  - WHY: Guides which functions and files should live in core and the minimal dependency set.
- **Usage Reference**: [Draft:Lines 30-66 and 31-66 earlier] - Locations where visualize is imported (scripts/train.py and tools/preprocess.py).
  - WHY: Determines which functions must be provided by visualize.py (plot_heatmap_pred_sample, plot_traj_pred_sample, write_to_tb, plot_median_files).

**Acceptance Criteria** (Manual QA commands):
- [ ] File existence: `test -f /workspace/TrackNetV3/tracknet-core/tracknet/core/config/constants.py && echo ok` prints `ok`.
- [ ] Import check: run (from repo root):
  - `python -c "import sys, importlib; sys.path.insert(0, '/workspace/TrackNetV3/tracknet-core'); importlib.import_module('tracknet.core.config.constants'); print('import-ok')"`
  - Expected output: `import-ok` and exit 0.
- [ ] For each new file (configs.py, preprocessing.py, visualize.py, helpers.py): run a similar import test (replace module path accordingly) and expect `import-ok`.
- [ ] Evidence Capture: Save the stdout of the import commands to `/workspace/TrackNetV3/.agent/plan/evidence/task1_imports.txt`.

**Commit Guidance**:
- **Commit Suggested**: YES
- **Message**: feat(core): add core package and missing utilities
- **Files**: moved constants.py, new configs.py, preprocessing.py, visualize.py, helpers.py

---

### Task 2: Create ONNX Module (tracknet-onnx)

**Objective**: Place ONNX Runtime inference code into a lightweight tracknet-onnx package and ensure it imports core utilities rather than machine-learning-specific code.

**Implementation Steps**:
1. Create target directories:
   - /workspace/TrackNetV3/tracknet-onnx/tracknet/onnx/inference/
2. Move/copy the existing ONNX streaming file:
   - FROM: /workspace/TrackNetV3/tracknetv3/inference/streaming_onnx.py
   - TO:   /workspace/TrackNetV3/tracknet-onnx/tracknet/onnx/inference/streaming_onnx.py
3. Update imports inside streaming_onnx.py to reference core deep paths where relevant (e.g., tracknet.core.utils.general, tracknet.core.config.constants). Do NOT reference torch.
4. Create /workspace/TrackNetV3/tracknet-onnx/pyproject.toml and declare dependency on tracknet-core (local path in workspace) and onnxruntime-gpu as runtime dependency (documented in the TOML).

**Parallelizable**: YES after Task 0 and Task 1 initial core files exist for import verification.

**References (CRITICAL)**:
- **Source File**: /workspace/TrackNetV3/tracknetv3/inference/streaming_onnx.py
  - WHY: The Draft identifies this file as the only ONNX-dependent code (Draft Lines ~104-108 and migration map lines ~53-56).
- **Dependency Reference**: [Draft:Lines 88-95, 176-195] - ONNX must not rely on torch; dependency list.
  - WHY: Ensures import updates avoid torch.

**Acceptance Criteria**:
- [ ] File exists at /workspace/TrackNetV3/tracknet-onnx/tracknet/onnx/inference/streaming_onnx.py
- [ ] Import check: `python -c "import sys, importlib; sys.path.insert(0,'/workspace/TrackNetV3/tracknet-core'); sys.path.insert(0,'/workspace/TrackNetV3/tracknet-onnx'); importlib.import_module('tracknet.onnx.inference.streaming_onnx'); print('onnx-import-ok')"`
  - Expected: `onnx-import-ok` and exit 0. If the module imports any torch symbols, the import will fail — this must be resolved.
- [ ] Evidence Capture: Save stdout to `/workspace/TrackNetV3/.agent/plan/evidence/task2_import.txt`.

**Commit Guidance**:
- **Commit Suggested**: YES
- **Message**: feat(onnx): add ONNX runtime inference module
- **Files**: streaming_onnx.py, tracknet-onnx/pyproject.toml

---

### Task 3: Create PT Module (tracknet-pt) and Move PT-Specific Code

**Objective**: Build tracknet-pt package containing PyTorch models, training/inference code, evaluation (ensemble kept here), tools, and demos as extras.

**Implementation Steps**:
1. Create target directories:
   - /workspace/TrackNetV3/tracknet-pt/tracknet/pt/models/
   - /workspace/TrackNetV3/tracknet-pt/tracknet/pt/inference/
   - /workspace/TrackNetV3/tracknet-pt/tracknet/pt/datasets/
   - /workspace/TrackNetV3/tracknet-pt/tracknet/pt/evaluation/
   - /workspace/TrackNetV3/tracknet-pt/tracknet/pt/utils/
   - /workspace/TrackNetV3/tracknet-pt/tracknet/pt/scripts/
2. Move files from the original tree into the PT module:
   - /workspace/TrackNetV3/tracknetv3/models/tracknet.py -> /workspace/TrackNetV3/tracknet-pt/tracknet/pt/models/tracknet.py
   - /workspace/TrackNetV3/tracknetv3/models/inpaintnet.py -> .../models/inpaintnet.py
   - /workspace/TrackNetV3/tracknetv3/models/blocks.py -> .../models/blocks.py
   - /workspace/TrackNetV3/tracknetv3/inference/streaming.py -> .../inference/streaming.py
   - /workspace/TrackNetV3/tracknetv3/inference/offline.py -> .../inference/offline.py
   - /workspace/TrackNetV3/tracknetv3/inference/helpers.py -> .../inference/helpers.py
   - /workspace/TrackNetV3/tracknetv3/inference/config.py -> .../inference/config.py
   - /workspace/TrackNetV3/tracknetv3/datasets/shuttlecock.py -> .../datasets/shuttlecock.py
   - /workspace/TrackNetV3/tracknetv3/datasets/video_iterable.py -> .../datasets/video_iterable.py
   - /workspace/TrackNetV3/tracknetv3/evaluation/ensemble.py -> .../evaluation/ensemble.py
   - /workspace/TrackNetV3/tracknetv3/evaluation/metrics.py -> .../evaluation/metrics.py
   - /workspace/TrackNetV3/tracknetv3/utils/metric.py -> .../utils/metric.py
   - /workspace/TrackNetV3/scripts/train.py -> /workspace/TrackNetV3/tracknet-pt/tracknet/pt/scripts/train.py
   - /workspace/TrackNetV3/scripts/test.py -> /workspace/TrackNetV3/tracknet-pt/tracknet/pt/scripts/test.py
3. Move extras into pt/extras/ per Draft:
   - /workspace/TrackNetV3/tools/ -> /workspace/TrackNetV3/tracknet-pt/extras/tools/
   - /workspace/TrackNetV3/demos/ -> /workspace/TrackNetV3/tracknet-pt/extras/demos/
4. Create /workspace/TrackNetV3/tracknet-pt/pyproject.toml and declare dependency on tracknet-core and torch/tensorboard in the TOML.

**Parallelizable**: YES (moves for models, datasets, utilities can be executed in parallel since they are separate files), but import verification depends on core moved first.

**References (CRITICAL)**:
- **Source Files**: /workspace/TrackNetV3/tracknetv3/models/*, /workspace/TrackNetV3/tracknetv3/inference/*, /workspace/TrackNetV3/tracknetv3/datasets/*, /workspace/TrackNetV3/tracknetv3/evaluation/*, /workspace/TrackNetV3/tracknetv3/utils/metric.py, /workspace/TrackNetV3/scripts/*.py
  - WHY: The Draft's migration map enumerates these files as PT-specific and needing relocation (Draft Lines ~59-74 and migration map lines ~60-74).
- **Extras**: /workspace/TrackNetV3/tools/, /workspace/TrackNetV3/demos/
  - WHY: Draft prescribes these become extras inside tracknet-pt (Draft Lines ~76-80 and 236-267).

**Acceptance Criteria**:
- [ ] Files exist at their target PT paths (check a representative subset with `test -f`).
- [ ] Import check (core must be on sys.path):
  - `python -c "import sys, importlib; sys.path.insert(0,'/workspace/TrackNetV3/tracknet-core'); sys.path.insert(0,'/workspace/TrackNetV3/tracknet-pt'); importlib.import_module('tracknet.pt.models.tracknet'); print('pt-model-import-ok')"`
  - Expected `pt-model-import-ok` and exit 0.
- [ ] Scripts test: run `python -m runpy /workspace/TrackNetV3/tracknet-pt/tracknet/pt/scripts/test.py` and verify it runs to completion (non-zero return indicates error) — for manual QA, capture stdout and exit code.
- [ ] Evidence Capture: Save outputs to `/workspace/TrackNetV3/.agent/plan/evidence/task3_imports.txt` and `/workspace/TrackNetV3/.agent/plan/evidence/task3_test_run.txt`.

**Commit Guidance**:
- **Commit Suggested**: YES (group this with Task 4 changes that adjust import paths)
- **Message**: feat(pt): add PyTorch models, inference, training scripts, extras
- **Files**: moved models/, inference/, datasets/, evaluation/ (ensemble kept here), utils/metric.py, scripts/, extras/

---

### Task 4: Remove Duplication and Update Imports (Deep Explicit Paths)

**Objective**: Remove duplicate get_ensemble_weight(), update imports across all modules to use deep explicit module paths as defined in Draft.

**Implementation Steps**:
1. Remove the duplicate implementation of get_ensemble_weight() from the script copy at:
   - /workspace/TrackNetV3/tracknet-pt/tracknet/pt/scripts/test.py
   and instead add an import at the top of that file:
   - `from tracknet.pt.evaluation.ensemble import get_ensemble_weight` (use exact deep path per final layout)
2. Search and replace import patterns (examples):
   - `from tracknetv3.config.constants import X` -> `from tracknet.core.config.constants import X` (adjust per file location and usage)
   - `from tracknetv3.inference.streaming_onnx import StreamingInferenceONNX` -> `from tracknet.onnx.inference.streaming_onnx import StreamingInferenceONNX`
   - `from tracknetv3.models.tracknet import TrackNet` -> `from tracknet.pt.models.tracknet import TrackNet`
3. Update any imports in tools and demos to reference core or pt modules appropriately. Tools that require model export or torch should import from tracknet.pt; ONNX runtime tools should import from tracknet.onnx and tracknet.core only.
4. Ensure no package-level re-exporting is relied upon (remove patterns that expect `from tracknetv3 import X`). Replace with explicit module imports.

**Parallelizable**: NO — import updates should be applied in a controlled pass and then validated.

**References (CRITICAL)**:
- **Draft:Decision 5 (Lines ~79-92)** - Example of deep explicit import paths and rationale.
  - WHY: Use the stated naming convention and examples to form the replacements.
- **Migration Map**: [Draft:Lines 39-74 mapping blocks] - Source→target file mapping to help decide correct replacement module path for each import.
  - WHY: Ensures imports point to the new target locations.

**Acceptance Criteria**:
- [ ] No remaining references to the old top-level package prefix `tracknetv3` in the moved code base. Run a search (repo root):
  - `grep -R "tracknetv3" || true` should return no matches in the new module directories.
- [ ] Representative import checks (run after setting sys.path for core and pt/onnx):
  - `python -c "import sys, importlib; sys.path.insert(0,'/workspace/TrackNetV3/tracknet-core'); sys.path.insert(0,'/workspace/TrackNetV3/tracknet-pt'); importlib.import_module('tracknet.pt.scripts.test'); print('scripts-test-import-ok')"`
  - Expected `scripts-test-import-ok` and exit 0.
- [ ] Evidence Capture: Save the grep output and import test outputs to `/workspace/TrackNetV3/.agent/plan/evidence/task4_checks.txt`.

**Commit Guidance**:
- **Commit Suggested**: YES
- **Message**: refactor(imports): replace tracknetv3 imports with deep explicit tracknet.* imports
- **Files**: multiple modified files (list in commit body)

---

### Task 5: Create pyproject.toml and Configure UV Workspace

**Objective**: Provide module-level pyproject files and root workspace pyproject so developers can install modules independently using uv (as per Draft requirement).

**Implementation Steps**:
1. Create /workspace/TrackNetV3/pyproject.toml with a workspace section listing the three modules as members: "tracknet-core", "tracknet-onnx", "tracknet-pt".
2. Create minimal pyproject.toml files in each module directory listing metadata and runtime dependencies per Draft:
   - tracknet-core/pyproject.toml -> dependencies: numpy, opencv-python, pillow, pandas, parse, matplotlib, scipy, pyyaml, tqdm, pytube
   - tracknet-onnx/pyproject.toml -> dependencies: tracknet-core (workspace local), onnxruntime-gpu
   - tracknet-pt/pyproject.toml -> dependencies: tracknet-core (workspace local), torch, tensorboard
3. Document extras for tracknet-pt in its pyproject (extras = tools, demos) so developers can optionally install them.

**Parallelizable**: YES

**References (CRITICAL)**:
- **Draft:Decision 4 (Lines ~236-267)** - Multi-level monorepo layout and uv workspace rationale.
  - WHY: Dictates workspace members and per-module pyproject placement.
- **Dependency Lists**: [Draft:Lines 69-87 and 232-235] - lists dependencies expected for each module.
  - WHY: Use to populate the dependencies section in each module pyproject.

**Acceptance Criteria**:
- [ ] Files exist: `test -f /workspace/TrackNetV3/pyproject.toml` and `test -f /workspace/TrackNetV3/tracknet-core/pyproject.toml` etc.
- [ ] The root pyproject contains the workspace members (manual inspection or `grep`): `grep -E "tracknet-core|tracknet-onnx|tracknet-pt" /workspace/TrackNetV3/pyproject.toml` should match.
- [ ] Evidence Capture: Save pyproject contents to `/workspace/TrackNetV3/.agent/plan/evidence/task5_pyproject.txt`.

**Commit Guidance**:
- **Commit Suggested**: YES
- **Message**: chore(build): add pyproject.toml workspace and module pyproject files

---

### Task 6: Final Verification & Manual QA

**Objective**: Run the prescribed manual checks to verify that each module is importable and that the critical scripts/tools run with the correct dependencies.

**Implementation Steps**:
1. With the repository root on PYTHONPATH or by inserting module paths in sys.path, run the import checks defined in Tasks 1-3 and Task 4.
2. Run key scripts/demos to ensure they behave:
   - PT training script (dry-run if necessary): `python -m runpy /workspace/TrackNetV3/tracknet-pt/tracknet/pt/scripts/train.py` and capture output.
   - ONNX streaming demo: `python -m runpy /workspace/TrackNetV3/tracknet-onnx/tracknet/onnx/inference/streaming_onnx.py` (or its demo wrapper) and capture output.
3. Verify duplicate removal: grep for get_ensemble_weight occurrences to confirm single authoritative definition:
   - `grep -R "get_ensemble_weight" /workspace/TrackNetV3 || true` should return exactly one implementation located at `/workspace/TrackNetV3/tracknet-pt/tracknet/pt/evaluation/ensemble.py` and any other callers (imports) but no duplicates.

**Parallelizable**: NO (final QA should be performed after all tasks complete)

**References (CRITICAL)**:
- **Draft:Issues Identified (Lines ~115-124)** - describes missing visualize imports and duplication of get_ensemble_weight.
  - WHY: Drives the specific checks to run.

**Acceptance Criteria**:
- [ ] All import checks from Tasks 1-3 pass.
- [ ] Scripts run without ImportError and exit with code 0 for simple smoke tests (capture output).
- [ ] Duplicate detection returns a single implementation of get_ensemble_weight (callers may remain).
- [ ] Evidence Capture: Collate all recorded outputs under `/workspace/TrackNetV3/.agent/plan/evidence/final_verification/`.

**Commit Guidance**:
- **Commit Suggested**: NO automated commit here; capture evidence first and then commit grouped changes if desired.

---

### Spike Task A: Ensemble Refactor Feasibility (Decision/Investigation)

**Objective**: Determine whether evaluation/ensemble.py can be implemented without torch (i.e., use numpy-based softmax) and be moved into core, or whether it must remain in tracknet-pt.

**Implementation Steps**:
1. Inspect `/workspace/TrackNetV3/tracknetv3/evaluation/ensemble.py` for any uses of torch beyond softmax (e.g., tensor-specific indexing, GPU tensors, torch.device usage, torch.nn functional beyond softmax).
2. If usage is limited to softmax/argmax/elementwise ops, prototype a small numpy replacement (local experiment) and validate behavior against the existing torch implementation on a small numeric example.
3. Decide:
   - If replacement successful and keeps behavior identical within numeric tolerance, document refactor plan to move ensemble into core and modify PT accordingly.
   - If replacement not safe (relies on torch-specific features or GPU tensors), keep ensemble in tracknet-pt and ensure the Draft's plan to keep it in PT is followed.

**Parallelizable**: YES (investigation can be done while core files are prepared)

**References (CRITICAL)**:
- **Source File**: /workspace/TrackNetV3/tracknetv3/evaluation/ensemble.py
  - WHY: Primary artifact whose dependencies determine whether ensemble can be moved.
- **Draft:Assumption 2 (Lines ~217-223)** - Suggests replacing torch.nn.functional.softmax with numpy softmax if feasible.
  - WHY: Directs the specific replacement to test.

**Acceptance Criteria**:
- [ ] A short decision memo is saved to `/workspace/TrackNetV3/.agent/plan/evidence/spike_ensemble_decision.txt` stating: "MOVE_TO_CORE" or "KEEP_IN_PT" and summarizing the evidence (lines of torch usage preventing move, or tests showing numpy parity).

**Commit Guidance**:
- **Commit Suggested**: NO (this is investigative; if code changes are made, commit in Task 3/4 scope)

---

## Success Criteria & Final Verification

### Final Integration Test
```bash
# Verify core import
python -c "import sys, importlib; sys.path.insert(0,'/workspace/TrackNetV3/tracknet-core'); importlib.import_module('tracknet.core.config.constants'); print('core-ok')" # Expected: core-ok

# Verify onnx import
python -c "import sys, importlib; sys.path.insert(0,'/workspace/TrackNetV3/tracknet-core'); sys.path.insert(0,'/workspace/TrackNetV3/tracknet-onnx'); importlib.import_module('tracknet.onnx.inference.streaming_onnx'); print('onnx-ok')" # Expected: onnx-ok

# Verify pt model import
python -c "import sys, importlib; sys.path.insert(0,'/workspace/TrackNetV3/tracknet-core'); sys.path.insert(0,'/workspace/TrackNetV3/tracknet-pt'); importlib.import_module('tracknet.pt.models.tracknet'); print('pt-ok')" # Expected: pt-ok
```

### Project Completion Checklist
- [ ] All "Core Deliverables" (from Context section) are implemented and verified.
- [ ] All key constraints have been respected (naming, no PyPI publishing, deep explicit imports, ONNX no-torch requirement).
- [ ] Duplicate get_ensemble_weight removed from scripts/test.py and only one authoritative implementation exists in tracknet-pt/tracknet/pt/evaluation/ensemble.py.
- [ ] All import statements updated to deep explicit forms and no references to `tracknetv3` remain.
- [ ] Root and module pyproject.toml files created and include workspace/module dependency declarations.
- [ ] All manual verification steps and evidence files are saved under `/workspace/TrackNetV3/.agent/plan/evidence/`.
