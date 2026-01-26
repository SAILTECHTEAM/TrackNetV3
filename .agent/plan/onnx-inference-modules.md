# ONNX-based Inference Modules for TrackNetV3
*Derived from the project name in the Draft.*

## Context

### Source Document
- **Draft**: `.agent/drafts/onnx-inference-modules-v1.0.md`
- **Objective**: Create ONNX Runtime-based inference modules (`TrackNetModuleONNX` and `InpaintModuleONNX`) that mirror the functionality of existing PyTorch-based modules while maintaining identical API.

### Summary of Requirements
- **Core Deliverables**:
  - New file `tracknetv3/inference/streaming_onnx.py` containing `TrackNetModuleONNX` and `InpaintModuleONNX` classes
  - Updated `pyproject.toml` to add `onnxruntime-gpu>=1.19.0` dependency
  - Updated `tracknetv3/inference/__init__.py` to export `TrackNetModuleONNX` and `InpaintModuleONNX`

- **Key Constraints**:
  - Maintain identical API signatures to PyTorch versions (same `__init__` parameters, same method names, same return types)
  - Use `onnxruntime-gpu` for GPU acceleration with CPU fallback
  - Validate ONNX model metadata against expected configuration (seq_len, bg_mode, batch_mode, batch_size)
  - Remove AMP-related timing statistics (not applicable to ONNX)
  - Accept `model_path: str` instead of `torch.nn.Module` in `__init__` parameter

- **Out of Scope**:
  - ONNX model export scripts (already exist in `tools/export_tracknet_onnx.py` and `tools/export_inpaintnet_onnx.py`)
  - ONNX optimization (using default `ORT_ENABLE_ALL` level only)
  - Multi-GPU support (will use default GPU device_id: 0 when CUDA provider is available)
  - Dynamic sequence lengths at runtime (assume fixed sequence length as exported)
  - Model quantization/int8 (focus on FP32 models only)

---

## Work Objectives & Verification Strategy

### Definition of Done
The project is complete when:
- [ ] `tracknetv3/inference/streaming_onnx.py` exists with both `TrackNetModuleONNX` and `InpaintModuleONNX` fully implemented
- [ ] `pyproject.toml` includes `onnxruntime-gpu>=1.19.0` in dependencies
- [ ] `tracknetv3/inference/__init__.py` exports both ONNX module classes
- [ ] Both ONNX modules maintain identical API to PyTorch versions (method signatures, return types)
- [ ] ONNX models can be loaded, validated, and used for inference
- [ ] Code passes static analysis (ruff) without new errors

### Verification Approach
*Based on the Draft's stated testing criteria and the absence of an automated test infrastructure for these modules.*

**Decision**:
- **Test Infrastructure Exists**: NO (No existing test infrastructure for streaming modules found)
- **Testing Mandate**: Manual QA Only
- **Framework/Tools**: Python REPL, simple verification scripts, manual inspection

**Manual QA Verification**:
- **CRITICAL**: Each task MUST include explicit, step-by-step manual verification procedures.
- **Evidence Required**: Import verification, metadata validation, simple inference test, error handling verification.

---

## Architecture & Design

### High-Level Overview

The ONNX-based inference modules mirror the existing PyTorch architecture but replace PyTorch model execution with ONNX Runtime inference:

```
┌─────────────────────────────────────────────────────────────┐
│                    tracknetv3/inference/                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  streaming.py (Existing PyTorch)                             │
│  ├── TrackNetModule                                          │
│  │   ├── push(frame_bgr, frame_id) → pred                    │
│  │   └── flush() → list[pred]                                │
│  └── InpaintModule                                           │
│      ├── push(pred) → pred                                   │
│      └── flush() → list[pred]                                │
│                                                               │
│  streaming_onnx.py (NEW - This implementation)                │
│  ├── TrackNetModuleONNX                                      │
│  │   ├── push(frame_bgr, frame_id) → pred                    │
│  │   └── flush() → list[pred]                                │
│  └── InpaintModuleONNX                                       │
│      ├── push(pred) → pred                                   │
│      └── flush() → list[pred]                                │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

**TrackNetModuleONNX Processing Pipeline**:

```
Input Frame (BGR)
    ↓
_ensure_scaler() - Set up image scaler
    ↓
_maybe_build_median() - Build median if needed (warmup)
    ↓
_process_one() - Preprocess single frame to NumPy array
    ↓
Buffer in self._proc (deque of seq_len)
    ↓
Concatenate frames with background handling
    ↓
ONNX Runtime Inference (session.run)
    ↓
Ensemble weight accumulation (self._acc_sum, self._acc_w)
    ↓
_heatmap_to_xy() - Convert heatmap to (X, Y, Visibility)
    ↓
Output: {"Frame": int, "X": int, "Y": int, "Visibility": int}
```

**InpaintModuleONNX Processing Pipeline**:

```
Input Prediction (from TrackNet)
    ↓
Normalize coordinates to [0, 1]
    ↓
Create mask for occlusions (m=1 if occluded, m=0 if visible)
    ↓
Buffer in self._coords and self._mask (deque of seq_len)
    ↓
ONNX Runtime Inference (session.run with coords and mask inputs)
    ↓
Post-process: out = out * mask + coords * (1 - mask)
    ↓
Denormalize coordinates to pixel space
    ↓
Output: {"Frame": int, "X": int, "Y": int, "Visibility": int}
```

### Key Architectural Decisions with Rationale

1. **Separate file (`streaming_onnx.py`) instead of modifying `streaming.py`**:
   - **Rationale**: Keeps PyTorch and ONNX implementations side-by-side for easy comparison and allows users to choose runtime based on their needs.
   - **Trade-off**: Some code duplication of preprocessing/logic, but cleaner separation of concerns.

2. **Accept `model_path: str` instead of `torch.nn.Module` in `__init__`**:
   - **Rationale**: ONNX models are loaded from disk via ONNX Runtime, not instantiated as Python objects. The `model_path` parameter maintains API compatibility while appropriately handling ONNX's loading mechanism.
   - **Impact**: Users must provide path to ONNX model file instead of PyTorch module object.

3. **Provider selection with CPU fallback**:
   - **Rationale**: Provides graceful degradation when CUDA is not available, allowing the same code to work in CPU-only environments.
   - **Implementation**: Use providers list `['CUDAExecutionProvider', 'CPUExecutionProvider']` to prioritize GPU but fallback to CPU.

4. **Metadata validation**:
   - **Rationale**: Export scripts store `seq_len`, `bg_mode`, `batch_mode`, `batch_size` in ONNX model metadata. Validating these at load time catches configuration mismatches early with clear error messages.
   - **Implementation**: Load metadata from `session.get_modelmeta().custom_metadata_map` and compare against constructor parameters.

5. **Remove AMP-related timing statistics**:
   - **Rationale**: ONNX Runtime uses graph optimization levels (`ORT_ENABLE_ALL`) for performance, not PyTorch's AMP. Timing statistics related to AMP are not applicable.
   - **Implementation**: Keep all other timing stats (t_preprocess, t_forward, t_post, t_median) but remove use_amp tracking.

---

## Task Breakdown

```
┌─────────────────────────────────────────────────────────────────┐
│                       Task Dependencies                         │
└─────────────────────────────────────────────────────────────────┘

Task 0: Update pyproject.toml (Add onnxruntime-gpu)
    │
    ├───────────────────────────────────────────────┐
    │                                               │
Task 1: Create TrackNetModuleONNX                  Task 2: Create InpaintModuleONNX
    │                                               │
    └───────────────────────────────────────────────┘
                         │
                    Task 3: Update __init__.py exports
```

### Parallelization Guide
| Task Group | Tasks | Rationale for Parallelism |
|------------|-------|---------------------------|
| Group A    | 1, 2  | Independent module implementations - can be developed in parallel once dependency (Task 0) is complete. |
| Group B    | 3     | Must wait for both modules to be implemented and verified before updating exports. |

### Dependency Mapping
| Task | Depends On | Reason for Dependency |
|------|------------|----------------------|
| 1    | 0          | Requires `onnxruntime-gpu` to be installed for imports and testing. |
| 2    | 0          | Requires `onnxruntime-gpu` to be installed for imports and testing. |
| 3    | 1, 2       | Both modules must exist and be importable before updating exports. |

---

## TODOs

### Task 0: Update pyproject.toml to add onnxruntime-gpu dependency

**Objective**: Add `onnxruntime-gpu>=1.19.0` to the dependencies list in `pyproject.toml`.

**Implementation Steps**:
1. Open `pyproject.toml`
2. Locate the `dependencies` array (lines 28-44)
3. Add `"onnxruntime-gpu>=1.19.0"` to the list
4. Maintain alphabetical order (currently dependencies are not strictly alphabetical, but place logically near `onnx>=1.19.0`)

**Parallelizable**: NO (This is the foundation task that must complete first)

**References (CRITICAL)**:
> Exhaustively list reference points from the Draft or implied project context. Explain **WHAT** to extract and **WHY** it's relevant.

- **Dependency Reference**: `pyproject.toml:28-44` - Current dependencies list where new entry should be added.
- **Version Requirement**: `Draft:76` - Specifies `onnxruntime-gpu = ">=1.19.0"` as the required version.
- **Existing ONNX dependency**: `pyproject.toml:32` - Already has `onnx>=1.19.0`, which is used for model manipulation, not inference. Place `onnxruntime-gpu` near this entry.

**Acceptance Criteria**:

**For Manual Verification**:
- **Configuration/Infra**:
  - [ ] Verify: `grep "onnxruntime-gpu" pyproject.toml` outputs a line containing `onnxruntime-gpu>=1.19.0`
  - [ ] Verify: The dependency is in the `dependencies` array (not `optional-dependencies` or `dev`)
  - [ ] Verify: Run `pip install -e .` completes successfully without errors

**Evidence Capture**:
- [ ] Terminal output of `grep "onnxruntime" pyproject.toml` showing both `onnx>=1.19.0` and `onnxruntime-gpu>=1.19.0`
- [ ] Terminal output of `pip install -e .` showing successful installation
- [ ] Terminal output of `python -c "import onnxruntime; print(onnxruntime.__version__)"` confirming import works

**Commit Guidance**:
- **Commit Suggested**: YES (Can be standalone)
- **Message**: `deps: add onnxruntime-gpu>=1.19.0 for ONNX inference`
- **Files**: `pyproject.toml`

---

### Task 1: Create TrackNetModuleONNX class in streaming_onnx.py

**Objective**: Implement `TrackNetModuleONNX` class in new file `tracknetv3/inference/streaming_onnx.py`, maintaining identical API to `TrackNetModule` but using ONNX Runtime for inference.

**Implementation Steps**:
1. Create file `tracknetv3/inference/streaming_onnx.py` with proper imports
2. Define `TrackNetModuleONNX` class with docstring matching PyTorch version
3. Implement `__init__` method:
   - Accept `model_path: str` (instead of `tracknet: torch.nn.Module`)
   - Accept all other parameters: `seq_len`, `bg_mode`, `device`, `eval_mode`, `median`, `median_warmup`
   - Store `device` as provider selection hint but convert to appropriate provider list
   - Initialize ONNX Runtime session with provider configuration and optimization
   - Cache input/output tensor names from model
   - Validate model metadata (seq_len, bg_mode) matches constructor parameters
   - Initialize all instance attributes (deques, scalers, ensemble weights, stats)
4. Implement `_ensure_scaler` method (identical to PyTorch version)
5. Implement `_maybe_build_median` method (identical to PyTorch version)
6. Implement `_heatmap_to_xy` method (identical to PyTorch version)
7. Implement `_process_one` method (identical to PyTorch version)
8. Implement `push` method with ONNX Runtime inference:
   - Preprocess frame using `_process_one`
   - Build input tensor for ONNX (NumPy array, no PyTorch conversion)
   - Call `session.run(self.output_names, {self.input_name: x_np})`
   - Remove AMP context and CUDA synchronization
   - Remove `.float().detach().cpu().numpy()` conversion (already NumPy)
   - Keep ensemble weight accumulation logic identical
9. Implement `reset` method (identical to PyTorch version)
10. Implement `flush` method (identical to PyTorch version)

**Parallelizable**: YES
- **Can run concurrently with**: Task 2 (InpaintModuleONNX implementation)

**References (CRITICAL)**:
> Exhaustively list reference points from the Draft or implied project context. Explain **WHAT** to extract and **WHY** it's relevant.

- **PyTorch TrackNetModule Structure**: `tracknetv3/inference/streaming.py:15-311` - Complete reference implementation including all methods, preprocessing, ensemble logic, deque management.
- **Constants**: `tracknetv3/config/constants.py:6-7` - `HEIGHT=288`, `WIDTH=512` used in preprocessing and coordinate conversion.
- **Ensemble Weights**: `tracknetv3/evaluation/ensemble.py:8-33` - `get_ensemble_weight()` function used in `__init__` to compute ensemble weights.
- **ONNX Session Setup**: `Draft:83-99` - Provider selection and session configuration code example.
- **ONNX Input Shapes**: `Draft:52-59` - Input dimensions based on `bg_mode` (critical for validation).
- **ONNX Metadata Keys**: `Draft:61-68` - Metadata stored by export scripts (model_name, seq_len, bg_mode, batch_mode, batch_size).
- **Stats Structure**: `tracknetv3/inference/streaming.py:57-64` - Statistics dictionary structure (remove use_amp from ONNX version).

**Acceptance Criteria**:

**For Manual Verification**:

**Library/Module**:
- [ ] Verify: Run `python -c "from tracknetv3.inference.streaming_onnx import TrackNetModuleONNX; print('Import successful')"` - Should output "Import successful" without errors.
- [ ] Verify: Check class has all required methods: `__init__`, `push`, `flush`, `reset`, `_ensure_scaler`, `_maybe_build_median`, `_heatmap_to_xy`, `_process_one`

**API Compatibility**:
- [ ] Verify: Check `__init__` signature matches expected: `TrackNetModuleONNX(model_path, seq_len, bg_mode, device, eval_mode='weight', median=None, median_warmup=0)`

**Evidence Capture**:
- [ ] Terminal output of import verification
- [ ] Screenshot or output of `python -c "from tracknetv3.inference.streaming_onnx import TrackNetModuleONNX; import inspect; print(inspect.signature(TrackNetModuleONNX.__init__))"`
- [ ] Output of `dir(TrackNetModuleONNX)` showing all required methods exist

**Commit Guidance**:
- **Commit Suggested**: YES (Can be standalone)
- **Message**: `feat(inference): add TrackNetModuleONNX with ONNX Runtime inference`
- **Files**: `tracknetv3/inference/streaming_onnx.py` (new file)

---

### Task 2: Create InpaintModuleONNX class in streaming_onnx.py

**Objective**: Implement `InpaintModuleONNX` class in `tracknetv3/inference/streaming_onnx.py`, maintaining identical API to `InpaintModule` but using ONNX Runtime for inference.

**Implementation Steps**:
1. Add `InpaintModuleONNX` class to existing `streaming_onnx.py` file
2. Define class with docstring matching PyTorch version
3. Implement `__init__` method:
   - Accept `model_path: str` (instead of `inpaintnet: torch.nn.Module`)
   - Accept parameters: `seq_len`, `device`, `img_scaler`
   - Store `device` as provider selection hint
   - Initialize ONNX Runtime session with provider configuration and optimization
   - Cache input tensor names (2 inputs: "coords", "mask") and output name
   - Validate model metadata (seq_len) matches constructor parameter
   - Initialize instance attributes (deques, stats)
4. Implement `reset` method (identical to PyTorch version)
5. Implement `push` method with ONNX Runtime inference:
   - Normalize input coordinates to [0, 1] range
   - Create mask for occlusions (m=1 if occluded, m=0 if visible)
   - Build input tensors as NumPy arrays
   - Call `session.run(self.output_names, {self.input_names_coords: coords_np, self.input_names_mask: mask_np})`
   - Remove device transfers (`.to(self.device)`)
   - Remove AMP context and CUDA synchronization
   - Remove `.float().cpu().numpy()` conversion (already NumPy)
   - Apply post-processing mask: `out = out * mask + coords * (1.0 - mask)`
   - Denormalize coordinates back to pixel space
6. Implement `flush` method (identical to PyTorch version)

**Parallelizable**: YES
- **Can run concurrently with**: Task 1 (TrackNetModuleONNX implementation)

**References (CRITICAL)**:
> Exhaustively list reference points from the Draft or implied project context. Explain **WHAT** to extract and **WHY** it's relevant.

- **PyTorch InpaintModule Structure**: `tracknetv3/inference/streaming.py:313-416` - Complete reference implementation including all methods, coordinate normalization, mask creation, post-processing.
- **Constants**: `tracknetv3/config/constants.py:6-7` - `HEIGHT=288`, `WIDTH=512` used in coordinate denormalization.
- **ONNX Input/Output**: `Draft:63-66` - Input shapes: coords `[batch, seq_len, 2]`, mask `[batch, seq_len, 1]`, output `[batch, seq_len, 2]`.
- **ONNX Input Names**: `tools/export_inpaintnet_onnx.py:42` - Input names are "coords" and "mask".
- **ONNX Metadata Keys**: `Draft:68` - Metadata stored by export scripts (model_name, seq_len, batch_mode, batch_size).
- **Post-processing Logic**: `tracknetv3/inference/streaming.py:387` - Critical line: `out = out * mask + coor * (1.0 - mask)` maintains original coords for masked positions.

**Acceptance Criteria**:

**For Manual Verification**:

**Library/Module**:
- [ ] Verify: Run `python -c "from tracknetv3.inference.streaming_onnx import InpaintModuleONNX; print('Import successful')"` - Should output "Import successful" without errors.
- [ ] Verify: Check class has all required methods: `__init__`, `push`, `flush`, `reset`

**API Compatibility**:
- [ ] Verify: Check `__init__` signature matches expected: `InpaintModuleONNX(model_path, seq_len, device, img_scaler)`

**Evidence Capture**:
- [ ] Terminal output of import verification
- [ ] Screenshot or output of `python -c "from tracknetv3.inference.streaming_onnx import InpaintModuleONNX; import inspect; print(inspect.signature(InpaintModuleONNX.__init__))"`
- [ ] Output of `dir(InpaintModuleONNX)` showing all required methods exist

**Commit Guidance**:
- **Commit Suggested**: YES (Can be standalone)
- **Message**: `feat(inference): add InpaintModuleONNX with ONNX Runtime inference`
- **Files**: `tracknetv3/inference/streaming_onnx.py`

---

### Task 3: Update tracknetv3/inference/__init__.py to export ONNX modules

**Objective**: Add `TrackNetModuleONNX` and `InpaintModuleONNX` to the exports in `tracknetv3/inference/__init__.py`.

**Implementation Steps**:
1. Open `tracknetv3/inference/__init__.py`
2. Add import statement: `from .streaming_onnx import TrackNetModuleONNX, InpaintModuleONNX`
3. Add `"TrackNetModuleONNX"` and `"InpaintModuleONNX"` to `__all__` list
4. Maintain alphabetical or logical ordering (currently PyTorch modules are last, add ONNX modules after)

**Parallelizable**: NO (Must wait for Tasks 1 and 2 to complete)

**References (CRITICAL)**:
> Exhaustively list reference points from the Draft or implied project context. Explain **WHAT** to extract and **WHY** it's relevant.

- **Current Exports**: `tracknetv3/inference/__init__.py:1-15` - Existing import and export structure.
- **Export Pattern**: Line 5 shows import from `.streaming`, line 12-13 shows `__all__` entries. Follow same pattern for `.streaming_onnx`.

**Acceptance Criteria**:

**For Manual Verification**:

**Library/Module**:
- [ ] Verify: Run `python -c "from tracknetv3.inference import TrackNetModuleONNX, InpaintModuleONNX; print('Exports successful')"` - Should output "Exports successful" without errors.
- [ ] Verify: Run `python -c "import tracknetv3.inference as inf; print(sorted(inf.__all__))"` - Output should include both `TrackNetModuleONNX` and `InpaintModuleONNX`
- [ ] Verify: Run `python -c "from tracknetv3 import inference; print(hasattr(inference, 'TrackNetModuleONNX'), hasattr(inference, 'InpaintModuleONNX'))"` - Should output `True True`

**Evidence Capture**:
- [ ] Terminal output of all three verification commands above
- [ ] Screenshot of updated `__init__.py` file showing new imports and exports

**Commit Guidance**:
- **Commit Suggested**: YES (Standalone or grouped with final verification)
- **Message**: `feat(inference): export TrackNetModuleONNX and InpaintModuleONNX from __init__`
- **Files**: `tracknetv3/inference/__init__.py`

---

## Success Criteria & Final Verification

### Final Integration Test

```bash
# 1. Verify all imports work
python -c "
from tracknetv3.inference import TrackNetModule, TrackNetModuleONNX, InpaintModule, InpaintModuleONNX
print('All imports successful')
print('PyTorch TrackNetModule:', TrackNetModule)
print('ONNX TrackNetModuleONNX:', TrackNetModuleONNX)
print('PyTorch InpaintModule:', InpaintModule)
print('ONNX InpaintModuleONNX:', InpaintModuleONNX)
"
# Expected: All imports successful and class names printed

# 2. Verify API compatibility (signatures match)
python -c "
import inspect
from tracknetv3.inference import TrackNetModule, TrackNetModuleONNX, InpaintModule, InpaintModuleONNX

# Check TrackNet signatures
tn_sig = inspect.signature(TrackNetModule.__init__)
tn_onnx_sig = inspect.signature(TrackNetModuleONNX.__init__)
tn_params = list(tn_sig.parameters.keys())[1:]  # Skip 'self'
tn_onnx_params = list(tn_onnx_sig.parameters.keys())[1:]
print('TrackNet parameters:', tn_params)
print('TrackNetONNX parameters:', tn_onnx_params)

# Check Inpaint signatures
inp_sig = inspect.signature(InpaintModule.__init__)
inp_onnx_sig = inspect.signature(InpaintModuleONNX.__init__)
inp_params = list(inp_sig.parameters.keys())[1:]  # Skip 'self'
inp_onnx_params = list(inp_onnx_sig.parameters.keys())[1:]
print('Inpaint parameters:', inp_params)
print('InpaintONNX parameters:', inp_onnx_params)

# Check that ONNX versions have same params except first one
assert tn_onnx_params == tn_params, 'TrackNetONNX params mismatch'
assert inp_onnx_params == inp_params, 'InpaintONNX params mismatch'
print('API compatibility verified!')
"
# Expected: Parameter lists match (except first param name: model_path vs module)

# 3. Verify ruff passes (no new errors)
ruff check tracknetv3/inference/streaming_onnx.py tracknetv3/inference/__init__.py
# Expected: No output or existing errors only, no new errors
```

### Project Completion Checklist
- [ ] All "Core Deliverables" are implemented:
  - [ ] `tracknetv3/inference/streaming_onnx.py` exists with `TrackNetModuleONNX`
  - [ ] `tracknetv3/inference/streaming_onnx.py` exists with `InpaintModuleONNX`
  - [ ] `pyproject.toml` includes `onnxruntime-gpu>=1.19.0`
  - [ ] `tracknetv3/inference/__init__.py` exports both ONNX modules
- [ ] All constraints have been respected:
  - [ ] API signatures match PyTorch versions (except first parameter)
  - [ ] ONNX Runtime session created with provider selection
  - [ ] Metadata validation implemented in `__init__`
  - [ ] AMP-related timing statistics removed from ONNX versions
  - [ ] `model_path: str` used instead of `torch.nn.Module`
- [ ] Nothing in the "Out of Scope" section has been implemented:
  - [ ] No ONNX export scripts created
  - [ ] No custom ONNX optimizations beyond `ORT_ENABLE_ALL`
  - [ ] No multi-GPU support added
  - [ ] No dynamic sequence length support at runtime
  - [ ] No model quantization/int8 support
- [ ] All verification steps (manual) defined in the TODOs have passed:
  - [ ] Task 0: Dependency added and installable
  - [ ] Task 1: TrackNetModuleONNX imports and has correct API
  - [ ] Task 2: InpaintModuleONNX imports and has correct API
  - [ ] Task 3: Exports work from package level
- [ ] Final integration test passes successfully
