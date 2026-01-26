# Implementation Plan: ONNX-based Inference Modules for TrackNetV3

## Context

### Source Document
- **Draft**: `.agent/drafts/onnx-inference-modules-v2.0.md`
- **Objective**: Create ONNX Runtime-based inference modules (`TrackNetModuleONNX` and `InpaintModuleONNX`) that mirror the functionality of the existing PyTorch-based modules in `tracknetv3/inference/streaming.py`.

### Summary of Requirements
- **Core Deliverables**:
  - New file `tracknetv3/inference/streaming_onnx.py` with two classes
  - `TrackNetModuleONNX`: ONNX Runtime-based version of `TrackNetModule`
  - `InpaintModuleONNX`: ONNX Runtime-based version of `InpaintModule`
  - Dependency addition: `onnxruntime-gpu` to `pyproject.toml`
  - Export update: Modify `tracknetv3/inference/__init__.py` to export new classes

- **Key Constraints**:
  - Use `onnxruntime-gpu` for GPU acceleration with CPU fallback
  - Maintain identical API signatures to PyTorch versions (except first parameter: `model_path: str` instead of module object)
  - Return types and formats must match PyTorch versions exactly
  - Error handling: `ValueError` for invalid paths/metadata, `RuntimeError` for provider issues
  - Batch dimension handling must be correct for ONNX Runtime
  - NumPy array conversions for ensemble weights

- **Out of Scope**:
  - ONNX model export scripts (already exist in `tools/`)
  - ONNX optimization beyond default `ORT_ENABLE_ALL`
  - Multi-GPU support (default to device_id: 0)
  - Dynamic sequence length changes (fixed seq_len as exported)
  - Model quantization/int8 (FP32 only)

---

## Work Objectives & Verification Strategy

### Definition of Done
The project is complete when:
- [ ] `streaming_onnx.py` is created with both ONNX-based modules
- [ ] ONNX modules produce identical outputs to PyTorch versions (within floating-point tolerance)
- [ ] All background modes work correctly (`""`, `"subtract"`, `"subtract_concat"`, `"concat"`)
- [ ] Error handling follows specified exception types and message formats
- [ ] Batch dimension handling is correct for both modules
- [ ] Provider selection works (CUDA when available, CPU fallback)
- [ ] Integration tests verify API compatibility including first parameter difference

### Verification Approach
**Decision**:
- **Test Infrastructure Exists**: UNKNOWN - No explicit test infrastructure mentioned in Draft
- **Testing Mandate**: Manual QA Only (Draft specifies functional verification steps but no automated test framework)
- **Framework/Tools**: Python REPL for basic verification, manual inspection for outputs

**For Manual QA**:
- Each task MUST include explicit, step-by-step manual verification procedures
- Evidence Required: Command outputs, API responses must be documented
- Functional verification steps for each module are explicitly defined in Draft

---

## Architecture & Design

### Architectural Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                              │
│  (TrackNetModuleONNX, InpaintModuleONNX - streaming_onnx.py)      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ONNX Runtime API Layer                          │
│  (InferenceSession, provider selection, NumPy I/O)               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Hardware Execution Layer                         │
│  (CUDAExecutionProvider or CPUExecutionProvider)                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow for TrackNetModuleONNX

```
BGR Frame Input
       │
       ▼
┌──────────────┐
│  Preprocess │  (NumPy operations, identical to PyTorch version)
│  _process_one│  → Returns [in_dim, HEIGHT, WIDTH] NumPy array
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Accumulate   │  Buffer seq_len processed frames
└──────┬───────┘
       │
       ▼
┌────────────────────────┐
│ Concatenate frames    │  → [in_dim * seq_len, HEIGHT, WIDTH]
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐
│ Add batch dimension    │  → [1, in_dim * seq_len, HEIGHT, WIDTH]
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐
│ ONNX Runtime Inference │  session.run({input_name: x_np})
│  (GPU or CPU)          │  → Output: [1, seq_len, HEIGHT, WIDTH]
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐
│ Remove batch dim       │  → [seq_len, HEIGHT, WIDTH]
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐
│ Ensemble accumulation │  Weighted averaging with ensemble weights
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐
│ Heatmap to coords      │  _heatmap_to_xy() → {"Frame", "X", "Y", "Visibility"}
└────────────────────────┘
```

### Data Flow for InpaintModuleONNX

```
TrackNet Prediction
       │
       ▼
┌────────────────────────┐
│ Normalize coords       │  → [0, 1] range
│ Create mask            │  1.0 for occluded, 0.0 for visible
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐
│ Accumulate in deque   │  Buffer seq_len coords and masks
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐
│ Add batch dimensions   │  coords: [seq_len, 2] → [1, seq_len, 2]
│                        │  mask: [seq_len, 1] → [1, seq_len, 1]
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐
│ ONNX Runtime Inference │  session.run({coords: ..., mask: ...})
│  (GPU or CPU)          │  → Output: [1, seq_len, 2]
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐
│ Remove batch dim       │  → [seq_len, 2]
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐
│ Apply mask & denorm     │  Refine coords, convert back to pixels
└────────────────────────┘
```

### Key Architectural Decisions

1. **ONNX Runtime Integration with Provider Selection**
   - **Decision**: Use `onnxruntime.InferenceSession` with explicit provider list `['CUDAExecutionProvider', 'CPUExecutionProvider']`
   - **Rationale**: Ensures GPU acceleration when available with graceful CPU fallback. ONNX Runtime has different device management than PyTorch, so we use provider selection instead of device tensors.
   - **Implementation Reference**: [Draft lines 89-99]

2. **Model Path vs Module Parameter**
   - **Decision**: Accept `model_path: str` as first `__init__` parameter instead of `torch.nn.Module` object
   - **Rationale**: ONNX models are loaded from disk, not instantiated as in-memory objects. This maintains API compatibility while adapting to ONNX Runtime's loading mechanism.
   - **Implementation Reference**: [Draft lines 106-127]

3. **NumPy Array Conversions for Ensemble Weights**
   - **Decision**: Convert `get_ensemble_weight()` output (torch.Tensor) to `np.ndarray` with explicit float32 dtype
   - **Rationale**: ONNX Runtime expects NumPy arrays, not PyTorch tensors. The pattern must match the existing conversion in PyTorch version at `streaming.py:44-46`.
   - **Implementation Reference**: [Draft lines 297-305]

4. **Batch Dimension Handling**
   - **Decision**: Add batch dimension `[None, ...]` before inference and remove with `[0]` after
   - **Rationale**: ONNX models expect batched inputs even for single-frame inference. This maintains compatibility with exported ONNX models.
   - **Implementation Reference**: [Draft lines 308-329]

5. **Error Handling Specification**
   - **Decision**: Use specific exception types with formatted messages
     - `ValueError` for invalid paths or metadata mismatches
     - `RuntimeError` for provider availability issues
   - **Rationale**: Provides clear, actionable error messages for debugging and user guidance.
   - **Implementation Reference**: [Draft lines 336-352]

---

## Task Breakdown

```
Task 0 (Foundation)
    ├── Task 1 (TrackNetModuleONNX)
    └── Task 2 (InpaintModuleONNX)
            ├── Task 3 (Dependency Update)
            └── Task 4 (Integration & Export)
```

### Parallelization Guide
| Task Group | Tasks | Rationale for Parallelism |
|------------|-------|---------------------------|
| Group A    | 1, 2  | Modify independent modules; no shared dependencies. |
| Group B    | 3     | Single dependency update task. |

### Dependency Mapping
| Task | Depends On | Reason for Dependency |
|------|------------|----------------------|
| 3    | 0, 1, 2    | Requires all modules to verify dependency works. |
| 4    | 1, 2, 3    | Integration test needs both modules and updated dependencies. |

---

## TODOs

### Task 0: Foundation - Project Setup and File Creation

**Objective**: Create the new file `tracknetv3/inference/streaming_onnx.py` with basic structure and imports.

**Implementation Steps**:
1. Create the file `tracknetv3/inference/streaming_onnx.py`
2. Add necessary imports:
   - Standard library: `time`, `deque` from `collections`, `Any` from `typing`, `os`
   - Third-party: `cv2`, `numpy as np`, `onnxruntime as ort`
   - Project: `from tracknetv3.config.constants import HEIGHT, WIDTH`
   - Project: `from tracknetv3.evaluation.ensemble import get_ensemble_weight`
3. Add module-level docstring describing ONNX-based inference modules
4. Create placeholder class stubs for `TrackNetModuleONNX` and `InpaintModuleONNX` with docstrings from PyTorch versions

**Parallelizable**: NO (Must be completed before dependent tasks)

**References (CRITICAL)**:
- **Import Pattern Reference**: [`streaming.py:1-12`] - Copy import structure, replace `torch` with `onnxruntime`
- **Docstring Reference**: [`streaming.py:15-20`] - Use identical docstrings for ONNX versions
- **Constants Reference**: [`constants.py:6-7`] - HEIGHT=288, WIDTH=512 for coordinate normalization

**Acceptance Criteria**:

**For Manual Verification**:
- **File Creation**:
  - [ ] Run: `ls -la tracknetv3/inference/streaming_onnx.py`
  - [ ] Verify: File exists and is readable
- **Import Validation**:
  - [ ] In REPL: `import tracknetv3.inference.streaming_onnx`
  - [ ] Verify: No ImportError, module loads successfully
  - [ ] Run: `print(dir(tracknetv3.inference.streaming_onnx))`
  - [ ] Verify: Output contains `['TrackNetModuleONNX', 'InpaintModuleONNX']`
- **ONNX Runtime Check**:
  - [ ] In REPL: `import onnxruntime as ort; print(ort.__version__)`
  - [ ] Verify: Version >= 1.19.0 is installed

**Evidence Capture**:
- [ ] Terminal output showing successful module import
- [ ] Dir listing of exported names

**Commit Guidance**:
- **Commit Suggested**: NO (Group with Task 1 and Task 2)
- **Message**: `feat(inference): add ONNX-based inference modules with basic structure`
- **Files**: `tracknetv3/inference/streaming_onnx.py`

---

### Task 1: Implement TrackNetModuleONNX

**Objective**: Implement full `TrackNetModuleONNX` class with ONNX Runtime inference, error handling, and functional verification.

**Implementation Steps**:

#### Step 1.1: Constructor Implementation
1. Define `__init__` with parameters:
   - `model_path: str` (first parameter, replaces `tracknet: torch.nn.Module`)
   - `seq_len: int`
   - `bg_mode: str`
   - `device` (for API compatibility, used for provider selection)
   - `eval_mode: str = "weight"`
   - `median: np.ndarray | None = None`
   - `median_warmup: int = 0`
2. **Error handling for model path**:
   - Check `os.path.exists(model_path)`
   - If not exists, raise `ValueError(f"ONNX model file not found: {model_path}")`
3. **Load ONNX session**:
   - Create `ort.SessionOptions()` with `graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL`
   - Set `intra_op_num_threads = 1` for GPU optimization
   - Create provider list: `providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']`
   - Initialize `session = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)`
   - **Error handling for providers**: If `session.get_providers()` is empty, raise `RuntimeError("No ONNX Runtime execution providers available")`
4. **Cache input/output names**:
   - Set `self.input_name = session.get_inputs()[0].name` (should be `"input"`)
   - Set `self.output_names = [output.name for output in session.get_outputs()]` (should be `["output"]`)
5. **Validate metadata**:
   - Extract metadata from model: `model_metadata = session.get_modelmeta().custom_metadata_map`
   - Validate `seq_len`:
     - If mismatch: raise `ValueError(f"Model metadata mismatch: expected seq_len={seq_len}, got {model_metadata.get('seq_len')}. Model={model_path}")`
   - Validate `bg_mode` (if not empty):
     - If mismatch: raise `ValueError(f"Model metadata mismatch: expected bg_mode='{bg_mode}', got '{model_metadata.get('bg_mode')}'. Model={model_path}")`
6. Initialize instance attributes (identical to PyTorch version):
   - `self.seq_len`, `self.bg_mode`, `self.eval_mode`, `self.device`, `self._median`, `self._median_warmup`
   - Initialize deques: `self.frames`, `self.frame_ids`, `self._proc`, `self._fidq`
   - Set `self.img_scaler`, `self.img_shape` to `None`
7. **NumPy ensemble weight conversion** (CRITICAL):
   - Convert torch tensor to NumPy array:
     ```python
     self._ens_w = np.asarray(
         get_ensemble_weight(self.seq_len, self.eval_mode).cpu().numpy(),
         dtype=np.float32
     )
     ```
8. Initialize ensemble accumulators: `self._acc_sum`, `self._acc_w`
9. Initialize warmup list and stats dict (without AMP timing)
10. Set `self._count = 0`

#### Step 1.2: Copy Unchanged Methods
1. Copy `reset()` method from [`streaming.py:68-76`] (identical)
2. Copy `_ensure_scaler()` method from [`streaming.py:78-82`] (identical)
3. Copy `_maybe_build_median()` method from [`streaming.py:84-104`] (identical)
4. Copy `_heatmap_to_xy()` method from [`streaming.py:170-181`] (identical)
5. Copy `_process_window()` method from [`streaming.py:106-168`] (identical)
6. Copy `_process_one()` method from [`streaming.py:183-219`] (identical)
7. Copy `flush()` method from [`streaming.py:293-310`] (identical)

#### Step 1.3: Implement ONNX-based `push()` Method
1. Keep identical preprocessing logic up to `x_np` formation (lines 221-246 from PyTorch version)
2. **Replace PyTorch tensor creation and inference**:
   ```python
   # OLD (PyTorch):
   # x = torch.from_numpy(x_np).unsqueeze(0).to(self.device, non_blocking=True)
   # t1 = time.perf_counter()
   # use_amp = self.device.type == "cuda"
   # with (torch.inference_mode(), torch.amp.autocast(device_type=self.device.type, enabled=use_amp)):
   #     y = self.tracknet(x)
   # if self.device.type == "cuda":
   #     torch.cuda.synchronize()
   # self.stats["t_forward"] += time.perf_counter() - t1
   # t2 = time.perf_counter()
   # y = y.float().detach().cpu().numpy()[0]

   # NEW (ONNX Runtime):
   # Add batch dimension - CRITICAL
   ort_inputs = {self.input_name: x_np[None, ...]}  # Shape: [1, in_dim*seq_len, H, W]
   t1 = time.perf_counter()
   outputs = self.session.run(self.output_names, ort_inputs)  # Returns list of outputs
   self.stats["t_forward"] += time.perf_counter() - t1
   t2 = time.perf_counter()
   y = outputs[0][0]  # Remove batch dimension, shape: [seq_len, H, W]
   ```
3. Keep identical ensemble accumulation logic (lines 264-273 from PyTorch version)
4. Keep identical output extraction logic (lines 275-291 from PyTorch version, but without AMP-related timing)

**Parallelizable**: NO (Independent module, but foundation must exist)

**References (CRITICAL)**:
- **Constructor Pattern Reference**: [`streaming.py:22-66`] - Identical structure, replace tracknet parameter with model_path
- **Preprocessing Reference**: [`streaming.py:221-246`] - Identical NumPy preprocessing logic
- **Inference Replacement**: [`streaming.py:248-263`] - Replace PyTorch inference with ONNX Runtime
- **Ensemble Logic**: [`streaming.py:264-289`] - Identical ensemble accumulation (uses self._ens_w NumPy array)
- **Batch Dim Handling**: [Draft lines 312-318] - Add `[None, ...]` before inference, `[0]` after
- **Error Handling**: [Draft lines 101-105, 124-126, 339-352] - ValueError for path/metadata, RuntimeError for providers
- **ONNX Input Name**: [`export_tracknet_onnx.py:55`] - Input name is `"input"`
- **Metadata Validation**: [Draft lines 150-158] - Validate seq_len and bg_mode from model metadata

**Acceptance Criteria**:

**For Manual Verification**:

**A. Model Loading and Metadata Validation**:
- [ ] Create test script or use REPL:
  ```python
  import numpy as np
  from tracknetv3.inference.streaming_onnx import TrackNetModuleONNX

  # Test with valid model path (adjust path as needed)
  try:
      module = TrackNetModuleONNX(
          model_path="path/to/tracknet.onnx",  # Replace with actual path
          seq_len=15,
          bg_mode="",
          device=None  # Will use default provider selection
      )
      print("✓ Model loaded successfully")
      print(f"  Input name: {module.input_name}")
      print(f"  Output names: {module.output_names}")
      print(f"  Providers: {module.session.get_providers()}")
  except Exception as e:
      print(f"✗ Failed: {type(e).__name__}: {e}")
  ```
- [ ] Verify: Module instantiates without errors
- [ ] Verify: Output shows input name `"input"` and available providers

**B. Metadata Validation Errors**:
- [ ] Test invalid path:
  ```python
  try:
      TrackNetModuleONNX(model_path="/nonexistent/path.onnx", seq_len=15, bg_mode="", device=None)
  except ValueError as e:
      print(f"✓ Invalid path error: {e}")
  ```
- [ ] Verify: `ValueError` raised with message containing `"ONNX model file not found: /nonexistent/path.onnx"`

- [ ] Test seq_len mismatch (if possible with test model):
  ```python
  try:
      TrackNetModuleONNX(model_path="path/to/tracknet.onnx", seq_len=99, bg_mode="", device=None)
  except ValueError as e:
      print(f"✓ seq_len mismatch error: {e}")
  ```
- [ ] Verify: `ValueError` raised with message containing `"Model metadata mismatch: expected seq_len=99"`

**C. Dummy Inference with Batch Dimension Handling**:
- [ ] Run dummy inference test:
  ```python
  import numpy as np
  from tracknetv3.inference.streaming_onnx import TrackNetModuleONNX

  module = TrackNetModuleONNX(
      model_path="path/to/tracknet.onnx",
      seq_len=15,
      bg_mode="",
      device=None
  )

  # Create dummy frame
  dummy_frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)

  # Push seq_len frames
  for i in range(15):
      result = module.push(dummy_frame, frame_id=i)

  print("✓ Dummy inference completed")
  if result:
      print(f"  Result: {result}")
      print(f"  Keys: {result.keys()}")
      print(f"  Types: {type(result['Frame'])}, {type(result['X'])}, {type(result['Y'])}, {type(result['Visibility'])}")
  ```
- [ ] Verify: No errors during inference
- [ ] Verify: Result dict has keys `["Frame", "X", "Y", "Visibility"]`
- [ ] Verify: All values are integers (not NumPy types)

**D. Provider Selection Verification**:
- [ ] Check available providers:
  ```python
  import onnxruntime as ort
  print("Available providers:", ort.get_available_providers())
  ```
- [ ] Verify: Output contains either `['CUDAExecutionProvider', ...]` or `['CPUExecutionProvider']`
- [ ] Run module and check:
  ```python
  module = TrackNetModuleONNX(model_path="path/to/tracknet.onnx", seq_len=15, bg_mode="", device=None)
  print("Active providers:", module.session.get_providers())
  ```
- [ ] Verify: CUDA provider appears first if available

**E. Error Handling for Provider Issues** (if testable):
- [ ] If environment lacks CUDA, verify graceful CPU fallback without RuntimeError
- [ ] If CUDA available, verify CUDAExecutionProvider is used

**Evidence Capture**:
- [ ] Terminal output from all verification steps
- [ ] Screenshot of provider selection output (if visual verification needed)

**Commit Guidance**:
- **Commit Suggested**: YES (Group with Task 2)
- **Message**: `feat(inference): implement TrackNetModuleONNX with ONNX Runtime inference`
- **Files**: `tracknetv3/inference/streaming_onnx.py`

---

### Task 2: Implement InpaintModuleONNX

**Objective**: Implement full `InpaintModuleONNX` class with ONNX Runtime inference and functional verification.

**Implementation Steps**:

#### Step 2.1: Constructor Implementation
1. Define `__init__` with parameters:
   - `model_path: str` (first parameter, replaces `inpaintnet: torch.nn.Module`)
   - `seq_len: int`
   - `device` (for API compatibility, used for provider selection)
   - `img_scaler: tuple[float, float]`
2. **Error handling for model path**:
   - Check `os.path.exists(model_path)`
   - If not exists, raise `ValueError(f"ONNX model file not found: {model_path}")`
3. **Load ONNX session** (identical to TrackNetModuleONNX):
   - Create `ort.SessionOptions()` with `graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL`
   - Set `intra_op_num_threads = 1`
   - Create provider list: `providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']`
   - Initialize `session = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)`
   - **Error handling for providers**: If `session.get_providers()` is empty, raise `RuntimeError("No ONNX Runtime execution providers available")`
4. **Cache input/output names** (CRITICAL - InpaintNet has 2 inputs):
   - Set `self.input_coords = session.get_inputs()[0].name` (should be `"coords"`)
   - Set `self.input_mask = session.get_inputs()[1].name` (should be `"mask"`)
   - Set `self.output_names = [output.name for output in session.get_outputs()]` (should be `["output"]`)
5. **Validate metadata**:
   - Extract metadata: `model_metadata = session.get_modelmeta().custom_metadata_map`
   - Validate `seq_len`:
     - If mismatch: raise `ValueError(f"Model metadata mismatch: expected seq_len={seq_len}, got {model_metadata.get('seq_len')}. Model={model_path}")`
6. Initialize instance attributes:
   - `self.seq_len`, `self.device`, `self.img_scaler`
   - Initialize deques: `self._coords`, `self._mask`, `self._frame_ids`
   - Initialize stats dict (without AMP timing)
7. Set `self._count = 0`

#### Step 2.2: Copy Unchanged Methods
1. Copy `reset()` method from [`streaming.py:338-341`] (identical)
2. Copy `flush()` method from [`streaming.py:402-416`] (identical)

#### Step 2.3: Implement ONNX-based `push()` Method
1. Keep identical input preprocessing logic (lines 343-365 from PyTorch version)
2. **Replace PyTorch tensor creation and inference**:
   ```python
   # OLD (PyTorch):
   # coor = torch.tensor(self._coords, dtype=torch.float32).unsqueeze(0)
   # mask = torch.tensor(self._mask, dtype=torch.float32).unsqueeze(0)
   # t1 = time.perf_counter()
   # use_amp = self.device.type == "cuda"
   # with (torch.inference_mode(), torch.amp.autocast(device_type=self.device.type, enabled=use_amp)):
   #     out = self.inpaintnet(coor.to(self.device), mask.to(self.device))
   # if self.device.type == "cuda":
   #     torch.cuda.synchronize()
   # self.stats["t_forward"] += time.perf_counter() - t1

   # NEW (ONNX Runtime):
   coords_np = np.array(self._coords, dtype=np.float32)  # Shape: [seq_len, 2]
   mask_np = np.array(self._mask, dtype=np.float32)      # Shape: [seq_len, 1]

   # Add batch dimension - CRITICAL
   ort_inputs = {
       self.input_coords: coords_np[None, ...],  # → [1, seq_len, 2]
       self.input_mask: mask_np[None, ...]        # → [1, seq_len, 1]
   }
   t1 = time.perf_counter()
   outputs = self.session.run(self.output_names, ort_inputs)  # Returns list
   self.stats["t_forward"] += time.perf_counter() - t1
   t2 = time.perf_counter()

   out = outputs[0][0]  # Remove batch dimension, shape: [seq_len, 2]
   ```
3. Replace postprocessing (PyTorch tensor operations to NumPy):
   ```python
   # OLD (PyTorch):
   # t2 = time.perf_counter()
   # out = out.float().cpu()
   # out = out * mask + coor * (1.0 - mask)
   # out_np = out.numpy()[0]

   # NEW (NumPy):
   # Apply mask and coordinate blending
   out = out * mask_np + coords_np * (1.0 - mask_np)  # Shape: [seq_len, 2]
   ```
4. Keep identical output denormalization and format (lines 390-400 from PyTorch version, but remove AMP-related timing)

**Parallelizable**: YES (Can run concurrently with Task 1)

**References (CRITICAL)**:
- **Constructor Pattern Reference**: [`streaming.py:320-336`] - Identical structure, replace inpaintnet parameter with model_path
- **Input Preprocessing**: [`streaming.py:343-365`] - Identical coordinate normalization and mask creation
- **Inference Replacement**: [`streaming.py:370-383`] - Replace PyTorch inference with ONNX Runtime (2 inputs)
- **Batch Dim Handling**: [Draft lines 320-329] - Add `[None, ...]` to both inputs, `[0]` to output
- **Error Handling**: [Draft lines 101-105, 124-126, 339-352] - ValueError for path/metadata, RuntimeError for providers
- **ONNX Input Names**: [`export_inpaintnet_onnx.py:42`] - Input names are `"coords"` and `"mask"`
- **Metadata Validation**: [Draft lines 150-158] - Validate seq_len from model metadata (InpaintNet has no bg_mode)

**Acceptance Criteria**:

**For Manual Verification**:

**A. Model Loading and Metadata Validation**:
- [ ] Create test script or use REPL:
  ```python
  import numpy as np
  from tracknetv3.inference.streaming_onnx import InpaintModuleONNX

  # Test with valid model path (adjust path as needed)
  try:
      module = InpaintModuleONNX(
          model_path="path/to/inpaintnet.onnx",  # Replace with actual path
          seq_len=15,
          device=None,
          img_scaler=(640/512, 360/288)
      )
      print("✓ Model loaded successfully")
      print(f"  Coords input name: {module.input_coords}")
      print(f"  Mask input name: {module.input_mask}")
      print(f"  Output names: {module.output_names}")
      print(f"  Providers: {module.session.get_providers()}")
  except Exception as e:
      print(f"✗ Failed: {type(e).__name__}: {e}")
  ```
- [ ] Verify: Module instantiates without errors
- [ ] Verify: Input names are `"coords"` and `"mask"`

**B. Metadata Validation Errors**:
- [ ] Test invalid path:
  ```python
  try:
      InpaintModuleONNX(model_path="/nonexistent/path.onnx", seq_len=15, device=None, img_scaler=(1.0, 1.0))
  except ValueError as e:
      print(f"✓ Invalid path error: {e}")
  ```
- [ ] Verify: `ValueError` raised with message containing `"ONNX model file not found: /nonexistent/path.onnx"`

- [ ] Test seq_len mismatch (if possible with test model):
  ```python
  try:
      InpaintModuleONNX(model_path="path/to/inpaintnet.onnx", seq_len=99, device=None, img_scaler=(1.0, 1.0))
  except ValueError as e:
      print(f"✓ seq_len mismatch error: {e}")
  ```
- [ ] Verify: `ValueError` raised with message containing `"Model metadata mismatch: expected seq_len=99"`

**C. Dummy Inference with Two-Input Handling**:
- [ ] Run dummy inference test:
  ```python
  import numpy as np
  from tracknetv3.inference.streaming_onnx import InpaintModuleONNX

  module = InpaintModuleONNX(
      model_path="path/to/inpaintnet.onnx",
      seq_len=15,
      device=None,
      img_scaler=(640/512, 360/288)
  )

  # Create dummy predictions
  for i in range(15):
      pred = {
          "Frame": i,
          "X": 320 + np.random.randint(-10, 10),
          "Y": 180 + np.random.randint(-10, 10),
          "Visibility": 1
      }
      result = module.push(pred)
      if result is not None:
          break

  print("✓ Dummy inference completed")
  if result:
      print(f"  Result: {result}")
      print(f"  Keys: {result.keys()}")
      print(f"  Types: {type(result['Frame'])}, {type(result['X'])}, {type(result['Y'])}, {type(result['Visibility'])}")
  ```
- [ ] Verify: No errors during inference
- [ ] Verify: Result dict has keys `["Frame", "X", "Y", "Visibility"]`
- [ ] Verify: All values are integers

**D. Two-Input Handling Verification**:
- [ ] Verify both inputs are being passed correctly by inspecting the internal state:
  ```python
  module = InpaintModuleONNX(
      model_path="path/to/inpaintnet.onnx",
      seq_len=5,
      device=None,
      img_scaler=(1.0, 1.0)
  )

  # Push seq_len predictions to buffer
  for i in range(5):
      module.push({"Frame": i, "X": 100, "Y": 100, "Visibility": 1})

  # Check buffered data (before first output)
  print(f"Buffered coords shape: {len(module._coords)}, {len(module._coords[0])}")
  print(f"Buffered mask shape: {len(module._mask)}, {len(module._mask[0])}")
  ```
- [ ] Verify: Coords buffer has 5 elements, each with 2 values
- [ ] Verify: Mask buffer has 5 elements, each with 1 value

**Evidence Capture**:
- [ ] Terminal output from all verification steps
- [ ] Screenshot of two-input verification output

**Commit Guidance**:
- **Commit Suggested**: YES (Group with Task 1)
- **Message**: `feat(inference): implement InpaintModuleONNX with ONNX Runtime inference`
- **Files**: `tracknetv3/inference/streaming_onnx.py`

---

### Task 3: Update Dependencies

**Objective**: Add `onnxruntime-gpu` dependency to `pyproject.toml`.

**Implementation Steps**:
1. Read `pyproject.toml` to locate the `dependencies` section
2. Add `"onnxruntime-gpu>=1.19.0"` to the dependencies list (maintain alphabetical order)
3. Add note in optional dependencies for CPU-only users

**Parallelizable**: YES (Can run concurrently with Tasks 1 and 2)

**References (CRITICAL)**:
- **Dependency Reference**: [`pyproject.toml:28-44`] - Current dependencies list, maintain alphabetical order
- **ONNX Version Reference**: [`pyproject.toml:32`] - `onnx>=1.19.0` exists, add `onnxruntime-gpu` nearby
- **Draft Reference**: [Draft lines 252-257] - Specifies exact version constraint

**Acceptance Criteria**:

**For Manual Verification**:
- **Dependency Addition**:
  - [ ] Run: `grep -n "onnxruntime-gpu" pyproject.toml`
  - [ ] Verify: Line found showing `"onnxruntime-gpu>=1.19.0"`
- **Install Verification**:
  - [ ] Run: `pip install -e .`
  - [ ] Verify: No errors, package installs successfully
  - [ ] Run: `pip list | grep onnxruntime`
  - [ ] Verify: `onnxruntime-gpu` or `onnxruntime` appears in list with version >= 1.19.0

**Evidence Capture**:
- [ ] Terminal output showing `pip install` success
- [ ] Output from `pip list` showing onnxruntime package

**Commit Guidance**:
- **Commit Suggested**: YES
- **Message**: `deps: add onnxruntime-gpu>=1.19.0 for ONNX Runtime GPU support`
- **Files**: `pyproject.toml`

---

### Task 4: Integration Testing and Module Exports

**Objective**: Update `__init__.py` to export new modules and run comprehensive integration tests.

**Implementation Steps**:

#### Step 4.1: Update Module Exports
1. Read `tracknetv3/inference/__init__.py`
2. Add imports for new ONNX modules
3. Add to `__all__` list (if present)
4. Ensure both PyTorch and ONNX versions are available

#### Step 4.2: API Compatibility Verification
1. Create integration test script that:
   - Imports both PyTorch and ONNX modules
   - Compares `__init__` signatures using `inspect.signature()`
   - Verifies first parameter differs (model_path vs tracknet/inpaintnet)
   - Verifies remaining parameters are identical

#### Step 4.3: Functional Integration Test
1. Create test that simulates real usage:
   - Load dummy ONNX models (or mock them if not available)
   - Run TrackNetModuleONNX with sample frames
   - Run InpaintModuleONNX with sample predictions
   - Verify output format matches PyTorch versions

**Parallelizable**: NO (Requires completion of Tasks 1, 2, and 3)

**References (CRITICAL)**:
- **Current Exports**: [`tracknetv3/inference/__init__.py`] - Add ONNX modules alongside PyTorch versions
- **API Compatibility Test**: [Draft lines 362-382] - Explicit integration test to verify first parameter difference
- **Verification Requirements**: [Draft lines 277-294] - Functional verification steps for both modules

**Acceptance Criteria**:

**For Manual Verification**:

**A. Module Exports**:
- [ ] Run:
  ```python
  from tracknetv3.inference import TrackNetModule, InpaintModule
  from tracknetv3.inference import TrackNetModuleONNX, InpaintModuleONNX

  print("✓ All modules imported successfully")
  print(f"  TrackNetModule: {TrackNetModule}")
  print(f"  TrackNetModuleONNX: {TrackNetModuleONNX}")
  print(f"  InpaintModule: {InpaintModule}")
  print(f"  InpaintModuleONNX: {InpaintModuleONNX}")
  ```
- [ ] Verify: All four modules import without errors

**B. API Compatibility - First Parameter Difference** (CRITICAL):
- [ ] Run integration test:
  ```python
  import inspect
  from tracknetv3.inference import TrackNetModule, TrackNetModuleONNX
  from tracknetv3.inference import InpaintModule, InpaintModuleONNX

  # Verify first parameter difference for TrackNet
  tn_pytorch_params = list(inspect.signature(TrackNetModule.__init__).parameters.keys())
  tn_onnx_params = list(inspect.signature(TrackNetModuleONNX.__init__).parameters.keys())

  print("TrackNetModule parameters:")
  print(f"  PyTorch: {tn_pytorch_params}")
  print(f"  ONNX: {tn_onnx_params}")

  assert tn_onnx_params[0] == 'model_path', f'First param should be model_path, got {tn_onnx_params[0]}'
  assert tn_pytorch_params[0] == 'tracknet', f'PyTorch first param should be tracknet, got {tn_pytorch_params[0]}'
  assert tn_onnx_params[1:] == tn_pytorch_params[1:], 'Remaining params should match between versions'
  print("✓ TrackNet API compatibility verified")

  # Same for InpaintModule
  inp_pytorch_params = list(inspect.signature(InpaintModule.__init__).parameters.keys())
  inp_onnx_params = list(inspect.signature(InpaintModuleONNX.__init__).parameters.keys())

  print("InpaintModule parameters:")
  print(f"  PyTorch: {inp_pytorch_params}")
  print(f"  ONNX: {inp_onnx_params}")

  assert inp_onnx_params[0] == 'model_path', f'First param should be model_path, got {inp_onnx_params[0]}'
  assert inp_pytorch_params[0] == 'inpaintnet', f'PyTorch first param should be inpaintnet, got {inp_pytorch_params[0]}'
  assert inp_onnx_params[1:] == inp_pytorch_params[1:], 'Remaining params should match between versions'
  print("✓ InpaintModule API compatibility verified")

  print("\n✓ All API compatibility tests passed!")
  ```
- [ ] Verify: All assertions pass
- [ ] Verify: Output shows first parameter differs (model_path vs tracknet/inpaintnet)
- [ ] Verify: Remaining parameters are identical

**C. Error Message Format Verification**:
- [ ] Test error message formats:
  ```python
  from tracknetv3.inference.streaming_onnx import TrackNetModuleONNX, InpaintModuleONNX
  import traceback

  # Test file not found error format
  try:
      TrackNetModuleONNX(model_path="/bad/path.onnx", seq_len=15, bg_mode="", device=None)
  except ValueError as e:
      error_msg = str(e)
      print(f"File not found error: {error_msg}")
      assert "ONNX model file not found:" in error_msg
      assert "/bad/path.onnx" in error_msg

  # Test metadata mismatch error format
  try:
      InpaintModuleONNX(model_path="/bad/path.onnx", seq_len=15, device=None, img_scaler=(1.0, 1.0))
  except ValueError as e:
      error_msg = str(e)
      print(f"File not found error: {error_msg}")
      assert "ONNX model file not found:" in error_msg
  ```
- [ ] Verify: Error messages match the format specified in Draft lines 339-352

**Evidence Capture**:
- [ ] Terminal output from API compatibility test showing first parameter difference
- [ ] Terminal output from error message verification

**Commit Guidance**:
- **Commit Suggested**: YES
- **Message**: `feat(inference): update exports and add integration tests for ONNX modules`
- **Files**: `tracknetv3/inference/__init__.py`

---

## Success Criteria & Final Verification

### Final Integration Test
```bash
# Expected: All API compatibility tests pass, all modules import successfully
python -c "
import inspect
from tracknetv3.inference import TrackNetModule, TrackNetModuleONNX
from tracknetv3.inference import InpaintModule, InpaintModuleONNX

# Verify first parameter difference
tn_pytorch_params = list(inspect.signature(TrackNetModule.__init__).parameters.keys())
tn_onnx_params = list(inspect.signature(TrackNetModuleONNX.__init__).parameters.keys())
assert tn_onnx_params[0] == 'model_path', f'Expected model_path, got {tn_onnx_params[0]}'
assert tn_pytorch_params[0] == 'tracknet', f'Expected tracknet, got {tn_pytorch_params[0]}'
assert tn_onnx_params[1:] == tn_pytorch_params[1:], 'Remaining params must match'

inp_pytorch_params = list(inspect.signature(InpaintModule.__init__).parameters.keys())
inp_onnx_params = list(inspect.signature(InpaintModuleONNX.__init__).parameters.keys())
assert inp_onnx_params[0] == 'model_path', f'Expected model_path, got {inp_onnx_params[0]}'
assert inp_pytorch_params[0] == 'inpaintnet', f'Expected inpaintnet, got {inp_pytorch_params[0]}'
assert inp_onnx_params[1:] == inp_pytorch_params[1:], 'Remaining params must match'

print('✓ All integration tests passed!')
print(f'  TrackNetModule: {tn_pytorch_params}')
print(f'  TrackNetModuleONNX: {tn_onnx_params}')
print(f'  InpaintModule: {inp_pytorch_params}')
print(f'  InpaintModuleONNX: {inp_onnx_params}')
"
```

**Expected Output**:
```
✓ All integration tests passed!
  TrackNetModule: ['tracknet', 'seq_len', 'bg_mode', 'device', 'eval_mode', 'median', 'median_warmup']
  TrackNetModuleONNX: ['model_path', 'seq_len', 'bg_mode', 'device', 'eval_mode', 'median', 'median_warmup']
  InpaintModule: ['inpaintnet', 'seq_len', 'device', 'img_scaler']
  InpaintModuleONNX: ['model_path', 'seq_len', 'device', 'img_scaler']
```

### Project Completion Checklist
- [ ] File `tracknetv3/inference/streaming_onnx.py` is created with both `TrackNetModuleONNX` and `InpaintModuleONNX`
- [ ] Both ONNX modules use `onnxruntime.InferenceSession` with proper provider selection
- [ ] Error handling uses `ValueError` for invalid paths/metadata and `RuntimeError` for provider issues
- [ ] Batch dimension handling is correct (`[None, ...]` before inference, `[0]` after) for both modules
- [ ] NumPy array conversions for ensemble weights are implemented correctly
- [ ] ONNX input names match export scripts: `"input"` for TrackNet, `"coords"` and `"mask"` for InpaintNet
- [ ] Metadata validation checks seq_len and bg_mode (TrackNet only) against model metadata
- [ ] Functional verification tests pass for both modules (model loading, dummy inference, provider selection)
- [ ] Error messages follow specified format patterns
- [ ] Dependency `onnxruntime-gpu>=1.19.0` is added to `pyproject.toml`
- [ ] Modules are exported in `tracknetv3/inference/__init__.py`
- [ ] API compatibility test verifies first parameter differs (model_path vs tracknet/inpaintnet)
- [ ] Remaining `__init__` parameters match between PyTorch and ONNX versions
- [ ] All background modes work correctly (`""`, `"subtract"`, `"subtract_concat"`, `"concat"`)
- [ ] Ensemble weighting produces same results as PyTorch version
- [ ] Statistics tracking is accurate (without AMP timing)
- [ ] Out of scope items are NOT implemented (custom optimizations, multi-GPU, dynamic seq_len, quantization)
