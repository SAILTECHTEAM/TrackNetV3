# Draft: ONNX-based Inference Modules for TrackNetV3

## User Requirements

### Primary Goal
Create ONNX Runtime-based inference modules (`TrackNetModuleONNX` and `InpaintModuleONNX`) that mirror the functionality of the existing PyTorch-based modules in `tracknetv3/inference/streaming.py`.

### Requirements
1. **Location**: Create new file `tracknetv3/inference/streaming_onnx.py`
2. **Runtime**: Use `onnxruntime-gpu` for GPU acceleration (with CPU fallback)
3. **API Compatibility**: Maintain identical API signatures to PyTorch versions
   - Same `__init__` parameters
   - Same method signatures (`push`, `flush`, `reset`)
   - Same return types and formats

## Discovered Context

### Existing PyTorch Modules
The reference modules in `tracknetv3/inference/streaming.py`:

#### `TrackNetModule` (Lines 15-310)
- **Purpose**: Streaming TrackNet with overlap ensemble
- **Key Features**:
  - Sequence-based processing with configurable `seq_len`
  - Background mode support: `""` (default), `"subtract"`, `"subtract_concat"`, `"concat"`
  - Median background computation (warmup-based or pre-provided)
  - Overlap ensemble with weighted averaging
  - Heatmap to coordinate conversion
  - Statistics tracking (timing)

- **Input/Flow**:
  - `push(frame_bgr, frame_id)` accepts BGR frames
  - Accumulates `seq_len` frames, preprocesses, runs inference
  - Returns prediction for frame_id - (seq_len - 1) once enough frames are buffered
  - Uses ensemble weights from `get_ensemble_weight()`

#### `InpaintModule` (Lines 313-416)
- **Purpose**: Streaming InpaintNet for coordinate refinement
- **Key Features**:
  - Takes predictions from TrackNet
  - Handles occlusions via mask
  - Buffer of `seq_len` coordinates
  - Statistics tracking

- **Input/Flow**:
  - `push(pred)` accepts `{"Frame": int, "X": int, "Y": int, "Visibility": int}`
  - Converts to normalized coords [0, 1], creates mask for occlusions
  - Returns refined prediction for oldest frame in buffer

### ONNX Export Scripts

#### TrackNet Export (`tools/export_tracknet_onnx.py`)
- **Input shapes** based on `bg_mode`:
  - `""` or `None`: `[batch, seq_len*3, 288, 512]` (RGB frames)
  - `"subtract"`: `[batch, seq_len, 288, 512]` (difference from median)
  - `"subtract_concat"`: `[batch, seq_len*4, 288, 512]` (RGB + difference)
  - `"concat"`: `[batch, (seq_len+1)*3, 288, 512]` (median + RGB frames)

- **Output**: `[batch, seq_len, 288, 512]` heatmaps

- **Metadata stored**: `model_name`, `seq_len`, `bg_mode`, `batch_mode`, `batch_size`

#### InpaintNet Export (`tools/export_inpaintnet_onnx.py`)
- **Input 1**: `coords` - `[batch, seq_len, 2]`
- **Input 2**: `mask` - `[batch, seq_len, 1]`
- **Output**: `[batch, seq_len, 2]` refined coords

- **Metadata stored**: `model_name`, `seq_len`, `batch_mode`, `batch_size`

### Constants (`tracknetv3/config/constants.py`)
- `HEIGHT = 288`
- `WIDTH = 512`

### Project Dependencies
- `pyproject.toml` already has `onnx>=1.19.0`
- **Missing**: `onnxruntime-gpu` needs to be added

## Technical Decisions & Rationale

### 1. ONNX Runtime Integration

**Decision**: Use `onnxruntime.InferenceSession` with provider selection

**Rationale**:
- ONNX Runtime expects NumPy arrays (not PyTorch tensors)
- Need explicit provider selection for GPU support
- Will check availability and provide graceful fallback to CPU

**Implementation approach**:
```python
import onnxruntime as ort

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 1  # For GPU

session = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)
```

**Error handling requirements**:
- Raise `ValueError` with clear message if model file doesn't exist
- Raise `ValueError` if model metadata validation fails (seq_len or bg_mode mismatch)
- Raise `RuntimeError` if no execution providers are available

### 2. Model Path Specification

**Decision**: Accept `model_path: str` instead of `torch.nn.Module` in `__init__`

**Rationale**:
- ONNX models are loaded from disk, not in-memory modules
- Need to maintain API compatibility - will accept `model_path` as additional parameter
- All other parameters remain identical

**Signature changes**:
```python
# PyTorch version
def __init__(self, tracknet: torch.nn.Module, ...)

# ONNX version
def __init__(self, model_path: str, ...)
```

**Error handling requirements**:
- Check `os.path.exists(model_path)` and raise `ValueError` with clear message if not found
- Verify model file can be loaded (catch ONNXRuntimeError and re-raise with context)

### 3. Input/Output Conversion

**Decision**: Convert PyTorch tensor operations to NumPy equivalents

**Rationale**:
- ONNX Runtime outputs are NumPy arrays
- Need to maintain identical behavior without PyTorch

**Key conversions**:
- Preprocessing: Already uses NumPy in PyTorch version → no change needed
- Inference: Replace `self.tracknet(x)` with `session.run(None, {input_name: x})`
- Postprocessing: `y.float().detach().cpu().numpy()` → direct NumPy output

### 4. Device Handling

**Decision**: Accept `device` parameter for API compatibility but use it for provider selection only

**Rationale**:
- ONNX Runtime doesn't have have the same device concept as PyTorch
- Device parameter will determine whether to prioritize GPU provider
- Maintains API signature while using appropriate ONNX mechanism

### 5. Metadata Validation

**Decision**: Validate ONNX model metadata matches expected configuration

**Rationale**:
- Export scripts store `seq_len`, `bg_mode` in metadata
- Helps catch mismatched model/config issues early
- Provides better error messages

### 6. AMP (Automatic Mixed Precision)

**Decision**: Disable AMP equivalent for ONNX (not applicable)

**Rationale**:
- ONNX Runtime has its own optimization through graph optimization levels
- `ORT_ENABLE_ALL` is the ONNX equivalent
- Remove timing statistics related to AMP from ONNX versions

## Implementation Approach

### File Structure

```
tracknetv3/inference/
├── streaming.py           # Existing PyTorch modules
├── streaming_onnx.py      # New: ONNX modules
├── offline.py
├── helpers.py
└── __init__.py            # To be updated
```

### Module 1: TrackNetModuleONNX

**Inheritance/Composition**:
- Similar structure to `TrackNetModule` (not inheriting, to avoid PyTorch dependencies)

**Key modifications from PyTorch version**:

1. **`__init__` changes**:
   - Accept `model_path: str` instead of `tracknet: torch.nn.Module`
   - Load ONNX session with provider configuration
   - Cache input/output names from model
   - Validate metadata (seq_len, bg_mode)

2. **`push` method changes**:
   - Replace PyTorch inference with ONNX Runtime:
     ```python
     # PyTorch:
     with torch.inference_mode(), torch.amp.autocast(...):
         y = self.tracknet(x)

     # ONNX:
     ort_inputs = {self.input_name: x_np}
     y = self.session.run(self.output_names, ort_inputs)[0]
     ```
   - Remove AMP-related timing (`use_amp` variable)
   - Remove CUDA synchronization (not needed with ONNX Runtime)
   - Use NumPy array directly (no `.float().detach().cpu().numpy()`)

3. **Unchanged components**:
   - Preprocessing logic (`_process_one`, `_ensure_scaler`, `_maybe_build_median`)
   - Median computation
   - Background mode handling
   - Ensemble weight accumulation
   - Heatmap to coordinate conversion (`_heatmap_to_xy`)
   - All deque management logic
   - Statistics tracking (except timing adjustments)

### Module 2: InpaintModuleONNX

**Key modifications from PyTorch version**:

1. **`__init__` changes**:
   - Accept `model_path: str` instead of `inpaintnet: torch.nn.Module`
   - Load ONNX session
   - Cache input/output names (2 inputs: coords, mask; 1 output)

2. **`push` method changes**:
   - Replace PyTorch inference with ONNX:
     ```python
     # PyTorch:
     out = self.inpaintnet(coor.to(self.device), mask.to(self.device))

     # ONNX:
     ort_inputs = {
         self.input_names_coords: coords_np,
         self.input_names_mask: mask_np
     }
     out = self.session.run(self.output_names, ort_inputs)[0]
     ```
   - Remove device transfers (`.to(self.device)`)
   - Remove AMP context and synchronization

3. **Unchanged components**:
   - Coordinate normalization
   - Mask creation for occlusions
   - Output postprocessing
   - Deque management
   - Statistics tracking

## Dependencies

### To Add
```toml
onnxruntime-gpu = ">=1.19.0"
```

**Note**: For CPU-only setups, user can install `onnxruntime` separately. The code should handle both cases gracefully.

## Testing Considerations

### Validation Criteria
1. ONNX modules produce identical outputs to PyTorch versions (within floating-point tolerance)
2. All background modes work correctly
3. Ensemble weighting produces same results
4. Device selection (CPU/GPU) works as expected
5. Statistics tracking is accurate
6. **Functional verification**: ONNX models can be loaded and used for actual inference

### Test Scenarios
1. Basic inference with default mode
2. All background modes (`subtract`, `concat`, `subtract_concat`)
3. Median warmup vs pre-provided median
4. Different sequence lengths
5. CPU vs GPU execution
6. Flush behavior for tail frames
7. **Error handling**: Invalid model path, corrupted model file, metadata mismatch

### Functional Verification Requirements

**TrackNetModuleONNX verification steps**:
1. Test ONNX model loading with valid path
2. Verify metadata validation (correct seq_len and bg_mode in model metadata)
3. Run inference with dummy input (properly shaped NumPy array)
4. Verify provider selection: confirm CUDA provider used when available
5. Test error handling: pass non-existent path, verify clear error message
6. Test error handling: pass model with mismatched seq_len, verify validation fails

**InpaintModuleONNX verification steps**:
1. Test ONNX model loading with valid path
2. Verify metadata validation (correct seq_len)
3. Run inference with dummy coords and mask inputs
4. Verify two-input handling (coords and mask)
5. Test error handling for invalid paths and metadata mismatches

## Critical Implementation Notes

### NumPy Array Conversions

**Ensemble weights**: `get_ensemble_weight()` returns `torch.Tensor`. Must convert to `np.ndarray`:
```python
self._ens_w = np.asarray(
    get_ensemble_weight(self.seq_len, self.eval_mode).cpu().numpy(),
    dtype=np.float32
)
```
This pattern is already present in PyTorch version at `streaming.py:44-46`.

### Batch Dimension Handling

ONNX models expect batched inputs even for single-frame inference. Must add batch dimension:

**TrackNetModuleONNX**:
```python
# x_np shape: [in_dim, HEIGHT, WIDTH]
ort_inputs = {self.input_name: x_np[None, ...]}  # Add batch dim → [1, in_dim, HEIGHT, WIDTH]
outputs = self.session.run(self.output_names, ort_inputs)[0]  # Output: [1, seq_len, HEIGHT, WIDTH]
y = outputs[0]  # Remove batch dim → [seq_len, HEIGHT, WIDTH]
```

**InpaintModuleONNX**:
```python
# coords_np shape: [seq_len, 2], mask_np shape: [seq_len, 1]
ort_inputs = {
    self.input_coords: coords_np[None, ...],  # → [1, seq_len, 2]
    self.input_mask: mask_np[None, ...]       # → [1, seq_len, 1]
}
outputs = self.session.run(self.output_names, ort_inputs)[0]  # Output: [1, seq_len, 2]
out = outputs[0]  # Remove batch dim → [seq_len, 2]
```

### ONNX Input Names

Based on export scripts:
- **TrackNet**: Input name is `"input"` (see `export_tracknet_onnx.py:55`)
- **InpaintNet**: Input names are `"coords"` and `"mask"` (see `export_inpaintnet_onnx.py:42`)

### Error Message Format

All error messages should follow this pattern:
```python
# File not found
raise ValueError(f"ONNX model file not found: {model_path}")

# Metadata mismatch
raise ValueError(
    f"Model metadata mismatch: expected seq_len={expected}, "
    f"got {actual}. Model={model_path}"
)

# Provider availability
raise RuntimeError("No ONNX Runtime execution providers available")
```

## Assumptions

1. **ONNX models are already exported**: User has run export scripts and has ONNX files
2. **Metadata consistency**: Exported models have proper metadata set by export scripts
3. **CUDA availability**: GPU support depends on CUDA being properly configured
4. **Floating-point precision**: Minor numerical differences between PyTorch and ONNX are acceptable due to precision differences in implementations
5. **ONNX Runtime 1.19+**: Using features available in onnxruntime>=1.19.0

## Integration Testing Requirements

The final integration test should explicitly verify API compatibility:

```python
# Verify first parameter difference
tn_pytorch_params = list(inspect.signature(TrackNetModule.__init__).parameters.keys())
tn_onnx_params = list(inspect.signature(TrackNetModuleONNX.__init__).parameters.keys())

assert tn_onnx_params[0] == 'model_path', f'First param should be model_path, got {tn_onnx_params[0]}'
assert tn_pytorch_params[0] == 'tracknet', f'PyTorch first param should be tracknet, got {tn_pytorch_params[0]}'
assert tn_onnx_params[1:] == tn_pytorch_params[1:], 'Remaining params should match between versions'

# Same for InpaintModule
inp_pytorch_params = list(inspect.signature(InpaintModule.__init__).parameters.keys())
inp_onnx_params = list(inspect.signature(InpaintModuleONNX.__init__).parameters.keys())

assert inp_onnx_params[0] == 'model_path', f'First param should be model_path, got {inp_onnx_params[0]}'
assert inp_pytorch_params[0] == 'inpaintnet', f'PyTorch first param should be inpaintnet, got {inp_pytorch_params[0]}'
assert inp_onnx_params[1:] == inp_pytorch_params[1:], 'Remaining params should match between versions'
```

## Exclusions (Out of Scope)

1. **ONNX model export scripts**: Already exist in `tools/export_tracknet_onnx.py` and `tools/export_inpaintnet_onnx.py`
2. **ONNX optimization**: Using default `ORT_ENABLE_ALL` level; custom optimizations not considered
3. **Multi-GPU support**: Will use default GPU (device_id: 0) when CUDA provider is available
4. **Dynamic sequence lengths**: Assume fixed sequence length as exported; no runtime sequence length changes
5. **Model quantization/int8**: Focus on FP32 models only
