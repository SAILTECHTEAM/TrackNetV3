# Refactoring Plan: Fix Module Boundary Violation - Move inpaint-related functions from scripts to tracknetv3

## Version
v1.0

## User Requirements

### Original Request
用户指出模块架构违反：`tracknetv3` 模块内的代码不应该导入模块外的 `scripts` 目录内容。具体问题是 `tracknetv3/inference/offline.py` 导入了 `scripts.test.generate_inpaint_mask`。

### Clarified Requirements (via Q&A)
1. **一起移动相关函数**: 将 `generate_inpaint_mask` 和 `linear_interp` 两个函数都移动到 `tracknetv3` 模块内，因为它们有依赖关系
2. **创建新模块**: 在 `tracknetv3/utils/` 下新建 `trajectory.py` 来存放这些轨迹处理相关的工具函数
3. **更新所有导入**: 直接修改 `scripts/test.py` 中的导入语句，从新的位置导入函数，不保留向后兼容层

## Context Investigation

### Problem Analysis

#### Architecture Violation
**Location**: `tracknetv3/inference/offline.py:15`
```python
from scripts.test import generate_inpaint_mask
```

**Why This is Wrong**:
- `tracknetv3` 是一个可复用的库模块，应该独立于使用它的代码
- 库不应该依赖外部的 `scripts/` 目录（这是应用层/工具脚本）
- 破坏了封装性和模块边界
- 会导致安装/打包问题（如果通过 pip 安装 tracknetv3，但依赖 scripts 目录会失败）
- 增加循环依赖风险

#### Functions to Move

**1. `generate_inpaint_mask`**
- **Source**: `scripts/test.py:284-318`
- **Purpose**: 根据预测轨迹生成 inpaint mask（用于图像修复的遮罩）
- **Signature**: `generate_inpaint_mask(pred_dict, th_h=30) -> List[int]`
- **Dependencies**: 纯 NumPy 函数，无外部依赖
- **Logic**:
  - 基于 `pred_dict` 中的 `visibility` 字段生成二进制 mask
  - 根据 y 坐标阈值 `th_h` 判断是否需要 inpaint
  - 返回一个 0/1 列表（0: 不需要 inpaint, 1: 需要 inpaint）

**2. `linear_interp`**
- **Source**: `scripts/test.py:321-347`
- **Purpose**: 在 inpaint_mask 标记的位置进行线性插值
- **Signature**: `linear_interp(pred_dict, inpaint_mask) -> List[float]`
- **Dependencies**: 被 `generate_inpaint_mask` 调用，纯 NumPy 函数
- **Logic**:
  - 找到 mask 中需要插值的连续段
  - 使用可见段的端点进行线性插值
  - 返回插值后的坐标值

#### Usage Locations

| File | Line | Context | Legitimate? |
|------|------|---------|-------------|
| `tracknetv3/inference/offline.py` | 369 | `_run_inpaintnet` 方法中 | ❌ **违反架构** |
| `scripts/test.py` | 670 | 测试/调试函数中 | ✅ 合理 |
| `scripts/test.py` | 866 | 主测试流程中 | ✅ 合理 |
| `scripts/test.py` | 1153 | 其他测试函数中 | ✅ 合理 |
| `scripts/test.py` | 1162 | 相关的线性插值使用 | ✅ 合理 |

### Project Structure Context

#### Existing Code Patterns

**tracknetv3/inference/helpers.py**
- Contains prediction utilities like `_predict_from_network_outputs_fast`
- Pure NumPy/PyTorch functions
- Good example of helper functions for inference
- Currently has ~100 lines

**tracknetv3/utils/general.py**
- Contains various utility functions
- General-purpose helpers not tied to specific inference logic
- Growing file, already has multiple utility functions

**Decision Rationale**:
- ✅ **Not in helpers.py**: Because these functions are trajectory processing utilities, not inference-specific
- ✅ **Not in general.py**: To avoid making the file too large and to create better semantic organization
- ✅ **New file trajectory.py**: Creates clear separation and follows single responsibility principle

### Dependencies and Imports

**Current Imports in affected files**:
- `scripts/test.py`: Imports `numpy as np` at the top
- `tracknetv3/inference/offline.py`: Currently has `from scripts.test import generate_inpaint_mask`

**No additional dependencies needed** - Both functions only use NumPy which is already a dependency.

## Decisions & Rationale

### Decision 1: Move Both Functions Together
**Rationale**:
- `linear_interp` is called by `generate_inpaint_mask` (tight coupling)
- Both functions work together to process trajectory data
- Keeping them together makes the code easier to maintain
- Reduces the risk of breaking the dependency if moved separately

### Decision 2: Create New Module `tracknetv3/utils/trajectory.py`
**Rationale**:
- Clear semantic grouping: both functions are trajectory processing utilities
- Follows single responsibility principle: file focuses on trajectory-related operations
- Better organization than putting in existing larger files
- Future trajectory utilities can be added here
- Easier to test and maintain as a self-contained module

### Decision 3: Update All Imports Directly (No Backward Compatibility)
**Rationale**:
- Keep codebase clean without redundant re-export layers
- Simpler maintenance (one source of truth)
- Scripts directory is not a public API, so backward compatibility not critical
- All usages are within the same repository, easy to update

### Decision 4: Keep Function Signatures Unchanged
**Rationale**:
- No behavioral changes required
- Minimal risk of breaking existing code
- Only import statements need to change

## Assumptions

1. **No Tests for These Functions**: Based on exploration, there are no dedicated unit tests for `generate_inpaint_mask` and `linear_interp`. Integration tests in `scripts/test.py` serve as validation.
   - **Implication**: After refactoring, manual testing or running `scripts/test.py` should be performed to verify correctness

2. **No Other Files Import from scripts**: Based on `grep` search, only `tracknetv3/inference/offline.py` and `scripts/test.py` itself use these functions.
   - **Implication**: Limited scope of changes needed

3. **NumPy is Already a Project Dependency**: Both functions only use `numpy`, which is already used throughout the project.
   - **Implication**: No new dependencies to add to requirements files

4. **Users Will Not Be Using These Functions Directly**: These are internal utility functions, not part of the public API exposed in `tracknetv3/__init__.py`.
   - **Implication**: No need to add to public API exports

## Implementation Scope

### Files to Create
1. `tracknetv3/utils/trajectory.py` - New module for trajectory utilities
2. `tracknetv3/utils/__init__.py` - Update to export new functions (if needed)

### Files to Modify
1. `scripts/test.py` - Remove `generate_inpaint_mask` and `linear_interp` functions, update imports
2. `tracknetv3/inference/offline.py` - Update import statement

### Verification Needed
- Run `scripts/test.py` to ensure functionality still works
- Verify no import errors
- Check that the moved functions are correctly exported and accessible

## Success Criteria

1. ✅ No architecture violations remain (no `from scripts` imports in `tracknetv3/`)
2. ✅ All functionality preserved (test script runs successfully)
3. ✅ Clear module organization (trajectory utilities properly grouped)
4. ✅ No code duplication (functions moved, not copied)
5. ✅ Updated imports in all affected files
