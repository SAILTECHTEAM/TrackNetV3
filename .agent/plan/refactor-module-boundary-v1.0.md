# Refactor Module Boundary: Move inpaint-related functions from scripts to tracknetv3

## Context

### Source Document
- **Draft**: `.agent/drafts/refactor-module-boundary-v1.0.md`
- **Objective**: Fix architecture violation by moving `generate_inpaint_mask` and `linear_interp` from `scripts/test.py` to `tracknetv3/utils/trajectory.py`, then update all imports to eliminate `tracknetv3 → scripts` dependency.

### Summary of Requirements
- **Core Deliverables**:
  - Create new module `tracknetv3/utils/trajectory.py` containing `generate_inpaint_mask` and `linear_interp` functions
  - Update `tracknetv3/utils/__init__.py` to export the new functions
  - Remove `generate_inpaint_mask` and `linear_interp` from `scripts/test.py`
  - Update import in `tracknetv3/inference/offline.py` from `from scripts.test import generate_inpaint_mask` to `from tracknetv3.utils.trajectory import generate_inpaint_mask, linear_interp`
  - Update imports in `scripts/test.py` to use the new location

- **Key Constraints**:
  - Function signatures MUST remain unchanged (no API changes)
  - No new dependencies to add (NumPy is already a project dependency)
  - No backward compatibility layer needed (scripts directory is not public API)
  - Functions must be moved, not copied (avoid code duplication)

- **Out of Scope**:
  - No changes to function logic/behavior
  - No addition of unit tests (none exist currently)
  - No modification to `tracknetv3/__init__.py` (these are internal utilities)
  - No changes to other utility files (`helpers.py`, `general.py`)

---

## Work Objectives & Verification Strategy

### Definition of Done
The project is complete when:
- [ ] `tracknetv3/utils/trajectory.py` exists with both functions correctly implemented
- [ ] `tracknetv3/utils/__init__.py` exports the new functions
- [ ] `scripts/test.py` no longer contains the moved functions and imports from the new location
- [ ] `tracknetv3/inference/offline.py` imports from the new location
- [ ] Running `scripts/test.py` executes successfully without errors
- [ ] No import errors when importing from `tracknetv3` module

### Verification Approach
*Based on the Draft's stated testing strategy.*

**Decision**:
- **Test Infrastructure Exists**: NO (no dedicated unit tests for these functions)
- **Testing Mandate**: Manual QA (running `scripts/test.py` as integration test)
- **Framework/Tools**: Python execution, import validation

**For Manual QA**:
- **CRITICAL**: Each task MUST include explicit, step-by-step manual verification procedures.
- **Evidence Required**: Command outputs showing successful execution and no import errors.

---

## Architecture & Design

### High-Level Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Before Refactoring                        │
├─────────────────────────────────────────────────────────────┤
│  tracknetv3/ (library)                                      │
│    └── inference/                                           │
│        └── offline.py                                       │
│            └── from scripts.test import generate_inpaint_mask  ❌
│                                                             │
│  scripts/ (application layer)                                │
│    └── test.py                                              │
│        └── generate_inpaint_mask() [lines 284-318]         │
│        └── linear_interp() [lines 321-347]                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    After Refactoring                         │
├─────────────────────────────────────────────────────────────┤
│  tracknetv3/ (library)                                      │
│    ├── inference/                                           │
│    │   └── offline.py                                       │
│    │       └── from tracknetv3.utils.trajectory import ...  ✅
│    └── utils/                                               │
│        ├── __init__.py                                      │
│        └── trajectory.py (NEW)                              │
│            ├── generate_inpaint_mask()                      │
│            └── linear_interp()                              │
│                                                             │
│  scripts/ (application layer)                                │
│    └── test.py                                              │
│        ├── from tracknetv3.utils.trajectory import ...      │
│        └── [functions removed from this file]              │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. Inference Process (offline.py):
   prediction → generate_inpaint_mask → linear_interp → final_trajectory

2. Testing Process (test.py):
   prediction → generate_inpaint_mask → linear_interp → validation
```

### Key Architectural Decisions

1. **Create New Module `trajectory.py`**
   - **Rationale (from Draft)**: Functions are trajectory processing utilities, not inference-specific. Creating a new file follows single responsibility principle and provides clear semantic grouping. Better organization than adding to existing larger files (`helpers.py`, `general.py`).

2. **Move Both Functions Together**
   - **Rationale (from Draft)**: `linear_interp` is called by `generate_inpaint_mask` (tight coupling). Keeping them together makes code easier to maintain and reduces risk of breaking dependency if moved separately.

3. **Direct Import Updates (No Backward Compatibility)**
   - **Rationale (from Draft)**: Keep codebase clean without redundant re-export layers. Scripts directory is not a public API, so backward compatibility not critical. All usages are within the same repository, easy to update.

---

## Task Breakdown

```
Task 0 (Create trajectory.py)
    │
    ├── Task 1 (Update utils/__init__.py)
    │
    ├── Task 2 (Update offline.py import)
    │
    └── Task 3 (Update and clean test.py)
            │
            └── Task 4 (Final verification)
```

### Parallelization Guide
| Task Group | Tasks | Rationale for Parallelism |
|------------|-------|---------------------------|
| Group A    | 2     | No parallelization - each task depends on the previous to maintain code integrity during refactoring. |

### Dependency Mapping
| Task | Depends On | Reason for Dependency |
|------|------------|----------------------|
| 1    | 0          | Task 0 must create the module before Task 1 can export it. |
| 2    | 0, 1       | Task 2 imports from the new module which must exist and be export-ready. |
| 3    | 0, 1       | Task 3 imports from the new module which must exist and be export-ready. |
| 4    | 1, 2, 3    | Task 4 verifies all changes are complete and functional. |

---

## TODOs

### Task 0: Create trajectory.py module with both functions

**Objective**: Create new module `tracknetv3/utils/trajectory.py` containing `generate_inpaint_mask` and `linear_interp` functions moved from `scripts/test.py`.

**Implementation Steps**:
1. Create new file `tracknetv3/utils/trajectory.py`
2. Add standard NumPy import at the top
3. Add docstring describing the module's purpose (trajectory processing utilities)
4. Copy `generate_inpaint_mask` function from `scripts/test.py:284-318`
5. Copy `linear_interp` function from `scripts/test.py:321-347`
6. Ensure function signatures remain exactly as specified

**Parallelizable**: NO
- **If YES**: Can run concurrently with: [none - this is the foundation task]

**References (CRITICAL)**:
> Exhaustively list reference points from the Draft or implied project context. Explain **WHAT** to extract and **WHY** it's relevant.

- **Function 1 Reference**: `[scripts/test.py:284-318]` - Extract `generate_inpaint_mask` function exactly as-is. This is the primary function being moved to fix the architecture violation.
- **Function 2 Reference**: `[scripts/test.py:321-347]` - Extract `linear_interp` function exactly as-is. This function is called by `generate_inpaint_mask` and must be moved together to maintain dependency.
- **Module Pattern Reference**: `[tracknetv3/utils/general.py]` - Follow the existing pattern of utility modules in the project (NumPy import, docstring structure).
- **Import Pattern Reference**: `[tracknetv3/inference/offline.py:15]` - Current import statement that needs to be updated after this task completes.

**Acceptance Criteria**:
*Follow the strategy defined in "Verification Approach".*

**For Manual Verification (ALWAYS include, even with tests)**:
- **Module Creation**:
  - [ ] Verify: File `tracknetv3/utils/trajectory.py` exists
  - [ ] Run: `python -c "import tracknetv3.utils.trajectory; print('Import successful')"`
  - [ ] Verify: Output contains `"Import successful"` and exits with code `0`.
- **Function Availability**:
  - [ ] Run: `python -c "from tracknetv3.utils.trajectory import generate_inpaint_mask, linear_interp; print('Functions imported:', generate_inpaint_mask, linear_interp)"`
  - [ ] Verify: Output shows both functions are imported successfully without errors.
- **Function Signatures**:
  - [ ] Run: `python -c "from tracknetv3.utils.trajectory import generate_inpaint_mask, linear_interp; import inspect; print('generate_inpaint_mask sig:', inspect.signature(generate_inpaint_mask)); print('linear_interp sig:', inspect.signature(linear_interp))"`
  - [ ] Verify: Output shows `generate_inpaint_mask sig: (pred_dict, th_h=30)` and `linear_interp sig: (pred_dict, inpaint_mask)`.

**Evidence Capture**:
- [ ] Terminal output of all verification commands is saved/captured.
- [ ] Content of `tracknetv3/utils/trajectory.py` is saved for reference.

**Commit Guidance**:
- **Commit Suggested**: YES (Group with Task 1)
- **Message**: `feat(utils): create trajectory.py module with generate_inpaint_mask and linear_interp`
- **Files**: `tracknetv3/utils/trajectory.py`, `tracknetv3/utils/__init__.py`

---

### Task 1: Update utils/__init__.py to export new functions

**Objective**: Modify `tracknetv3/utils/__init__.py` to export `generate_inpaint_mask` and `linear_interp` from the new `trajectory.py` module.

**Implementation Steps**:
1. Read `tracknetv3/utils/__init__.py` to understand current export pattern
2. Add import statements for the new functions from `trajectory` module
3. Add the functions to the `__all__` list (if exists) or create appropriate exports
4. Follow existing code style and formatting patterns

**Parallelizable**: NO
- **If YES**: Can run concurrently with: [none - depends on Task 0]

**References (CRITICAL)**:
> Exhaustively list reference points from the Draft or implied project context. Explain **WHAT** to extract and **WHY** it's relevant.

- **Current __init__ Structure**: `[tracknetv3/utils/__init__.py]` - Understand existing export pattern (uses `__all__` or direct imports) to maintain consistency.
- **Module Pattern Reference**: `[tracknetv3/inference/__init__.py]` - Reference for how other submodules export their functions.
- **Decision Reference**: `[Draft section "Decision Rationale: Create New Module"]` - Functions should be accessible via `tracknetv3.utils` namespace.
- **Function Location**: `[tracknetv3/utils/trajectory.py]` - Source of functions to export.

**Acceptance Criteria**:
*Follow the strategy defined in "Verification Approach".*

**For Manual Verification (ALWAYS include, even with tests)**:
- **Module Export Verification**:
  - [ ] Run: `python -c "from tracknetv3.utils import generate_inpaint_mask, linear_interp; print('Functions imported via tracknetv3.utils:', generate_inpaint_mask, linear_interp)"`
  - [ ] Verify: Output shows both functions imported successfully without errors.
- **Import Path Verification**:
  - [ ] Run: `python -c "from tracknetv3.utils.trajectory import generate_inpaint_mask as func1; from tracknetv3.utils import generate_inpaint_mask as func2; print('Same function:', func1 is func2)"`
  - [ ] Verify: Output shows `Same function: True`.
- **Namespace Availability**:
  - [ ] Run: `python -c "import tracknetv3.utils as utils; print('Available functions:', dir(utils))"`
  - [ ] Verify: Output list contains `'generate_inpaint_mask'` and `'linear_interp'`.

**Evidence Capture**:
- [ ] Terminal output of all verification commands is saved/captured.
- [ ] Content of modified `tracknetv3/utils/__init__.py` is saved for reference.

**Commit Guidance**:
- **Commit Suggested**: YES (Group with Task 0)
- **Message**: `feat(utils): create trajectory.py module with generate_inpaint_mask and linear_interp`
- **Files**: `tracknetv3/utils/trajectory.py`, `tracknetv3/utils/__init__.py`

---

### Task 2: Update import in tracknetv3/inference/offline.py

**Objective**: Replace the architecture-violating import `from scripts.test import generate_inpaint_mask` with `from tracknetv3.utils.trajectory import generate_inpaint_mask, linear_interp` in `tracknetv3/inference/offline.py`.

**Implementation Steps**:
1. Locate the import statement at line 15 in `tracknetv3/inference/offline.py`
2. Replace `from scripts.test import generate_inpaint_mask` with `from tracknetv3.utils.trajectory import generate_inpaint_mask, linear_interp`
3. Verify no other references to `scripts.test` remain in the file
4. Ensure no other changes to the file logic or structure

**Parallelizable**: NO
- **If YES**: Can run concurrently with: [none - depends on Tasks 0 and 1]

**References (CRITICAL)**:
> Exhaustively list reference points from the Draft or implied project context. Explain **WHAT** to extract and **WHY** it's relevant.

- **Current Import Location**: `[tracknetv3/inference/offline.py:15]` - Exact line with the violating import that must be replaced.
- **Usage Context**: `[tracknetv3/inference/offline.py:369]` - Usage in `_run_inpaintnet` method to verify import is correct.
- **Decision Reference**: `[Draft section "Usage Locations"]` - Confirms this is the only usage location in tracknetv3 that needs updating.
- **New Import Target**: `[tracknetv3/utils/trajectory.py]` - Source of the functions after refactoring.
- **Constraint Reference**: `[Draft section "Decision Rationale: Keep Function Signatures Unchanged"]` - Function names must remain exactly the same.

**Acceptance Criteria**:
*Follow the strategy defined in "Verification Approach".*

**For Manual Verification (ALWAYS include, even with tests)**:
- **Import Update Verification**:
  - [ ] Run: `grep -n "from scripts" tracknetv3/inference/offline.py`
  - [ ] Verify: Output is empty (no matches found).
  - [ ] Run: `grep -n "from tracknetv3.utils.trajectory import" tracknetv3/inference/offline.py`
  - [ ] Verify: Output shows a match at line 15 with both functions imported.
- **Module Import Test**:
  - [ ] Run: `python -c "from tracknetv3.inference.offline import *; print('Module imported successfully')"` (if offline.py allows star import) OR `python -c "import tracknetv3.inference.offline; print('Module imported successfully')"`
  - [ ] Verify: Output contains `"Module imported successfully"` and exits with code `0`.
- **Syntax Verification**:
  - [ ] Run: `python -m py_compile tracknetv3/inference/offline.py`
  - [ ] Verify: No syntax errors (command exits with code `0`).

**Evidence Capture**:
- [ ] Terminal output of all verification commands is saved/captured.
- [ ] Content of modified import statement (line 15 of `tracknetv3/inference/offline.py`) is saved.

**Commit Guidance**:
- **Commit Suggested**: YES
- **Message**: `fix(inference): remove scripts dependency in offline.py by using tracknetv3.utils.trajectory`
- **Files**: `tracknetv3/inference/offline.py`

---

### Task 3: Update scripts/test.py and remove moved functions

**Objective**: Update `scripts/test.py` to import `generate_inpaint_mask` and `linear_interp` from the new location, then remove the function definitions from the file.

**Implementation Steps**:
1. Locate and remove the `generate_inpaint_mask` function definition (lines 284-318)
2. Locate and remove the `linear_interp` function definition (lines 321-347)
3. Add import statement at the top of the file: `from tracknetv3.utils.trajectory import generate_inpaint_mask, linear_interp`
4. Ensure all existing usages of these functions in the file (at lines 670, 866, 1153, 1162) continue to work
5. Verify no other code changes are made

**Parallelizable**: NO
- **If YES**: Can run concurrently with: [none - depends on Tasks 0 and 1]

**References (CRITICAL)**:
> Exhaustively list reference points from the Draft or implied project context. Explain **WHAT** to extract and **WHY** it's relevant.

- **Function 1 Definition**: `[scripts/test.py:284-318]` - Exact lines to remove (`generate_inpaint_mask`).
- **Function 2 Definition**: `[scripts/test.py:321-347]` - Exact lines to remove (`linear_interp`).
- **Usage Locations**: `[scripts/test.py:670, 866, 1153, 1162]` - All places where functions are used within the file (must continue to work after import update).
- **Import Pattern Reference**: `[scripts/test.py:1-50]` - Location of existing imports to add new import statement in correct location.
- **Decision Reference**: `[Draft section "Clarified Requirements #3"]` - Update all imports directly, no backward compatibility layer.
- **Constraint Reference**: `[Draft section "Out of Scope"]` - No changes to function logic, only imports and removal.

**Acceptance Criteria**:
*Follow the strategy defined in "Verification Approach".*

**For Manual Verification (ALWAYS include, even with tests)**:
- **Function Removal Verification**:
  - [ ] Run: `grep -n "def generate_inpaint_mask" scripts/test.py`
  - [ ] Verify: Output is empty (function definition removed).
  - [ ] Run: `grep -n "def linear_interp" scripts/test.py`
  - [ ] Verify: Output is empty (function definition removed).
- **Import Update Verification**:
  - [ ] Run: `grep -n "from tracknetv3.utils.trajectory import" scripts/test.py`
  - [ ] Verify: Output shows a match with both functions imported.
- **Module Import Test**:
  - [ ] Run: `python -c "import sys; sys.path.insert(0, '.'); import scripts.test; print('Module imported successfully')"`
  - [ ] Verify: Output contains `"Module imported successfully"` and exits with code `0`.
- **Syntax Verification**:
  - [ ] Run: `python -m py_compile scripts/test.py`
  - [ ] Verify: No syntax errors (command exits with code `0`).

**Evidence Capture**:
- [ ] Terminal output of all verification commands is saved/captured.
- [ ] Content of new import statement in `scripts/test.py` is saved.
- [ ] Diff showing removed function definitions is captured.

**Commit Guidance**:
- **Commit Suggested**: YES
- **Message**: `refactor(scripts): remove generate_inpaint_mask and linear_interp, now imported from tracknetv3.utils.trajectory`
- **Files**: `scripts/test.py`

---

### Task 4: Final verification and integration testing

**Objective**: Verify that all changes work correctly together by running `scripts/test.py` and checking that no architecture violations remain.

**Implementation Steps**:
1. Search entire `tracknetv3/` directory for any remaining imports from `scripts`
2. Run `scripts/test.py` to verify functionality is preserved
3. Verify all imports resolve correctly
4. Check that the moved functions are accessible from their new location
5. Confirm no import errors when importing from `tracknetv3` module

**Parallelizable**: NO
- **If YES**: Can run concurrently with: [none - this is the final verification step]

**References (CRITICAL)**:
> Exhaustively list reference points from the Draft or implied project context. Explain **WHAT** to extract and **WHY** it's relevant.

- **Success Criteria Reference**: `[Draft section "Success Criteria"]` - All criteria must be met (no architecture violations, functionality preserved, clear organization, no code duplication, updated imports).
- **Verification Reference**: `[Draft section "Verification Needed"]` - Run `scripts/test.py`, verify no import errors, check functions are correctly exported.
- **Assumption Reference**: `[Draft section "Assumptions #1"]` - Integration tests in `scripts/test.py` serve as validation (no dedicated unit tests exist).
- **Usage Locations Reference**: `[Draft section "Usage Locations"]` - All locations should still work after refactoring.

**Acceptance Criteria**:
*Follow the strategy defined in "Verification Approach".*

**For Manual Verification (ALWAYS include, even with tests)**:
- **Architecture Violation Check**:
  - [ ] Run: `grep -r "from scripts" tracknetv3/ --include="*.py"`
  - [ ] Verify: Output is empty (no architecture violations remain).
- **Module Integration Test**:
  - [ ] Run: `cd /workspace/TrackNetV3 && python scripts/test.py` (or appropriate command to run the test script)
  - [ ] Verify: Script executes successfully without import errors and produces expected output.
- **Function Accessibility Test**:
  - [ ] Run: `python -c "from tracknetv3.utils.trajectory import generate_inpaint_mask, linear_interp; from tracknetv3.utils import generate_inpaint_mask as g1; print('Direct import:', g1 is generate_inpaint_mask)"`
  - [ ] Verify: Output shows `Direct import: True`.
- **Import Hierarchy Verification**:
  - [ ] Run: `python -c "import tracknetv3; print('tracknetv3 imported'); from tracknetv3.inference import offline; print('offline imported'); from tracknetv3.utils import trajectory; print('trajectory imported')"`
  - [ ] Verify: All imports succeed without errors.
- **No Code Duplication Check**:
  - [ ] Run: `grep -r "def generate_inpaint_mask" . --include="*.py" | grep -v ".git"`
  - [ ] Verify: Output shows exactly one match (in `tracknetv3/utils/trajectory.py`).
  - [ ] Run: `grep -r "def linear_interp" . --include="*.py" | grep -v ".git"`
  - [ ] Verify: Output shows exactly one match (in `tracknetv3/utils/trajectory.py`).

**Evidence Capture**:
- [ ] Terminal output of `scripts/test.py` execution is captured.
- [ ] Output of all grep/verification commands is saved.
- [ ] Screenshot or log showing successful test execution.

**Commit Guidance**:
- **Commit Suggested**: NO (All work should be committed in previous tasks)
- **Message**: N/A
- **Files**: N/A

---

## Success Criteria & Final Verification

### Final Integration Test
```bash
# Run the main test script to verify everything works
cd /workspace/TrackNetV3 && python scripts/test.py

# Verify no architecture violations remain
grep -r "from scripts" tracknetv3/ --include="*.py"
# Expected: No output (empty)

# Verify functions are accessible from new location
python -c "from tracknetv3.utils.trajectory import generate_inpaint_mask, linear_interp; print('SUCCESS: All functions imported correctly')"

# Verify functions are exported via utils module
python -c "from tracknetv3.utils import generate_inpaint_mask, linear_interp; print('SUCCESS: Functions exported via utils')"
```

### Project Completion Checklist
- [ ] All "Core Deliverables" (from Context section) are implemented and verified:
  - [ ] `tracknetv3/utils/trajectory.py` exists with both functions
  - [ ] `tracknetv3/utils/__init__.py` exports the new functions
  - [ ] `scripts/test.py` no longer contains the moved functions
  - [ ] `scripts/test.py` imports from the new location
  - [ ] `tracknetv3/inference/offline.py` imports from the new location
- [ ] All constraints have been respected:
  - [ ] Function signatures remain unchanged
  - [ ] No new dependencies added
  - [ ] No backward compatibility layer created
  - [ ] Functions moved (not copied)
- [ ] Nothing in the "Out of Scope" section has been inadvertently implemented:
  - [ ] No changes to function logic/behavior
  - [ ] No addition of unit tests
  - [ ] No modification to `tracknetv3/__init__.py`
  - [ ] No changes to other utility files (`helpers.py`, `general.py`)
- [ ] All verification steps (automated or manual) defined in the TODOs have passed:
  - [ ] Task 0 verification passed
  - [ ] Task 1 verification passed
  - [ ] Task 2 verification passed
  - [ ] Task 3 verification passed
  - [ ] Task 4 final verification passed
  - [ ] `scripts/test.py` runs successfully
  - [ ] No import errors
  - [ ] No architecture violations remain
