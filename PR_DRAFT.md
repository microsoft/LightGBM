# Pull Request Handoff

## Selected Issue

- Issue: #6622
- Title: Training fails bagging_freq > 1 and bagging_fraction is very small
- URL: https://github.com/lightgbm-org/LightGBM/issues/6622

This issue was selected because it is a focused correctness bug in the C++ bagging strategy with a small, testable fix. The reported failure is deterministic for a one-row dataset when `bagging_fraction * num_data` is less than one, and the expected behavior is clear: training should continue with at least one sampled data point and explain how to avoid the warning.

The issue is still unresolved. A final read-only `gh issue view 6622 --repo lightgbm-org/LightGBM --json number,state,title,url` query returned `OPEN`, and `upstream/main` still contains the zero-sized bagging count calculation and the subsequent `Dataset(bag_data_cnt_)` construction without a lower bound. The open-PR search `gh pr list --repo lightgbm-org/LightGBM --state open --search '6622 OR bagging_fraction' --limit 100` found no PR addressing this issue. PR #7397 in that result concerns a separate quantized-training histogram bug, not empty bagging samples. No duplicate active PR was found.

## Reproduction

Original behavior can be reproduced with:

```python
import lightgbm as lgb
import numpy as np

X = np.array([[0.0, 1.0]])
y = np.array([1.0])
lgb.train(
    {"seed": 1, "bagging_fraction": 0.5, "bagging_freq": 5},
    lgb.Dataset(X, label=y),
)
```

Expected behavior is for training to return a booster with a warning that the requested fraction is too small, while using one data point for bagging. The original behavior raises `LightGBMError: Check failed: (num_data) > (0)` because the bagging subset is constructed with zero rows.

## Root Cause

`ResetSampleConfig()` calculates the planned bagging size by truncating `bagging_fraction * num_data_`. For very small fractions this produces zero, which is passed to `new Dataset(bag_data_cnt_)`; `Dataset` correctly rejects a zero-row dataset. Independently, the randomized Bernoulli sampling can select zero rows even when the planned count is at least one. Query-based bagging has the analogous possibility of selecting zero queries.

## Implementation

Files changed:

- `src/boosting/bagging.hpp`
- `tests/python_package_test/test_engine.py`
- `PR_DRAFT.md`

The bagging configuration now clamps a zero planned sample count to one and emits a warning. The warning distinguishes ordinary and balanced bagging and, for ordinary bagging, reports the minimum `bagging_fraction` that avoids the warning. The regular data and query sampling paths also provide a one-element fallback when randomized selection returns an empty sample. This keeps the invariant required by downstream tree learners without changing normal sampling behavior.

The implementation stays within the existing `BaggingSampleStrategy`, uses the existing `Random` helper and buffers, and does not add dependencies or change public APIs. The fallback is guarded for datasets without query metadata.

## Tests

Added `test_bagging_does_not_use_empty_sample()` in `tests/python_package_test/test_engine.py`. It reproduces the one-row, tiny-fraction case, verifies that one tree is produced instead of an exception, and checks the user-facing warning. The explicit verbosity setting keeps the assertion isolated from preceding tests that intentionally suppress LightGBM logs.

Additional manual checks exercised ordinary issue reproduction, query-based bagging, and balanced bagging with deterministic seeds that can otherwise produce empty samples; all three completed training successfully.

## Validation

- `git fetch --prune upstream '+refs/heads/main:refs/remotes/upstream/main' '+refs/pull/*/head:refs/remotes/upstream/pr/*'` — passed. `upstream/main` remained at `a7c897696360500d670405857eda7313e366d563`.
- `PATH=/tmp/lightgbm-6622-validation.d0WmAn/venv/bin:$PATH CMAKE_BUILD_PARALLEL_LEVEL=4 sh ./build-python.sh install --no-isolation` — passed before the final test-only verbosity adjustment; the native extension compiled and `lightgbm-4.7.0.99-py3-none-linux_x86_64.whl` was installed.
- `/tmp/lightgbm-6622-validation.d0WmAn/venv/bin/python -m pip install cloudpickle joblib` — passed; `cloudpickle` was added for the repository test utilities and `joblib` was already installed.
- `PATH=/tmp/lightgbm-6622-validation.d0WmAn/venv/bin:$PATH pytest -q tests/python_package_test/test_engine.py::test_bagging_does_not_use_empty_sample` — passed before and after the test-isolation adjustment; final paired validation below also covers the relevant ordering interaction.
- `PATH=/tmp/lightgbm-6622-validation.d0WmAn/venv/bin:$PATH pytest -q tests/python_package_test/test_engine.py::test_rf tests/python_package_test/test_engine.py::test_bagging_does_not_use_empty_sample` — passed, `2 passed in 6.80s`.
- The exact issue reproduction, query-based bagging, and balanced-bagging manual script — all three cases passed and produced one tree.
- `PATH=/tmp/lightgbm-6622-validation.d0WmAn/venv/bin:$PATH pytest --lf -vv --maxfail=1 tests/python_package_test/test_engine.py` — initially identified a test-isolation failure: the new warning assertion saw empty output after `test_rf` left the process-local log level suppressed. Adding `"verbosity": 1` to the regression test resolved that failure; the paired command above passed afterward.
- `PATH=/tmp/lightgbm-6622-validation.d0WmAn/venv/bin:$PATH pytest -q tests/python_package_test/test_engine.py` — not completed. The run reached the new test, reported the test-isolation failure described above, and was stopped before a final full-module result.
- `git diff --check` — passed before handoff staging.

## Limitations

- The complete `test_engine.py` module was not rerun to completion after the test-isolation fix.
- The separate C++ unit-test target, full Python package suite, pre-commit hooks, and CI matrix were not run. No additional build or compile command was run after the user requested that machine-heavy builds stop.
- Validation used the isolated Python 3.12 environment created for this task; other supported Python versions, operating systems, and optional GPU/MPI configurations were not exercised.

## Backward Compatibility

No public API, configuration name, model format, or normal sampling behavior changes. The change only affects configurations that would otherwise produce an empty bagging sample: they now train with one data point and receive a warning explaining how to choose a larger fraction. Balanced and query-based bagging retain their existing selection logic and gain the same non-empty-sample safety invariant.

## Recommended Commit Message

`[c++] Prevent empty bagging samples for tiny bagging fractions (#6622)`

## Recommended Pull Request Title

`[c++] Prevent empty bagging samples for tiny bagging fractions`

## Pull Request Description

Fixes #6622

## Summary

Training with a very small `bagging_fraction` can fail when truncation or random sampling produces an empty bagging sample. This change guarantees that ordinary, balanced, and query-based bagging provide at least one sample to the tree learner.

## Root cause

The configured sample count was calculated with integer truncation, so `bagging_fraction * num_data` could become zero. That zero was used to construct a bagging subset, which correctly failed its positive-row check. Randomized selection could also return zero rows even when the configured count was positive.

## Changes

- Clamp a zero configured bagging count to one and emit an actionable warning.
- Fall back to one sampled row or query when randomized selection returns an empty result.
- Add a regression test for the one-row, tiny-fraction reproduction.

## Validation

- Native Python extension build completed successfully.
- The regression test passed.
- The regression test passed immediately after the existing random-forest test, which verifies log-level isolation.
- Manual issue-reproduction, query-bagging, and balanced-bagging checks passed.

The complete engine test module and separate C++ test target were not rerun to completion in this environment.

## Compatibility

There are no public API or configuration changes. Only previously empty bagging samples are changed; they now use one data point and produce an explanatory warning.
