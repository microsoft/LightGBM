# GitHub Copilot Instructions for LightGBM

## Project Overview

LightGBM is a gradient boosting framework that uses tree-based learning algorithms. It's designed for distributed and efficient machine learning with:
- Fast training speed and high efficiency
- Low memory usage
- Support for parallel, distributed, and GPU learning
- Capability to handle large-scale data

## Repository Structure

### Core Components
- **`src/`**: Core C++ implementation
  - `application/`: Training and prediction logic
  - `boosting/`: Boosting implementations (GBDT, DART, etc.)
  - `io/`: Data structures (Bin, Config, Dataset, DatasetLoader, Feature, Tree)
  - `metric/`: Evaluation metrics
  - `network/`: Network and communication algorithms
  - `objective/`: Objective functions for training
  - `treelearner/`: Tree learning algorithms
  - `utils/`: Common utility functions
  - `cuda/`: CUDA-accelerated implementations

- **`include/`**: C++ header files
  - `include/LightGBM/c_api.h`: C API definitions

- **`python-package/`**: Python package implementation
  - Uses C API via bindings
  - Built with `pyproject.toml`

- **`R-package/`**: R package implementation
  - Uses C API via bindings

- **`tests/`**: Test suites
  - `cpp_tests/`: C++ unit tests using Google Test
  - `python_package_test/`: Python tests using pytest
  - `distributed/`: Distributed learning tests

- **`docs/`**: Documentation (reStructuredText format)
  - Built with Sphinx
  - Hosted at https://lightgbm.readthedocs.io/

## Programming Languages and Standards

### C++ (Core Library)
- **Standard**: C++17
- **Build System**: CMake (minimum version 3.28)
- **Style Guide**: Follow cpplint conventions
  - Use `pre-commit run --all-files` to check style
  - Line length is flexible for headers, but keep reasonable
  - Header guards should follow cpplint patterns

### Python
- **Versions**: Python 3.9+
- **Dependencies**: numpy>=1.17.0, scipy
- **Style**: Use `ruff` for linting and formatting
  - Config in `python-package/pyproject.toml`
  - Run via pre-commit hooks
- **Type Hints**: Use mypy for type checking
- **Testing**: pytest framework

### R
- Standard R package structure
- Tests using testthat

## Key Classes and Architecture

| Class | Purpose |
|-------|---------|
| `Application` | Entry point for training and prediction |
| `Bin` | Stores discretized feature values |
| `Boosting` | Boosting interface (GBDT, DART, etc.) |
| `Config` | Parameter storage and configuration |
| `Dataset` | Dataset information storage |
| `DatasetLoader` | Dataset construction |
| `FeatureGroup` | Feature data storage (can contain multiple features) |
| `Metric` | Evaluation metrics |
| `Network` | Network interfaces and communication |
| `ObjectiveFunction` | Training objective functions |
| `Tree` | Tree model information |
| `TreeLearner` | Tree learning algorithms |

## Build System

### CMake Options
Key build options include:
- `USE_MPI`: Enable MPI-based distributed learning
- `USE_OPENMP`: Enable OpenMP (default ON)
- `USE_GPU`: Enable GPU-accelerated training
- `USE_CUDA`: Enable CUDA-accelerated training
- `USE_DEBUG`: Debug mode
- `USE_SANITIZER`: Enable sanitizers (address, leak, undefined, thread)
- `BUILD_CLI`: Build command-line interface
- `BUILD_CPP_TEST`: Build C++ tests with Google Test
- `BUILD_STATIC_LIB`: Build static library

### Building
```bash
mkdir build && cd build
cmake ..
make -j4
```

For Python package:
```bash
sh build-python.sh
```

## Testing Guidelines

### C++ Tests
- Use Google Test framework
- Located in `tests/cpp_tests/`
- Build with: `cmake -DBUILD_CPP_TEST=ON ..`
- **Strongly recommended**: Build with sanitizers for memory safety

### Python Tests
- Use pytest framework
- Located in `tests/python_package_test/`
- Import test utilities from `tests/python_package_test/utils.py`
- Use fixtures from `conftest.py`
- Example test structure:
  ```python
  def test_feature(tmp_path):
      X_train, y_train = load_data()
      train_data = lgb.Dataset(X_train, label=y_train)
      params = {"objective": "binary", "verbose": -1}
      bst = lgb.train(params, train_data)
      # assertions
  ```

### Test Data
- Example datasets in `examples/` directory
- Use `load_breast_cancer()` from test utils for quick tests

## Code Style and Linting

### Pre-commit Hooks
Run before committing:
```bash
pre-commit run --all-files
```

This runs:
- **cpplint**: C++ style checking
- **ruff**: Python linting and formatting (check + format)
- **mypy**: Python type checking
- **cmakelint**: CMake file linting (max line length: 120)
- **yamllint**: YAML file linting (strict mode)
- **shellcheck**: Shell script checking
- **rstcheck**: reStructuredText validation
- **biome**: JavaScript/JSON formatting
- **typos**: Spell checking

### Editor Configuration
Follow `.editorconfig`:
- **Indentation**: 2 spaces (4 for Python, shell scripts, JS, JSON)
- **Encoding**: UTF-8
- **Line endings**: LF
- **Trim trailing whitespace**: Yes
- **Insert final newline**: Yes
- **Max line length**: 120 (for Python, shell, JS, JSON)

### Python Specific
- Use type hints where appropriate
- Follow ruff configuration in `python-package/pyproject.toml`
- Import lightgbm as `lgb`
- For pandas: Use `lightgbm.compat.pd_DataFrame`, `pd_Series` for compatibility

### C++ Specific
- Include guards: Use `#ifndef` pattern following cpplint
- OpenMP pragmas: Special checking via `.ci/check-omp-pragmas.sh`
- Avoid build/include_subdir warnings (filtered in pre-commit)

## Documentation

### Format
- Documentation uses **reStructuredText** (.rst) format
- Built with Sphinx
- Configuration in `docs/conf.py`

### Building Documentation
```bash
cd docs
sh build-docs.sh
```

### Key Documentation Files
- `docs/Development-Guide.rst`: Development guidelines
- `docs/Installation-Guide.rst`: Installation instructions
- `docs/Parameters.rst`: Parameter reference
- `docs/Features.rst`: Feature descriptions
- `docs/C-API.rst`: C API documentation (generated from c_api.h)

### Adding Documentation
- Use proper reStructuredText syntax
- Run `pre-commit` to validate with rstcheck
- Cross-reference using Sphinx directives

## Common Patterns

### Creating Datasets
```python
import lightgbm as lgb
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = train_data.create_valid(X_test, label=y_test)
```

### Training Models
```python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbose': -1
}
bst = lgb.train(params, train_data, valid_sets=[valid_data])
```

### Using the C API
- See `include/LightGBM/c_api.h` for function signatures
- Check return codes: 0 = success, -1 = error
- Use proper handle management (DatasetHandle, BoosterHandle)

## Dependencies and External Libraries

### Core Dependencies
- **OpenMP**: Parallel processing (optional but recommended)
- **Eigen**: Linear algebra (included in external_libs/)
- **fast_double_parser**: Fast parsing (included in external_libs/)

### Python Package Dependencies
- numpy >= 1.17.0
- scipy

### Optional Dependencies
- **MPI**: For distributed learning
- **CUDA/ROCm**: For GPU acceleration
- **SWIG**: For Java API generation

### Managing Dependencies
- External libraries in `external_libs/` as git submodules
- Don't commit changes to `external_libs/` unless intentional
- Python deps managed via `pyproject.toml`

## Git Workflow

### Branches
- `master`: Main development branch
- Feature branches: Develop new features

### Commits
- Write clear, descriptive commit messages
- Run pre-commit hooks before committing
- Ensure tests pass locally

### Pull Requests
- Reference related issues
- Include tests for new features
- Update documentation as needed
- Wait for CI checks to pass

## CI/CD

### GitHub Actions Workflows
Located in `.github/workflows/`:
- `cpp.yml`: C++ builds and tests
- `python_package.yml`: Python package builds and tests
- `r_package.yml`: R package builds and tests
- `cuda.yml`: CUDA builds
- `static_analysis.yml`: Static analysis checks

### AppVeyor
- Windows-specific builds
- Config in `.appveyor.yml`

## Performance Considerations

- **Memory**: LightGBM is designed for low memory usage
- **Speed**: Optimized for fast training
- **Parallelism**: Use OpenMP, MPI, or GPU for acceleration
- **Binning**: Uses histogram-based learning for efficiency

## Security

- Report security issues via `SECURITY.md`
- Don't commit sensitive data
- Sanitizer builds recommended for memory safety

## Release Process

Documented in `MAINTAINING.md`:
1. Create release PR
2. Merge after approval
3. Wait for CI on master
4. Create GitHub release
5. Upload artifacts
6. Publish to package managers

## Common Anti-Patterns to Avoid

- Don't modify `external_libs/` contents directly
- Don't commit auto-generated files (build artifacts, docs)
- Don't skip pre-commit checks
- Don't use hard-coded paths in tests (use tmp_path fixture)
- Don't add dependencies without checking for security issues
- Don't modify R package's generated files (configure, man/*.Rd)

## Useful Commands

```bash
# Run pre-commit checks
pre-commit run --all-files

# Build and test C++ (with sanitizers)
mkdir build && cd build
cmake -DBUILD_CPP_TEST=ON -DUSE_SANITIZER=ON ..
make -j4
./testlightgbm

# Build Python package
sh build-python.sh

# Run Python tests
pytest tests/python_package_test/

# Check OpenMP pragmas
sh .ci/check-omp-pragmas.sh

# Regenerate parameters
python .ci/parameter-generator.py

# Build documentation
cd docs && sh build-docs.sh
```

## Getting Help

- **FAQ**: `docs/FAQ.rst`
- **Issues**: https://github.com/microsoft/LightGBM/issues
- **Documentation**: https://lightgbm.readthedocs.io/
- **Feature Requests**: See issue #2302

## Contributing

See `CONTRIBUTING.md` for contribution guidelines. Key points:
- Check existing issues before starting work
- Follow the development guide
- Write tests for new features
- Update documentation
- Run pre-commit hooks before submitting PRs
