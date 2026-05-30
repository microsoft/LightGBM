# LightGBM

[![NuGet Version](https://img.shields.io/nuget/v/lightgbm?logo=nuget&logoColor=white)](https://www.nuget.org/packages/LightGBM)
[![License](https://img.shields.io/github/license/lightgbm-org/lightgbm.svg)](https://github.com/lightgbm-org/LightGBM/blob/master/LICENSE)

LightGBM is a gradient boosting framework that uses tree-based learning algorithms. It is designed to be distributed and efficient with the following advantages:

- Faster training speed and higher efficiency.
- Lower memory usage.
- Better accuracy.
- Support for parallel, distributed, and GPU learning.
- Capable of handling large-scale data.

## About This Package

The `LightGBM` NuGet package provides the native LightGBM shared library (`lib_lightgbm`) for .NET applications on Windows, Linux, and macOS (x64).

It includes:
- `lib_lightgbm.dll` / `lib_lightgbm.so` / `lib_lightgbm.dylib` — the native LightGBM library
- `lightgbm.exe` — the LightGBM CLI executable (Windows only)
- MSBuild integration via `.props` and `.targets` files for automatic native library deployment

## Getting Started

### Installation

```shell
dotnet add package LightGBM
```

### Usage

After installing the package, the native LightGBM library is automatically available in your build output. You can use it with any .NET binding (such as [LightGBM.Net](https://github.com/ralfbrown/LightGBM.Net) or via P/Invoke) to call LightGBM's C API from your .NET application.

```csharp
// Example: Load LightGBM native library via P/Invoke
[DllImport("lib_lightgbm")]
public static extern int LGBM_GetLastError();

// Check that the library loads correctly
var lastError = LGBM_GetLastError();
Console.WriteLine($"LightGBM loaded successfully, last error code: {lastError}");
```

### Resources

- [GitHub Repository](https://github.com/lightgbm-org/LightGBM)
- [Documentation](https://lightgbm.readthedocs.io/)
- [Python Package Documentation](https://lightgbm.readthedocs.io/en/latest/Python-Intro.html)
- [R Package Documentation](https://lightgbm.readthedocs.io/en/latest/R/articles/)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/lightgbm-org/LightGBM/blob/master/LICENSE) file for details.
