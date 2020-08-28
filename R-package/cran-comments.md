# CRAN Submission History

## v3.0.0 - Submission 2 - (August 28, 2020)

### CRAN response

Failing pre-checks.

### `R CMD check` results

* Debian: 2 NOTEs

    ```text
    * checking CRAN incoming feasibility ... NOTE
    Maintainer: 'Guolin Ke <guolin.ke@microsoft.com>'

    New submission

    Possibly mis-spelled words in DESCRIPTION:
      Guolin (13:52)
      Ke (13:48)
      LightGBM (14:20)
      al (13:62)
      et (13:59)

    * checking top-level files ... NOTE
    Non-standard files/directories found at top level:
      'docs' 'lightgbm-hex-logo.png' 'lightgbm-hex-logo.svg'
    ```

* Windows: 2 NOTEs

    ```text
    * checking CRAN incoming feasibility ... NOTE
    Maintainer: 'Guolin Ke <guolin.ke@microsoft.com>'

    New submission

    Possibly mis-spelled words in DESCRIPTION:
      Guolin (13:52)
      Ke (13:48)
      LightGBM (14:20)
      al (13:62)
      et (13:59)

    * checking top-level files ... NOTE
    Non-standard files/directories found at top level:
      'docs' 'lightgbm-hex-logo.png' 'lightgbm-hex-logo.svg'
    ```

### Maintainer Notes

We should tell them the misspellings note is a false positive.

For the note about included files, that is my fault. I had extra files laying around when I generated the package. I'm surprised to see `docs/` in that list, since it is ignored in  `.Rbuildignore`. I even tested that with [the exact code Rbuildignore uses](https://github.com/wch/r-source/blob/9d13622f41cfa0f36db2595bd6a5bf93e2010e21/src/library/tools/R/build.R#L85). For now, I added `rm -r  docs/` to `build-cran-package.sh`. We can figure out what is happening with `.Rbuildignore` in the future, but it shouldn't block a release.

## v3.0.0 - Submission 1 - (August 24, 2020)

NOTE: 3.0.0-1 was never released to CRAN. CRAN was on vacation August 14-24, 2020, and in that time version 3.0.0-1 (a release candidate) became 3.0.0.

### CRAN respoonse

> Please only ship the CRAN template for the MIT license.

> Is there some reference about the method you can add in the Description field in the form Authors (year) doi:.....?

> Please fix and resubmit.

### `R CMD check` results

* Debian: 1 NOTE

    ```text
    * checking CRAN incoming feasibility ... NOTE
    Maintainer: ‘Guolin Ke <guolin.ke@microsoft.com>’

    New submission

    License components with restrictions and base license permitting such:
      MIT + file LICENSE
    ```

* Windows: 1 NOTE

    ```text
    * checking CRAN incoming feasibility ... NOTE
    Maintainer: 'Guolin Ke <guolin.ke@microsoft.com>'

    New submission

    License components with restrictions and base license permitting such:
      MIT + file LICENSE
    ```

### Maintainer Notes

Tried updating `LICENSE` file to this template:

```yaml
YEAR: 2016
COPYRIGHT HOLDER: Microsoft Corporation
```

Added a citation and link for [the main paper](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision) in `DESCRIPTION`.

## v3.0.0-1 - Submission 3 - (August 12, 2020)

### CRAN response

Failing pre-checks.

### `R CMD check` results

* Debian: 1 NOTE

    ```text
    * checking CRAN incoming feasibility ... NOTE
    Maintainer: ‘Guolin Ke <guolin.ke@microsoft.com>’

    New submission

    License components with restrictions and base license permitting such:
      MIT + file LICENSE
    ```

* Windows: 1 ERROR, 1 NOTE

    ```text
    * checking CRAN incoming feasibility ... NOTE
    Maintainer: ‘Guolin Ke <guolin.ke@microsoft.com>’

    New submission

    License components with restrictions and base license permitting such:
      MIT + file LICENSE

    ** running tests for arch 'i386' ... [9s] ERROR
      Running 'testthat.R' [8s]
    Running the tests in 'tests/testthat.R' failed.
    Complete output:
      > library(testthat)
      > library(lightgbm)
      Loading required package: R6
      >
      > test_check(
      +     package = "lightgbm"
      +     , stop_on_failure = TRUE
      +     , stop_on_warning = FALSE
      + )
      -- 1. Error: predictions do not fail for integer input (@test_Predictor.R#7)  --
      lgb.Dataset.construct: cannot create Dataset handle
      Backtrace:
       1. lightgbm::lgb.train(...)
       2. data$construct()
    ```

### Maintainer Notes

The "checking CRAN incoming feasibility" NOTE can be safely ignored. It only shows up the first time you submit a package to CRAN.

So the only thing I see broken right now is the test error on 32-bit Windows. This is documented in https://github.com/microsoft/LightGBM/issues/3187.

## v3.0.0-1 - Submission 2 - (August 10, 2020)

### CRAN response

Failing pre-checks.

### `R CMD check` results

* Debian: 2 NOTEs

    ```text
    * checking CRAN incoming feasibility ... NOTE
    Maintainer: ‘Guolin Ke <guolin.ke@microsoft.com>’

    New submission

    License components with restrictions and base license permitting such:
      MIT + file LICENSE

    Non-standard files/directories found at top level:
    ‘cran-comments.md’ ‘docs’
    ```

* Windows: 1 ERROR, 2 NOTEs

    ```text
    * checking CRAN incoming feasibility ... NOTE
    Maintainer: 'Guolin Ke <guolin.ke@microsoft.com>'

    New submission

    License components with restrictions and base license permitting such:
      MIT + file LICENSE

    * checking top-level files ... NOTE
    Non-standard files/directories found at top level:
      'cran-comments.md' 'docs'

    ** checking whether the package can be loaded ... ERROR
    Loading this package had a fatal error status code 1
    Loading log:
    Error: package 'lightgbm' is not installed for 'arch = i386'
    Execution halted
    ```

### Maintainer Notes

Seems removing `Biarch` field didn't work. Noticed this in the install logs:

> Warning: this package has a non-empty 'configure.win' file, so building only the main architecture

Tried adding `Biarch: true` to `DESCRIPTION` to overcome this.

NOTE about non-standard files was the result of a mistake in `.Rbuildignore` syntax, and something strange with how `cran-comments.md` line in `.Rbuildignore`  was treated. Updated `.Rbuildignore` and added an `rm cran-comments.md` to `build-cran-package.sh`.

## v3.0.0-1 - Submission 1 - (August 9, 2020)

### CRAN response

Failing pre-checks.

### `R CMD check` results

* Debian: 1 NOTE

    ```text
    Possibly mis-spelled words in DESCRIPTION:
      LightGBM (12:88, 19:41, 20:60, 20:264)
    ```

* Windows: 1 ERROR, 1 NOTE

    ```text
    Possibly mis-spelled words in DESCRIPTION:
      LightGBM (12:88, 19:41, 20:60, 20:264)

    ** checking whether the package can be loaded ... ERROR
    Loading this package had a fatal error status code 1
    Loading log:
    Error: package 'lightgbm' is not installed for 'arch = i386'
    Execution halted
    ```

### Maintainer Notes

Thought the issue on Windows was caused by `Biarch: false` in `DESCRIPTION`. Removed `Biarch` field.

Thought the "misspellings" issue could be resolved by adding single quotes around LightGBM, like `'LightGBM'`.
