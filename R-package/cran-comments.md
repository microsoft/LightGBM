# CRAN Submission History

## v3.0.0 - Submission 3 - (August 29, 2020)

### CRAN response

* Please write references in the description of the DESCRIPTION file in
the form
  - authors (year) doi:...
  - authors (year) arXiv:...
  - authors (year, ISBN:...)
* if those are not available: authors (year) https:... with no space after 'doi:', 'arXiv:', 'https:' and angle brackets for auto-linking.
* (If you want to add a title as well please put it in quotes: "Title")

* \dontrun{} should only be used if the example really cannot be executed (e.g. because of missing additional software, missing API keys, ...) by the user. That's why wrapping examples in \dontrun{} adds the comment ("# Not run:") as a warning for the user. Does not seem necessary. Please unwrap the examples if they are executable in < 5 sec, or replace
\dontrun{} with \donttest{}.

* Please do not modify the global environment (e.g. by using <<-) in your
functions. This is not allowed by the CRAN policies.

* Please always add all authors, contributors and copyright holders in the Authors@R field with the appropriate roles. From CRAN policies you agreed to: "The ownership of copyright and intellectual property rights of all components of the package must be clear and unambiguous (including from the authors specification in the DESCRIPTION file). Where code is copied (or derived) from the work of others (including from R itself), care must be taken that any copyright/license statements are preserved and authorship is not misrepresented." e.g.: Microsoft Corporation, Dropbox Inc. Please explain in the submission comments what you did about this issue.

Please fix and resubmit

### Maintainer Notes

responded to CRAN with the following:

The paper citation has been adjusted as requested. We were using 'glmnet' as a  guide on how to include the URL but maybe they are no longer in compliance with CRAN policies: https://github.com/cran/glmnet/blob/b1a4b50de01e0cd24343959d7cf86452bac17b26/DESCRIPTION

All authors from the original LightGBM paper have been added to Authors@R as `"aut"`. We have also added Microsoft and DropBox, Inc. as `"cph"` (copyright holders). These roles were chosen based on the guidance in https://journal.r-project.org/archive/2012-1/RJournal_2012-1_Hornik~et~al.pdf.

lightgbm's code does use `<<-`, but it does not modify the global environment.  The uses of `<<-` in R/lgb.interprete.R and R/callback.R are in functions which are called in an environment created by the lightgbm functions that call them, and this operator is used to reach one level up into the calling function's environment.

We chose to wrap our examples in `\dontrun{}` because we found, through testing on https://builder.r-hub.io/ and in our own continuous integration environments, that their run time varies a lot between platforms, and we cannot guarantee that all examples will run in under 5 seconds. We intentionally chose `\dontrun{}` over `\donttest{}` because this item in the R 4.0.0 changelog (https://cran.r-project.org/doc/manuals/r-devel/NEWS.html) seems to indicate that \donttest will be ignored by CRAN's automated checks:

> "`R CMD check --as-cran` now runs \donttest examples (which are run by example()) instead of instructing the tester to do so. This can be temporarily circumvented during development by setting environment variable `_R_CHECK_DONTTEST_EXAMPLES_` to a false value."

We run all examples with `R CMD check --as-cran --run-dontrun` in our continuous integration tests on every commit to the package, so we have high confidence that they are working correctly.

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
