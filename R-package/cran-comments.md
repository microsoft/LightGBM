# CRAN Submission History

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

Seems removing `Biarch` field did't work. Noticed this in the install logs:

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
