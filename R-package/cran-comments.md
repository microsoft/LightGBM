# CRAN Submission History

## v3.2.1 - Submission 1 - (TBD)

### CRAN response

### Maintainer Notes

## v3.2.0 - Submission 1 - (March 22, 2021)

### CRAN response

Package is failing checks in the `r-devel-linux-x86_64-debian-clang` environment (described [here](https://cran.r-project.org/web/checks/check_flavors.html#r-devel-linux-x86_64-debian-clang)). Specifically, one unit test on the use of non-ASCII feature names in `Booster$dump_model()` fails.

> Apparently your package fails its checks in a strict Latin-1* locale,
e.g. under Linux using LANG=en_US.iso88591 (see the debian-clang
results).

> Please correct before 2021-04-21 to safely retain your package on CRAN.

### Maintainer Notes

Submitted a version 3.2.1 to correct the errors noted.

## v3.1.1 - Submission 1 - (December 7, 2020)

### CRAN response

Accepted to CRAN, December 8.

### Maintainer Notes

Submitted a fix to 3.1.0 that skips some learning-to-rank tests on 32-bit Windows.

## v3.1.0 - Submission 1 - (November 15, 2020)

### CRAN response

Accepted to CRAN, November 18.

On November 21, found out that the CRAN's `r-oldrel-windows-ix86+x86_64` check was failing, with an issue similar to the one faced on Solaris and fixed in https://github.com/microsoft/LightGBM/pull/3534.

CRAN did not ask for a re-submission, but this was fixed in 3.1.1.

### Maintainer Notes

This package was submitted with the following information in the "optional comments" box.

```text
Hello,

I'm submitting {lightgbm} 3.1.0 on behalf of the maintainer, Guolin Ke. I am a co-author on the package, and he has asked me to handle this submission. We saw in https://cran.r-project.org/web/packages/policies.html#Submission that this is permitted.

{lightgbm} was removed from CRAN in October for issues found by valgrind checks. We have invested significant effort in addressing those issues and creating an automatic test that tries to replicate CRAN's valgrind checks: https://github.com/microsoft/LightGBM/blob/742d72f8bb051105484fd5cca11620493ffb0b2b/.github/workflows/r_valgrind.yml.

We see two warnings from valgrind that we believe are not problematic.

==2063== Conditional jump or move depends on uninitialised value(s)
==2063==    at 0x49CF138: gregexpr_Regexc (grep.c:2439)
==2063==    by 0x49D1F13: do_regexpr (grep.c:3100)
==2063==    by 0x49A0058: bcEval (eval.c:7121)
==2063==    by 0x498B67F: Rf_eval (eval.c:727)
==2063==    by 0x498E414: R_execClosure (eval.c:1895)
==2063==    by 0x498E0C7: Rf_applyClosure (eval.c:1821)
==2063==    by 0x499FC8C: bcEval (eval.c:7089)
==2063==    by 0x498B67F: Rf_eval (eval.c:727)
==2063==    by 0x498B1CB: forcePromise (eval.c:555)
==2063==    by 0x49963AB: FORCE_PROMISE (eval.c:5142)
==2063==    by 0x4996566: getvar (eval.c:5183)
==2063==    by 0x499D1A5: bcEval (eval.c:6873)
==2063==  Uninitialised value was created by a stack allocation
==2063==    at 0x49CEC37: gregexpr_Regexc (grep.c:2369)

This seems to be related to R itself and not any code in {lightgbm}.

==2063== 336 bytes in 1 blocks are possibly lost in loss record 153 of 2,709
==2063==    at 0x483DD99: calloc (in /usr/lib/x86_64-linux-gnu/valgrind/vgpreload_memcheck-amd64-linux.so)
==2063==    by 0x40149CA: allocate_dtv (dl-tls.c:286)
==2063==    by 0x40149CA: _dl_allocate_tls (dl-tls.c:532)
==2063==    by 0x5702322: allocate_stack (allocatestack.c:622)
==2063==    by 0x5702322: pthread_create@@GLIBC_2.2.5 (pthread_create.c:660)
==2063==    by 0x56D0DDA: ??? (in /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0)
==2063==    by 0x56C88E0: GOMP_parallel (in /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0)
==2063==    by 0x1544D29C: LGBM_DatasetCreateFromCSC (c_api.cpp:1286)
==2063==    by 0x1546F980: LGBM_DatasetCreateFromCSC_R (lightgbm_R.cpp:91)
==2063==    by 0x4941E2F: R_doDotCall (dotcode.c:634)
==2063==    by 0x494CCC6: do_dotcall (dotcode.c:1281)
==2063==    by 0x499FB01: bcEval (eval.c:7078)
==2063==    by 0x498B67F: Rf_eval (eval.c:727)
==2063==    by 0x498E414: R_execClosure (eval.c:1895)

We believe this is a false positive, and related to a misunderstanding between valgrind and openmp (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=36298).

We have also added automated tests with ASAN/UBSAN to our testing setup, and have checked the package on Solaris 10 and found no issues.

Thanks for your time and consideration.
```

## v3.0.0.2 - Submission 1 - (September 29, 2020)

### CRAN response

First response was a message talking about failing checks on 3.0.0.

```text
package lightgbm_3.0.0.2.tar.gz has been auto-processed.
The auto-check found additional issues for the last version released on CRAN:
gcc-UBSAN <link>
valgrind <link>
CRAN incoming checks do not test for these additional issues and you will need an appropriately instrumented build of R to reproduce these.
Hence please reply-all and explain: Have these been fixed?

Please correct before 2020-10-05 to safely retain your package on CRAN.

There is still a valgrind error. This did not happen when tested on
submission, but the tests did run until timeout at 4 hours. When you
write illegally, corruption is common.

Illegal writes are serious errors.
```

Then in later responses to email correspondence with CRAN, CRAN expressed frustration with the number of failed submission and banned this package from new submissions for a month.

The content of that frustrated message was regrettable and it does not need to be preserved forever in this file.

### Maintainer Notes

The 3.0.0.x series is officially not making it to CRAN. We will wait until November, and try again.

Detailed plan about what will be tried before November 2020 to increase the likelihood of success for that package: https://github.com/microsoft/LightGBM/pull/3338#issuecomment-702756840.

## v3.0.0.1 - Submission 1 - (September 24, 2020)

### CRAN response

```text
Thanks, we see:

Still lots of alignment errors, such as

lightgbm.Rcheck/tests/testthat.Rout:io/dataset_loader.cpp:340:59:
runtime error: reference binding to misaligned address 0x7f51fefad81e for type 'const value_type', which requires 4 byte alignment
lightgbm.Rcheck/tests/testthat.Rout:/usr/include/c++/10/bits/stl_vector.h:1198:21:
runtime error: reference binding to misaligned address 0x7f51fefad81e for type 'const int', which requires 4 byte alignment lightgbm.Rcheck/tests/testthat.Rout:/usr/include/c++/10/bits/vector.tcc:449:28:runtime
error: reference binding to misaligned address 0x7f51fefad81e for type 'const type', which requires 4 byte alignment
lightgbm.Rcheck/tests/testthat.Rout:/usr/include/c++/10/bits/move.h:77:36:
runtime error: reference binding to misaligned address 0x7f51fefad81e for type 'const int', which requires 4 byte alignment
lightgbm.Rcheck/tests/testthat.Rout:/usr/include/c++/10/bits/alloc_traits.h:512:17:
runtime error: reference binding to misaligned address 0x7f51fefad81e for type 'const type', which requires 4 byte alignment

Please fix and resubmit.
```

### Maintainer Notes

Ok, these are the notes from the UBSAN tests. Was able to reproduce them with https://github.com/microsoft/LightGBM/pull/3338#issuecomment-700399862, and they were fixed in https://github.com/microsoft/LightGBM/pull/3415.

Struggling to replicate the valgrind result (running `R CMD check --use-valgrind` returns no issues), so trying submission again. Hoping that the fixes for mis-alignment fix the other errors too.

## v3.0.0 - Submission 6 - (September 24, 2020)

### CRAN response

Failing pre-checks.

### `R CMD check` results

```text
* checking CRAN incoming feasibility ... WARNING
Maintainer: ‘Guolin Ke <guolin.ke@microsoft.com>’

Insufficient package version (submitted: 3.0.0, existing: 3.0.0)

Days since last update: 4
```

### Maintainer Notes

Did not think the version needed to be incremented if submitting a package in response to CRAN saying "you are failing checks and will be kicked off if you don't fix it", but I guess you do!

This can be fixed by just re-submitting but with the version changed from `3.0.0` to `3.0.0.1`.

## v3.0.0 - Submission 5 - (September 11, 2020)

### CRAN Response

Accepted to CRAN!

Please correct the problems below before 2020-10-05 to safely retain your package on CRAN:

```text
checking installed package size ... NOTE
  installed size is 49.7Mb
  sub-directories of 1Mb or more:
    libs 49.1Mb

"network/socket_wrapper.hpp", line 30: Error: Could not open include file<ifaddrs.h>.
"network/socket_wrapper.hpp", line 216: Error: The type "ifaddrs" is incomplete.
"network/socket_wrapper.hpp", line 217: Error: The type "ifaddrs" is incomplete.
"network/socket_wrapper.hpp", line 220: Error: The type "ifaddrs" is incomplete.
"network/socket_wrapper.hpp", line 222: Error: The type "ifaddrs" is incomplete.
"network/socket_wrapper.hpp", line 214: Error: The function "getifaddrs" must have a prototype.
"network/socket_wrapper.hpp", line 228: Error: The function "freeifaddrs" must have a prototype.
"network/linkers_socket.cpp", line 76: Warning: A non-POD object of type "std::chrono::duration<double, std::ratio<1, 1000>>" passed as a variable argument to function "static LightGBM::Log::Info(const char*, ...)".
7 Error(s) and 1 Warning(s) detected.
*** Error code 2
make: Fatal error: Command failed for target `network/linkers_socket.o'
Current working directory /tmp/RtmpNfaavG/R.INSTALL40a84f70130a/lightgbm/src
ERROR: compilation failed for package ‘lightgbm’
* removing ‘/home/ripley/R/Lib32/lightgbm’
```

### Maintainer Notes

Added a patch that `psutil` has used to fix missing `ifaddrs.h` on Solaris 10: https://github.com/microsoft/LightGBM/issues/629#issuecomment-665091451.

## v3.0.0 - Submission 4 - (September 4, 2020)

### CRAN Response

> Thanks, if the running time is the only reason to wrap the examples in
\donttest, please replace \donttest by \donttest (\donttest examples are
not executed in the CRAN checks).

> Please replace cat() by message() or warning() in your functions (except
for print() and summary() functions). Messages and warnings can be
suppressed if needed.

> Missing Rd-tags:
  lightgbm/man/dimnames.lgb.Dataset.Rd: \value
  lightgbm/man/lgb.Dataset.construct.Rd: \value
  lightgbm/man/lgb.prepare.Rd: \value
  ...

> Please add the tag and explain in detail the returned objects.

### Maintainer Notes

Responded to CRAN with the following:

All examples have been wrapped with `\donttest` as requested. We have replied to Swetlana Herbrandt asking for clarification on the donttest news item in the R 4.0.2 changelog (https://cran.r-project.org/doc/manuals/r-devel/NEWS.html).

All uses of `cat()` have been replaced with `print()`. We chose `print()` over `message()` because it's important that they be written to stdout alongside all the other logs coming from the library's C++ code. `message()` and `warning()` write to stderr.

All exported objects now have `\value{}` statements in their documentation files in `man/`.

**We also replied directly to CRAN's feedback email**

> Swetlana,

> Thank you for your comments. I've just created a new submission that I believe addresses them.

> Can you help us understand something? In your message you said "\donttest examples are
not executed in the CRAN checks)", but in https://cran.r-project.org/doc/manuals/r-devel/NEWS.html  we see the following:

> > "`R CMD check --as-cran` now runs \donttest examples (which are run by example()) instead of instructing the tester to do so. This can be temporarily circumvented during development by setting environment variable `_R_CHECK_DONTTEST_EXAMPLES_` to a false value."

> Could you help us understand how both of those statements can be true?

## v3.0.0 - Submission 3 - (August 29, 2020)

### CRAN response

* Please write references in the description of the DESCRIPTION file in
the form
  - authors (year) doi:...
  - authors (year) arXiv:...
  - authors (year, ISBN:...)
* if those are not available: authors (year) https:... with no space after 'doi:', 'arXiv:', 'https:' and angle brackets for auto-linking.
* (If you want to add a title as well please put it in quotes: "Title")

* \donttest{} should only be used if the example really cannot be executed (e.g. because of missing additional software, missing API keys, ...) by the user. That's why wrapping examples in \donttest{} adds the comment ("# Not run:") as a warning for the user. Does not seem necessary. Please unwrap the examples if they are executable in < 5 sec, or replace
\donttest{} with \donttest{}.

* Please do not modify the global environment (e.g. by using <<-) in your
functions. This is not allowed by the CRAN policies.

* Please always add all authors, contributors and copyright holders in the Authors@R field with the appropriate roles. From CRAN policies you agreed to: "The ownership of copyright and intellectual property rights of all components of the package must be clear and unambiguous (including from the authors specification in the DESCRIPTION file). Where code is copied (or derived) from the work of others (including from R itself), care must be taken that any copyright/license statements are preserved and authorship is not misrepresented." e.g.: Microsoft Corporation, Dropbox Inc. Please explain in the submission comments what you did about this issue.

Please fix and resubmit

### Maintainer Notes

Responded to CRAN with the following:

The paper citation has been adjusted as requested. We were using 'glmnet' as a  guide on how to include the URL but maybe they are no longer in compliance with CRAN policies: https://github.com/cran/glmnet/blob/b1a4b50de01e0cd24343959d7cf86452bac17b26/DESCRIPTION

All authors from the original LightGBM paper have been added to Authors@R as `"aut"`. We have also added Microsoft and DropBox, Inc. as `"cph"` (copyright holders). These roles were chosen based on the guidance in https://journal.r-project.org/archive/2012-1/RJournal_2012-1_Hornik~et~al.pdf.

lightgbm's code does use `<<-`, but it does not modify the global environment. The uses of `<<-` in R/lgb.interprete.R and R/callback.R are in functions which are called in an environment created by the lightgbm functions that call them, and this operator is used to reach one level up into the calling function's environment.

We chose to wrap our examples in `\donttest{}` because we found, through testing on https://builder.r-hub.io/ and in our own continuous integration environments, that their run time varies a lot between platforms, and we cannot guarantee that all examples will run in under 5 seconds. We intentionally chose `\donttest{}` over `\donttest{}` because this item in the R 4.0.0 changelog (https://cran.r-project.org/doc/manuals/r-devel/NEWS.html) seems to indicate that \donttest will be ignored by CRAN's automated checks:

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

### CRAN response

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
