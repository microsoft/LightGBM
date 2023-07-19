# LightGBM-internal fix to comply with CRAN policy of only using up to 2 threads in tests and example.
#
# per https://cran.r-project.org/web/packages/policies.html
#
# > If running a package uses multiple threads/cores it must never use more than two simultaneously:
#   the check farm is a shared resource and will typically be running many checks simultaneously.
#
# This mechanism could be removed at any time, and isn't considered part of the public API.
#
# ref for this file:
# * https://r-pkgs.org/testing-design.html#testthat-setup-files
# * https://stackoverflow.com/questions/73648282/how-can-i-set-an-option-in-testthat-before-my-package-is-loaded
# * https://github.com/r-lib/testthat/issues/1702
# * https://stackoverflow.com/questions/12598242/global-variables-in-packages-in-r

.LGB_MAX_THREADS <- 2L
