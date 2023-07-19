# ref for this file:
#
# * https://r-pkgs.org/testing-design.html#testthat-setup-files

# LightGBM-internal fix to comply with CRAN policy of only using up to 2 threads in tests and example.
#
# per https://cran.r-project.org/web/packages/policies.html
#
# > If running a package uses multiple threads/cores it must never use more than two simultaneously:
#   the check farm is a shared resource and will typically be running many checks simultaneously.
#
.LGB_MAX_THREADS <- 2L
