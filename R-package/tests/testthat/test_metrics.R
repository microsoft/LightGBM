context(".METRICS_HIGHER_BETTER()")

test_that(".METRICS_HIGHER_BETTER() should be well formed", {
    metrics <- .METRICS_HIGHER_BETTER()
    metric_names <- names(.METRICS_HIGHER_BETTER())
    # should be a logical vector
    expect_true(is.logical(metrics))
    # no metrics should be repeated
    expect_true(length(unique(metric_names)) == length(metrics))
    # should not be any NAs
    expect_false(any(is.na(metrics)))
})
