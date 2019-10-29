context("lgb.importance")

test_that("lgb.importance() should reject bad inputs", {
    bad_inputs <- list(
        .Machine$integer.max
        , Inf
        , -Inf
        , NA
        , NA_real_
        , -10L:10L
        , list(c("a", "b", "c"))
        , data.frame(
            x = rnorm(20L)
            , y = sample(
                x = c(1, 2)
                , size = 20L
                , replace = TRUE
            )
        )
        , data.table::data.table(
            x = rnorm(20L)
            , y = sample(
                x = c(1, 2)
                , size = 20L
                , replace = TRUE
            )
        )
        , lgb.Dataset(
            data = matrix(rnorm(100L), ncol = 2)
            , label = matrix(sample(c(0, 1), 50, replace = TRUE))
        )
        , "lightgbm.model"
    )
    for (input in bad_inputs) {
        expect_error({
            lgb.importance(input)
        }, regexp = "'model' has to be an object of class lgb\\.Booster")
    }
})
