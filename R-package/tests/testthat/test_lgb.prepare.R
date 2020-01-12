context("lgb.prepare()")

test_that("lgb.prepare() rejects inputs that are not a data.table or data.frame", {
    bad_inputs <- list(
        matrix(1.0:10.0, 2L, 5L)
        , TRUE
        , c("a", "b")
        , NA
        , 10L
        , lgb.Dataset(
            data = matrix(1.0:10.0, 2L, 5L)
            , params = list()
        )
    )
    for (bad_input in bad_inputs) {
        expect_error({
            converted_dataset <- lgb.prepare(bad_input)
        }, regexp = "lgb.prepare: you provided", fixed = TRUE)
    }
})

test_that("lgb.prepare() should work correctly for a dataset with only character columns", {
    testDF <- data.frame(
        col1 = c("a", "b", "c")
        , col2 =  c("green", "green", "red")
        , stringsAsFactors = FALSE
    )
    testDT <- data.table::as.data.table(testDF)
    for (input_data in list(testDF, testDT)) {
        converted_dataset <- lgb.prepare(input_data)
        expect_identical(class(input_data), class(converted_dataset))
        expect_identical(class(converted_dataset[["col1"]]), "numeric")
        expect_identical(class(converted_dataset[["col2"]]), "numeric")
        expect_identical(converted_dataset[["col1"]], c(1.0, 2.0, 3.0))
        expect_identical(converted_dataset[["col2"]], c(1.0, 1.0, 2.0))
    }
})

test_that("lgb.prepare() should work correctly for a dataset with only factor columns", {
    testDF <- data.frame(
        col1 = as.factor(c("a", "b", "c"))
        , col2 =  as.factor(c("green", "green", "red"))
        , stringsAsFactors = FALSE
    )
    testDT <- data.table::as.data.table(testDF)
    for (input_data in list(testDF, testDT)) {
        converted_dataset <- lgb.prepare(input_data)
        expect_identical(class(input_data), class(converted_dataset))
        expect_identical(class(converted_dataset[["col1"]]), "numeric")
        expect_identical(class(converted_dataset[["col2"]]), "numeric")
        expect_identical(converted_dataset[["col1"]], c(1.0, 2.0, 3.0))
        expect_identical(converted_dataset[["col2"]], c(1.0, 1.0, 2.0))
    }
})

test_that("lgb.prepare() should not change a dataset with only numeric columns", {
    testDF <- data.frame(
        col1 = 11.0:15.0
        , col2 = 16.0:20.0
        , stringsAsFactors = FALSE
    )
    testDT <- data.table::as.data.table(testDF)
    for (input_data in list(testDF, testDT)) {
        converted_dataset <- lgb.prepare(input_data)
        expect_identical(converted_dataset, input_data)
    }
})

test_that("lgb.prepare() should work correctly for a dataset with numeric, factor, and character columns", {
    testDF <- data.frame(
        character_col = c("a", "b", "c")
        , numeric_col = c(1.0, 9.0, 10.0)
        , factor_col = as.factor(c("n", "n", "y"))
        , stringsAsFactors = FALSE
    )
    testDT <- data.table::as.data.table(testDF)
    for (input_data in list(testDF, testDT)) {
        converted_dataset <- lgb.prepare(input_data)
        expect_identical(class(input_data), class(converted_dataset))
        expect_identical(class(converted_dataset[["character_col"]]), "numeric")
        expect_identical(class(converted_dataset[["numeric_col"]]), "numeric")
        expect_identical(class(converted_dataset[["factor_col"]]), "numeric")
        expect_identical(converted_dataset[["character_col"]], c(1.0, 2.0, 3.0))
        expect_identical(converted_dataset[["numeric_col"]], c(1.0, 9.0, 10.0))
        expect_identical(converted_dataset[["factor_col"]], c(1.0, 1.0, 2.0))
    }
})

test_that("lgb.prepare() should work correctly for a dataset with missing values", {
    testDF <- data.frame(
        character_col = c("a", NA_character_, "c")
        , na_col = rep(NA, 3L)
        , na_real_col = rep(NA_real_, 3L)
        , na_int_col = rep(NA_integer_,  3L)
        , na_character_col = rep(NA_character_, 3L)
        , numeric_col = c(1.0, 9.0, NA_real_)
        , factor_col = as.factor(c("n", "n", "y"))
        , integer_col = c(1L, 9L, NA_integer_)
        , stringsAsFactors = FALSE
    )
    testDT <- data.table::as.data.table(testDF)
    for (input_data in list(testDF, testDT)) {
        converted_dataset <- lgb.prepare(input_data)
        expect_identical(class(input_data), class(converted_dataset))

        expect_identical(class(converted_dataset[["character_col"]]), "numeric")
        expect_identical(converted_dataset[["character_col"]], c(1.0, NA_real_, 2.0))

        expect_identical(class(converted_dataset[["numeric_col"]]), "numeric")
        expect_identical(converted_dataset[["numeric_col"]], c(1.0, 9.0, NA_real_))

        expect_identical(class(converted_dataset[["factor_col"]]), "numeric")
        expect_identical(converted_dataset[["factor_col"]], c(1.0, 1.0, 2.0))

        # NAs of any type should be converted to numeric
        for (col in c("na_real_col", "na_character_col")) {
            expect_identical(class(converted_dataset[[col]]), "numeric")
            expect_identical(converted_dataset[[col]], rep(NA_real_, nrow(converted_dataset)))
        }

        # today, lgb.prepare() does not convert logical columns
        expect_identical(class(converted_dataset[["na_col"]]), "logical")

        # today, lgb.prepare() does not convert integer columns to numeric
        expect_identical(class(converted_dataset[["na_int_col"]]), "integer")
        expect_identical(converted_dataset[["na_int_col"]], rep(NA_integer_, nrow(converted_dataset)))
        expect_identical(class(converted_dataset[["integer_col"]]), "integer")
        expect_identical(converted_dataset[["integer_col"]], c(1L, 9L, NA_integer_))
    }
})

test_that("lgb.prepare() should modify data.tables in-place", {
    testDT <- data.table::data.table(
        character_col = c("a", NA_character_, "c")
        , na_col = rep(NA, 3L)
        , na_real_col = rep(NA_real_, 3L)
        , na_int_col = rep(NA_integer_,  3L)
        , na_character_col = rep(NA_character_, 3L)
        , numeric_col = c(1.0, 9.0, NA_real_)
        , factor_col = as.factor(c("n", "n", "y"))
        , integer_col = c(1L, 9L, NA_integer_)
    )
    resultDT <- lgb.prepare(testDT)
    expect_identical(resultDT, testDT)
})
