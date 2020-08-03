context("lgb.convert()")

test_that("lgb.convert() rejects inputs that are not a data.table or data.frame", {
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
            converted_dataset <- lgb.convert(bad_input)
        }, regexp = "lgb.convert: you provided", fixed = TRUE)
    }
})

test_that("lgb.convert() should work correctly for a dataset with only character columns", {
    testDF <- data.frame(
        col1 = c("a", "b", "c")
        , col2 =  c("green", "green", "red")
        , stringsAsFactors = FALSE
    )
    testDT <- data.table::as.data.table(testDF)
    for (input_data in list(testDF, testDT)) {
        converted_dataset <- lgb.convert(input_data)
        expect_identical(class(input_data), class(converted_dataset))
        expect_identical(class(converted_dataset[["col1"]]), "integer")
        expect_identical(class(converted_dataset[["col2"]]), "integer")
        expect_identical(converted_dataset[["col1"]], c(1L, 2L, 3L))
        expect_identical(converted_dataset[["col2"]], c(1L, 1L, 2L))
    }
})

test_that("lgb.convert() should work correctly for a dataset with only factor columns", {
    testDF <- data.frame(
        col1 = as.factor(c("a", "b", "c"))
        , col2 =  as.factor(c("green", "green", "red"))
        , stringsAsFactors = FALSE
    )
    testDT <- data.table::as.data.table(testDF)
    for (input_data in list(testDF, testDT)) {
        converted_dataset <- lgb.convert(input_data)
        expect_identical(class(input_data), class(converted_dataset))
        expect_identical(class(converted_dataset[["col1"]]), "integer")
        expect_identical(class(converted_dataset[["col2"]]), "integer")
        expect_identical(converted_dataset[["col1"]], c(1L, 2L, 3L))
        expect_identical(converted_dataset[["col2"]], c(1L, 1L, 2L))
    }
})

test_that("lgb.convert() should not change a dataset with only integer columns", {
    testDF <- data.frame(
        col1 = 11L:15L
        , col2 = 16L:20L
        , stringsAsFactors = FALSE
    )
    testDT <- data.table::as.data.table(testDF)
    for (input_data in list(testDF, testDT)) {
        converted_dataset <- lgb.convert(input_data)
        expect_identical(converted_dataset, input_data)
    }
})

test_that("lgb.convert() should work correctly for a dataset with numeric, factor, and character columns", {
    testDF <- data.frame(
        character_col = c("a", "b", "c")
        , numeric_col = c(1.0, 9.0, 10.0)
        , factor_col = as.factor(c("n", "n", "y"))
        , stringsAsFactors = FALSE
    )
    testDT <- data.table::as.data.table(testDF)
    for (input_data in list(testDF, testDT)) {
        converted_dataset <- lgb.convert(input_data)
        expect_identical(class(input_data), class(converted_dataset))
        expect_identical(class(converted_dataset[["character_col"]]), "integer")
        expect_identical(class(converted_dataset[["factor_col"]]), "integer")
        expect_identical(converted_dataset[["character_col"]], c(1L, 2L, 3L))
        expect_identical(converted_dataset[["factor_col"]], c(1L, 1L, 2L))

        # today, lgb.convert() does  not convert numeric  columns
        expect_identical(class(converted_dataset[["numeric_col"]]), "numeric")
        expect_identical(converted_dataset[["numeric_col"]], c(1.0, 9.0, 10.0))
    }
})


test_that("lgb.convert() should work correctly for a dataset with all logical columns", {
    testDF <- data.frame(
        all_trues = rep(TRUE, 5L)
        , all_falses = rep(FALSE, 5L)
        , back_and_forth = c(TRUE, FALSE, TRUE, FALSE, TRUE)
    )
    testDT <- data.table::as.data.table(testDF)
    for (input_data in list(testDF, testDT)) {
        converted_dataset <- lgb.convert(input_data)
        expect_identical(class(input_data), class(converted_dataset))
        for (col_name in names(input_data)) {
            expect_identical(class(converted_dataset[[col_name]]), "integer")
        }
        expect_identical(converted_dataset[["all_trues"]], rep(1L, 5L))
        expect_identical(converted_dataset[["all_falses"]], rep(0L, 5L))
        expect_identical(converted_dataset[["back_and_forth"]], c(1L, 0L, 1L, 0L, 1L))
    }
})

test_that("lgb.convert() should convert missing values and should work with columns that have NAs", {
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
        converted_dataset <- lgb.convert(input_data)
        expect_identical(class(input_data), class(converted_dataset))

        expect_identical(class(converted_dataset[["character_col"]]), "integer")
        expect_identical(converted_dataset[["character_col"]], c(1L, 0L, 2L))

        # should not convert integer columns
        expect_identical(class(converted_dataset[["integer_col"]]), "integer")
        expect_identical(converted_dataset[["integer_col"]], c(1L, 9L, NA_integer_))
        expect_identical(class(converted_dataset[["na_int_col"]]), "integer")
        expect_identical(converted_dataset[["na_int_col"]], rep(NA_integer_, nrow(converted_dataset)))

        expect_identical(class(converted_dataset[["factor_col"]]), "integer")
        expect_identical(converted_dataset[["factor_col"]], c(1L, 1L, 2L))


        # even columns of all NAs should be converted if they are character
        expect_identical(class(converted_dataset[["na_character_col"]]), "integer")
        expect_identical(converted_dataset[["na_character_col"]], rep(0L, nrow(converted_dataset)))

        # columns of all logical NAs should have been converted to integer too
        expect_identical(class(converted_dataset[["na_col"]]), "integer")
        expect_identical(converted_dataset[["na_col"]], rep(-1L, 3L))

        # lgb.convert() should convert numeric columns to integer
        expect_identical(class(converted_dataset[["na_real_col"]]), "numeric")
        expect_identical(converted_dataset[["na_real_col"]], rep(NA_real_, nrow(converted_dataset)))
        expect_identical(class(converted_dataset[["numeric_col"]]), "numeric")
        expect_identical(converted_dataset[["numeric_col"]], c(1.0, 9.0, NA_real_))
    }
})

test_that("lgb.convert() should modify data.tables in-place", {
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
    resultDT <- lgb.convert(testDT)
    expect_identical(resultDT, testDT)
})

test_that("lgb.convert() should warn on the presence of columns it cannot convert", {
    testDF <- data.frame(
        character_col = c("a", NA_character_, "c")
        , posix_col = rep(as.POSIXct(Sys.time()), 3L)
    )
    testDT <- data.table::as.data.table(testDF)
    for (input_data in list(testDF, testDT)) {
        expect_warning({
            converted_dataset <- lgb.convert(input_data)
        }, regexp = "columns are not numeric or integer")
        expect_identical(converted_dataset[["character_col"]], c(1L, 0L, 2L))
        expect_identical(converted_dataset[["posix_col"]], input_data[["posix_col"]])
    }
})
