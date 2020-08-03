# [description] get all column classes of a data.table or data.frame.
#               This function collapses the result of class() into a single string
.get_column_classes <- function(df) {
    return(
        vapply(
            X = df
            , FUN = function(x) {paste0(class(x), collapse = ",")}
            , FUN.VALUE = character(1L)
        )
    )
}

# [description] check a data frame or data table for columns tthat are any
#               type other than numeric and integer. This is used by lgb.convert()
#               and lgb.convert_with_rules() too warn if more action is needed by users
#               before a dataset can be converted to a lgb.Dataset.
.warn_for_unconverted_columns <- function(df, function_name) {
    column_classes <- .get_column_classes(df)
    unconverted_columns <- column_classes[!(column_classes %in% c("numeric", "integer"))]
    if (length(unconverted_columns) > 0L) {
        col_detail_string <- paste0(
            paste0(
                names(unconverted_columns)
                , " ("
                , unconverted_columns
                , ")"
            )
            , collapse = ", "
        )
        msg <- paste0(
            function_name
            , ": "
            , length(unconverted_columns)
            , " columns are not numeric or integer. These need to be dropped or converted to "
            , "be used in an lgb.Dataset object. "
            , col_detail_string
        )
        warning(msg)
    }
    return(invisible(NULL))
}

.LGB_CONVERT_DEFAULT_FOR_LOGICAL_NA <- function() {return(-1L)}
.LGB_CONVERT_DEFAULT_FOR_NON_LOGICAL_NA <- function() {return(0L)}

#' @name lgb.convert
#' @title Data preparator for LightGBM datasets (integer)
#' @description Attempts to prepare a clean dataset to put in a \code{lgb.Dataset}.
#'              Factor, character, and logical columns are converted to integer.
#'
#'              Missing values in factor and character columns will be replaced with 0. Missing
#'              values in logical columns will be replaced with -1.
#'
#'              Please use \code{\link{lgb.convert_with_rules}} if you want to apply this
#'              transformation to other datasets.
#'
#'              NOTE: In previous releases of LightGBM, this function was called \code{lgb.prepare}.
#' @param data A data.frame or \code{data.table} to prepare. If a \code{data.table} is provided,
#'             it will be modified in place for speed and too avoid out-of-memory erros.
#' @return The cleaned dataset. It must be converted to a matrix format (\code{as.matrix})
#'         for input in \code{lgb.Dataset}. If \code{data} is a \code{data.table}, it will be
#'         modified in place.
#'
#' @examples
#' data(iris)
#'
#' str(iris)
#'
#' # Convert all factors/chars to integer
#' str(lgb.convert(data = iris))
#'
#' \dontrun{
#' # When lightgbm package is installed, and you do not want to load it
#' # You can still use the function!
#' lgb.unloader()
#' str(lightgbm::lgb.convert(data = iris))
#' # 'data.frame':	150 obs. of  5 variables:
#' # $ Sepal.Length: num  5.1 4.9 4.7 4.6 5 5.4 4.6 5 4.4 4.9 ...
#' # $ Sepal.Width : num  3.5 3 3.2 3.1 3.6 3.9 3.4 3.4 2.9 3.1 ...
#' # $ Petal.Length: num  1.4 1.4 1.3 1.5 1.4 1.7 1.4 1.5 1.4 1.5 ...
#' # $ Petal.Width : num  0.2 0.2 0.2 0.2 0.2 0.4 0.3 0.2 0.2 0.1 ...
#' # $ Species     : int  1 1 1 1 1 1 1 1 1 1 ...
#' }
#'
#' @importFrom data.table := is.data.table
#' @export
lgb.convert <- function(data) {

    column_classes <- .get_column_classes(data)

    is_char <- which(column_classes == "character")
    is_factor <- which(column_classes == "factor")
    is_logical <- which(column_classes == "logical")

    is_data_table <- data.table::is.data.table(data)
    is_data_frame <- is.data.frame(data)

    if (!(is_data_table || is_data_frame)) {
        stop(
            "lgb.convert: you provided "
            , paste(class(data), collapse = " & ")
            , " but data should have class data.frame or data.table"
        )
    }

    # data.table not behaving like data.frame
    if (is_data_table) {

        if (length(is_char) > 0L) {
            #data[, (is_char) := lapply(.SD, function(x) {as.integer(as.factor(x))}), .SDcols = is_char]
            for (col_name in names(is_char)) {
                data[, (col_name) := as.integer(as.factor(get(col_name)))]
                data[is.na(get(col_name)), (col_name) := .LGB_CONVERT_DEFAULT_FOR_NON_LOGICAL_NA()]
            }
        }

        if (length(is_factor) > 0L) {
            #data[, (is_factor) := lapply(.SD, function(x) {as.integer(x)}), .SDcols = is_factor]
            for (col_name in names(is_factor)) {
                data[, (col_name) := as.integer(get(col_name))]
                data[is.na(get(col_name)), (col_name) := .LGB_CONVERT_DEFAULT_FOR_NON_LOGICAL_NA()]
            }
        }

        if (length(is_logical) > 0L) {
            # data[, (is_logical) := lapply(.SD, function(x) {as.integer(x)}), .SDcols = is_logical]
            for (col_name in names(is_logical)) {
                data[, (col_name) := as.integer(get(col_name))]
                data[is.na(get(col_name)), (col_name) := .LGB_CONVERT_DEFAULT_FOR_LOGICAL_NA()]
            }
        }

    } else if (is_data_frame) {

        if (length(is_char) > 0L) {
            for (col_name in names(is_char)) {
                data[[col_name]] <- as.integer(as.factor(data[[col_name]]))
                data[is.na(data[col_name]), col_name] <- .LGB_CONVERT_DEFAULT_FOR_NON_LOGICAL_NA()
            }
            #data[is_char] <- lapply(data[is_char], function(x) {as.integer(as.factor(x))})
        }

        if (length(is_factor) > 0L) {
            #data[is_factor] <- lapply(data[is_factor], function(x) {as.integer(x)})
            for (col_name in names(is_factor)) {
                data[[col_name]] <- as.integer(data[[col_name]])
                data[is.na(data[col_name]), col_name] <- .LGB_CONVERT_DEFAULT_FOR_NON_LOGICAL_NA()
            }
        }

        if (length(is_logical) > 0L) {
            #data[is_logical] <- lapply(data[is_logical], function(x) {as.integer(x)})
            for (col_name in names(is_logical)) {
                data[[col_name]] <- as.integer(data[[col_name]])
                data[is.na(data[col_name]), col_name] <- .LGB_CONVERT_DEFAULT_FOR_LOGICAL_NA()
            }
        }

    }

    .warn_for_unconverted_columns(df = data, function_name = "lgb.convert")

    return(data)

}
