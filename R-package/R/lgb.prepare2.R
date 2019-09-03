#' Data preparator for LightGBM datasets (integer)
#'
#' Attempts to prepare a clean dataset to prepare to put in a \code{lgb.Dataset}. Factors and characters are converted to numeric (specifically: integer). Please use \code{lgb.prepare_rules2} if you want to apply this transformation to other datasets. This is useful if you have a specific need for integer dataset instead of numeric dataset. Note that there are programs which do not support integer-only input. Consider this as a half memory technique which is dangerous, especially for LightGBM.
#'
#' @param data A data.frame or data.table to prepare.
#'
#' @return The cleaned dataset. It must be converted to a matrix format (\code{as.matrix}) for input in \code{lgb.Dataset}.
#'
#' @examples
#' library(lightgbm)
#' data(iris)
#'
#' str(iris)
#'
#' # Convert all factors/chars to integer
#' str(lgb.prepare2(data = iris))
#'
#' \dontrun{
#' # When lightgbm package is installed, and you do not want to load it
#' # You can still use the function!
#' lgb.unloader()
#' str(lightgbm::lgb.prepare2(data = iris))
#' # 'data.frame':	150 obs. of  5 variables:
#' # $ Sepal.Length: num  5.1 4.9 4.7 4.6 5 5.4 4.6 5 4.4 4.9 ...
#' # $ Sepal.Width : num  3.5 3 3.2 3.1 3.6 3.9 3.4 3.4 2.9 3.1 ...
#' # $ Petal.Length: num  1.4 1.4 1.3 1.5 1.4 1.7 1.4 1.5 1.4 1.5 ...
#' # $ Petal.Width : num  0.2 0.2 0.2 0.2 0.2 0.4 0.3 0.2 0.2 0.1 ...
#' # $ Species     : int  1 1 1 1 1 1 1 1 1 1 ...
#' }
#'
#' @export
lgb.prepare2 <- function(data) {

  # data.table not behaving like data.frame
  if (inherits(data, "data.table")) {

    # Get data classes
    list_classes <- vapply(data, class, character(1))

    # Convert characters to factors only (we can change them to numeric after)
    is_char <- which(list_classes == "character")
    if (length(is_char) > 0) {
      data[, (is_char) := lapply(.SD, function(x) {as.integer(as.factor(x))}), .SDcols = is_char]
    }

    # Convert factors to numeric (integer is more efficient actually)
    is_fact <- c(which(list_classes == "factor"), is_char)
    if (length(is_fact) > 0) {
      data[, (is_fact) := lapply(.SD, function(x) {as.integer(x)}), .SDcols = is_fact]
    }

  } else {

    # Default routine (data.frame)
    if (inherits(data, "data.frame")) {

      # Get data classes
      list_classes <- vapply(data, class, character(1))

      # Convert characters to factors to numeric (integer is more efficient actually)
      is_char <- which(list_classes == "character")
      if (length(is_char) > 0) {
        data[is_char] <- lapply(data[is_char], function(x) {as.integer(as.factor(x))})
      }

      # Convert factors to numeric (integer is more efficient actually)
      is_fact <- which(list_classes == "factor")
      if (length(is_fact) > 0) {
        data[is_fact] <- lapply(data[is_fact], function(x) {as.integer(x)})
      }

    } else {

      # What do you think you are doing here? Throw error.
      stop("lgb.prepare: you provided ", paste(class(data), collapse = " & "), " but data should have class data.frame")

    }

  }

  return(data)

}
