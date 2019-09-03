#' Data preparator for LightGBM datasets (numeric)
#'
#' Attempts to prepare a clean dataset to prepare to put in a \code{lgb.Dataset}. Factors and characters are converted to numeric without integers. Please use \code{lgb.prepare_rules} if you want to apply this transformation to other datasets.
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
#' str(lgb.prepare(data = iris)) # Convert all factors/chars to numeric
#'
#' \dontrun{
#' # When lightgbm package is installed, and you do not want to load it
#' # You can still use the function!
#' lgb.unloader()
#' str(lightgbm::lgb.prepare(data = iris))
#' # 'data.frame':	150 obs. of  5 variables:
#' # $ Sepal.Length: num  5.1 4.9 4.7 4.6 5 5.4 4.6 5 4.4 4.9 ...
#' # $ Sepal.Width : num  3.5 3 3.2 3.1 3.6 3.9 3.4 3.4 2.9 3.1 ...
#' # $ Petal.Length: num  1.4 1.4 1.3 1.5 1.4 1.7 1.4 1.5 1.4 1.5 ...
#' # $ Petal.Width : num  0.2 0.2 0.2 0.2 0.2 0.4 0.3 0.2 0.2 0.1 ...
#' # $ Species     : num  1 1 1 1 1 1 1 1 1 1 ...
#' }
#'
#' @export
lgb.prepare <- function(data) {

  # data.table not behaving like data.frame
  if ("data.table" %in% class(data)) {

    # Get data classes
    list_classes <- sapply(data, class)

    # Convert characters to factors only (we can change them to numeric after)
    is_char <- which(list_classes == "character")
    if (length(is_char) > 0) {
      data[, (is_char) := lapply(.SD, function(x) {as.numeric(as.factor(x))}), .SDcols = is_char]
    }

    # Convert factors to numeric (integer is more efficient actually)
    is_fact <- c(which(list_classes == "factor"), is_char)
    if (length(is_fact) > 0) {
      data[, (is_fact) := lapply(.SD, function(x) {as.numeric(x)}), .SDcols = is_fact]
    }

  } else {

    # Default routine (data.frame)
    if ("data.frame" %in% class(data)) {

      # Get data classes
      list_classes <- sapply(data, class)

      # Convert characters to factors to numeric (integer is more efficient actually)
      is_char <- which(list_classes == "character")
      if (length(is_char) > 0) {
        data[is_char] <- lapply(data[is_char], function(x) {as.numeric(as.factor(x))})
      }

      # Convert factors to numeric (integer is more efficient actually)
      is_fact <- which(list_classes == "factor")
      if (length(is_fact) > 0) {
        data[is_fact] <- lapply(data[is_fact], function(x) {as.numeric(x)})
      }

    } else {

      # What do you think you are doing here? Throw error.
      stop("lgb.prepare2: you provided ", paste(class(data), collapse = " & "), " but data should have class data.frame")

    }

  }

  return(data)

}
