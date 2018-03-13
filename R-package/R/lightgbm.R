#' Simple interface for training an lightgbm model.
#' Its documentation is combined with lgb.train.
#'
#' @rdname lgb.train
#' @export
lightgbm <- function(data,
                     label = NULL,
                     weight = NULL,
                     params = list(),
                     nrounds = 10,
                     verbose = 1,
                     eval_freq = 1L,
                     early_stopping_rounds = NULL,
                     save_name = "lightgbm.model",
                     init_model = NULL,
                     callbacks = list(),
                     ...) {
  
  # Set data to a temporary variable
  dtrain <- data
  if (nrounds <= 0) {
    stop("nrounds should be greater than zero")
  }
  # Check whether data is lgb.Dataset, if not then create lgb.Dataset manually
  if (!lgb.is.Dataset(dtrain)) {
    dtrain <- lgb.Dataset(data, label = label, weight = weight)
  }

  # Set validation as oneself
  valids <- list()
  if (verbose > 0) {
    valids$train = dtrain
  }
  
  # Train a model using the regular way
  bst <- lgb.train(params, dtrain, nrounds, valids, verbose = verbose, eval_freq = eval_freq,
                   early_stopping_rounds = early_stopping_rounds,
                   init_model = init_model, callbacks = callbacks, ...)
  
  # Store model under a specific name
  bst$save_model(save_name)
  
  # Return booster
  return(bst)
}

#' Training part from Mushroom Data Set
#'
#' This data set is originally from the Mushroom data set,
#' UCI Machine Learning Repository.
#'
#' This data set includes the following fields:
#'
#' \itemize{
#'  \item \code{label} the label for each record
#'  \item \code{data} a sparse Matrix of \code{dgCMatrix} class, with 126 columns.
#' }
#'
#' @references
#' https://archive.ics.uci.edu/ml/datasets/Mushroom
#'
#' Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository
#' [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
#' School of Information and Computer Science.
#'
#' @docType data
#' @keywords datasets
#' @name agaricus.train
#' @usage data(agaricus.train)
#' @format A list containing a label vector, and a dgCMatrix object with 6513
#' rows and 127 variables
NULL

#' Test part from Mushroom Data Set
#'
#' This data set is originally from the Mushroom data set,
#' UCI Machine Learning Repository.
#'
#' This data set includes the following fields:
#'
#' \itemize{
#'  \item \code{label} the label for each record
#'  \item \code{data} a sparse Matrix of \code{dgCMatrix} class, with 126 columns.
#' }
#'
#' @references
#' https://archive.ics.uci.edu/ml/datasets/Mushroom
#'
#' Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository
#' [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
#' School of Information and Computer Science.
#'
#' @docType data
#' @keywords datasets
#' @name agaricus.test
#' @usage data(agaricus.test)
#' @format A list containing a label vector, and a dgCMatrix object with 1611
#' rows and 126 variables
NULL

#' Bank Marketing Data Set
#'
#' This data set is originally from the Bank Marketing data set,
#' UCI Machine Learning Repository.
#'
#' It contains only the following: bank.csv with 10% of the examples and 17 inputs,
#' randomly selected from 3 (older version of this dataset with less inputs).
#'
#' @references
#' http://archive.ics.uci.edu/ml/datasets/Bank+Marketing
#' 
#' S. Moro, P. Cortez and P. Rita. (2014)
#' A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems
#'
#' @docType data
#' @keywords datasets
#' @name bank
#' @usage data(bank)
#' @format A data.table with 4521 rows and 17 variables
NULL

# Various imports
#' @import methods
#' @importFrom R6 R6Class
#' @useDynLib lib_lightgbm
NULL

# Suppress false positive warnings from R CMD CHECK about
# "unrecognized global variable"
globalVariables(c(
    "."
    , ".N"
    , ".SD"
    , "Contribution"
    , "Cover"
    , "Feature"
    , "Frequency"
    , "Gain"
    , "internal_count"
    , "internal_value"
    , "leaf_index"
    , "leaf_parent"
    , "leaf_value"
    , "node_parent"
    , "split_feature"
    , "split_gain"
    , "split_index"
    , "tree_index"
))
