# Simple interface for training an lightgbm model.
# Its documentation is combined with lgb.train.
#
#' @rdname lgb.train
#' @export
lightgbm <- function(data, label = NULL, weight = NULL,
                    params = list(), nrounds=10,
                    verbose = 1, eval_freq = 1L, 
                    early_stopping_rounds = NULL,
                    save_name = "lightgbm.model",
                    init_model = NULL, callbacks = list(), ...) {
  dtrain <- data
  if(!lgb.is.Dataset(dtrain)) {
    dtrain <- lgb.Dataset(data, label=label, weight=weight)
  }

  valids <- list()
  if (verbose > 0)
    valids$train = dtrain

  bst <- lgb.train(params, dtrain, nrounds, valids, verbose = verbose, eval_freq=eval_freq,
                   early_stopping_rounds = early_stopping_rounds,
                   init_model = init_model, callbacks = callbacks, ...)
  bst$save_model(save_name)
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

# Various imports
#' @import methods
#' @importFrom R6 R6Class
#' @useDynLib lightgbm
NULL