#' @name lgb_shared_params
#' @title Shared parameter docs
#' @description Parameter docs shared by \code{lgb.train}, \code{lgb.cv}, and \code{lightgbm}
#' @param callbacks list of callback functions
#'        List of callback functions that are applied at each iteration.
#' @param data a \code{lgb.Dataset} object, used for training
#' @param early_stopping_rounds int
#'        Activates early stopping.
#'        Requires at least one validation data and one metric
#'        If there's more than one, will check all of them except the training data
#'        Returns the model with (best_iter + early_stopping_rounds)
#'        If early stopping occurs, the model will have 'best_iter' field
#' @param eval_freq evaluation output frequency, only effect when verbose > 0
#' @param init_model path of model file of \code{lgb.Booster} object, will continue training from this model
#' @param nrounds number of training rounds
#' @param params List of parameters
#' @param verbose verbosity for output, if <= 0, also will disable the print of evaluation during training
NULL


#' @title Train a LightGBM model
#' @name lightgbm
#' @description Simple interface for training a LightGBM model.
#' @inheritParams lgb_shared_params
#' @param label Vector of labels, used if \code{data} is not an \code{\link{lgb.Dataset}}
#' @param weight vector of response values. If not NULL, will set to dataset
#' @param save_name File name to use when writing the trained model to disk. Should end in ".model".
#' @param ... Additional arguments passed to \code{\link{lgb.train}}. For example
#'     \itemize{
#'        \item{valids}{a list of \code{lgb.Dataset} objects, used for validation}
#'        \item{obj}{objective function, can be character or custom objective function. Examples include
#'                   \code{regression}, \code{regression_l1}, \code{huber},
#'                    \code{binary}, \code{lambdarank}, \code{multiclass}, \code{multiclass}}
#'        \item{eval}{evaluation function, can be (a list of) character or custom eval function}
#'        \item{record}{Boolean, TRUE will record iteration message to \code{booster$record_evals}}
#'        \item{colnames}{feature names, if not null, will use this to overwrite the names in dataset}
#'        \item{categorical_feature}{list of str or int. type int represents index, type str represents feature names}
#'        \item{reset_data}{Boolean, setting it to TRUE (not the default value) will transform the booster model
#'                          into a predictor model which frees up memory and the original datasets}
#'         \item{boosting}{Boosting type. \code{"gbdt"} or \code{"dart"}}
#'         \item{num_leaves}{number of leaves in one tree. defaults to 127}
#'         \item{max_depth}{Limit the max depth for tree model. This is used to deal with
#'                          overfit when #data is small. Tree still grow by leaf-wise.}
#'          \item{num_threads}{Number of threads for LightGBM. For the best speed, set this to
#'                             the number of real CPU cores, not the number of threads (most
#'                             CPU using hyper-threading to generate 2 threads per CPU core).}
#'     }
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
#' @useDynLib lib_lightgbm , .registration = TRUE
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
