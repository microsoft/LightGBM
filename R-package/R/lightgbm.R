#' @name lgb_shared_params
#' @title Shared parameter docs
#' @description Parameter docs shared by \code{lgb.train}, \code{lgb.cv}, and \code{lightgbm}
#' @param callbacks List of callback functions that are applied at each iteration.
#' @param data a \code{lgb.Dataset} object, used for training. Some functions, such as \code{\link{lgb.cv}},
#'             may allow you to pass other types of data like \code{matrix} and then separately supply
#'             \code{label} as a keyword argument.
#' @param early_stopping_rounds int. Activates early stopping. When this parameter is non-null,
#'                              training will stop if the evaluation of any metric on any validation set
#'                              fails to improve for \code{early_stopping_rounds} consecutive boosting rounds.
#'                              If training stops early, the returned model will have attribute \code{best_iter}
#'                              set to the iteration number of the best iteration.
#' @param eval evaluation function(s). This can be a character vector, function, or list with a mixture of
#'             strings and functions.
#'
#'             \itemize{
#'                 \item{\bold{a. character vector}:
#'                     If you provide a character vector to this argument, it should contain strings with valid
#'                     evaluation metrics.
#'                     See \href{https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric}{
#'                     The "metric" section of the documentation}
#'                     for a list of valid metrics.
#'                 }
#'                 \item{\bold{b. function}:
#'                      You can provide a custom evaluation function. This
#'                      should accept the keyword arguments \code{preds} and \code{dtrain} and should return a named
#'                      list with three elements:
#'                      \itemize{
#'                          \item{\code{name}: A string with the name of the metric, used for printing
#'                              and storing results.
#'                          }
#'                          \item{\code{value}: A single number indicating the value of the metric for the
#'                              given predictions and true values
#'                          }
#'                          \item{
#'                              \code{higher_better}: A boolean indicating whether higher values indicate a better fit.
#'                              For example, this would be \code{FALSE} for metrics like MAE or RMSE.
#'                          }
#'                      }
#'                 }
#'                 \item{\bold{c. list}:
#'                     If a list is given, it should only contain character vectors and functions.
#'                     These should follow the requirements from the descriptions above.
#'                 }
#'             }
#' @param eval_freq evaluation output frequency, only effect when verbose > 0
#' @param init_model path of model file of \code{lgb.Booster} object, will continue training from this model
#' @param nrounds number of training rounds
#' @param obj objective function, can be character or custom objective function. Examples include
#'            \code{regression}, \code{regression_l1}, \code{huber},
#'            \code{binary}, \code{lambdarank}, \code{multiclass}, \code{multiclass}
#' @param params a list of parameters. See \href{https://lightgbm.readthedocs.io/en/latest/Parameters.html}{
#'               the "Parameters" section of the documentation} for a list of parameters and valid values.
#' @param verbose verbosity for output, if <= 0, also will disable the print of evaluation during training
#' @param serializable whether to make the resulting objects serializable through functions such as
#' \code{save} or \code{saveRDS} (see section "Model serialization").
#' @section Early Stopping:
#'
#'          "early stopping" refers to stopping the training process if the model's performance on a given
#'          validation set does not improve for several consecutive iterations.
#'
#'          If multiple arguments are given to \code{eval}, their order will be preserved. If you enable
#'          early stopping by setting \code{early_stopping_rounds} in \code{params}, by default all
#'          metrics will be considered for early stopping.
#'
#'          If you want to only consider the first metric for early stopping, pass
#'          \code{first_metric_only = TRUE} in \code{params}. Note that if you also specify \code{metric}
#'          in \code{params}, that metric will be considered the "first" one. If you omit \code{metric},
#'          a default metric will be used based on your choice for the parameter \code{obj} (keyword argument)
#'          or \code{objective} (passed into \code{params}).
#' @section Model serialization:
#'
#'          LightGBM model objects can be serialized and de-serialized through functions such as \code{save}
#'          or \code{saveRDS}, but similarly to libraries such as 'xgboost', serialization works a bit differently
#'          from typical R objects. In order to make models serializable in R, a copy of the underlying C++ object
#'          as serialized raw bytes is produced and stored in the R model object, and when this R object is
#'          de-serialized, the underlying C++ model object gets reconstructed from these raw bytes, but will only
#'          do so once some function that uses it is called, such as \code{predict}. In order to forcibly
#'          reconstruct the C++ object after deserialization (e.g. after calling \code{readRDS} or similar), one
#'          can use the function \link{lgb.restore_handle} (for example, if one makes predictions in parallel or in
#'          forked processes, it will be faster to restore the handle beforehand).
#'
#'          Producing and keeping these raw bytes however uses extra memory, and if they are not required,
#'          it is possible to avoid producing them by passing `serializable=FALSE`. In such cases, these raw
#'          bytes can be added to the model on demand through function \link{lgb.make_serializable}.
#' @keywords internal
NULL

#' @rdname lightgbm
#' @title Train a LightGBM model
#' @description Simplified interface for training / fitting a LightGBM model which follows typical
#'              R idioms for model fitting and predictions. Note that this interface does not
#'              expose the full spectrum of library features as \link{lgb.train} does.
#' @details This is a thin wrapper over \link{lgb.Dataset} and then \link{lgb.train} which performs
#'          extra steps such as  automatically detecting categorical variables and handling their
#'          encoding. It is intended as an easy-to-use interface that follows common R idioms for
#'          predictive models.
#'
#'          It uses base R's functions for processing the data, such as `factor`, which are not
#'          particularly efficient - for serious usage, it is recommended to use the \link{lgb.train}
#'          interface with \link{lgb.Dataset} instead, handling aspects such as encoding of categorical
#'          variables externally through your favorite tools.
#'
#'          \bold{Important:} using the `formula` interface relies on R's own formula handling, which
#'          might be very slow for large inputs and will dummy-encode all categorical variables
#'          (meaning: they will not be treated as categorical in tree splits, rather each level will be
#'          treated as a separate variable, without exploiting the sparsity and independence patterns
#'          in the encoded data).
#'
#'          When models are produced through this interface (as opposed to \link{lgb.train}), the
#'          method \link{predict.lgb.Booster} will additionally gain new behaviors, such as taking
#'          columns by name from the new input data or adding names to the resulting predicted matrices
#'          (based on the classes or features depending on what is being predicted).
#' @inheritParams lgb_shared_params
#' @param formula A formula for specifying the response/label and predictors/features in the
#'                model to be fitted. This is provided for ease of use, but using the `formula` interface
#'                is discouraged for a couple reasons (see details section for mode details):\itemize{
#'                \item It converts all factor variables to dummy encoding, which typically does not lead to
#'                      models as good as those in which categorical variables are treated as such.
#'                \item It uses base R's formula handling for inputs, which can be particularly
#'                      computationally inefficient compared to the alternatives.
#'                \item If the number of variables is large, it can increase model size quite a bit.
#'                }
#'
#'                If using the `formula` interface, then `data` must be a `data.frame`.
#' @param data A `data.frame`. In the non-formula interface, it will use all available variables
#'             (those not specified as being `label`, `weight`, or `init_score`) as features / predictors,
#'             and will assume their types are:\itemize{
#'             \item Numeric, if they are of type `numeric`, `integer`, `Date`, `POSIXct`.
#'             \item Categorical, if they are of type `factor`, `character`.
#'             }
#'
#'             Other variable types are not accepted. Note that the underlying core library only accepts
#'             `numeric` inputs, thus other types will end up being casted.
#'
#'             Note that, if using the `data.frame` interface, it is not possible to manually specify
#'             categorical variables through `params` - instead, these will be deduced from the data types,
#'             and their encoding will be handled internally in the fitting and prediction functions.
#'             Under the `data.frame` interface, if the data contains any categorical variables, then at
#'             prediction time only `data.frame` inputs will be allowed.
#' @param X Data features / covariates / predictors with which the model will try to predict `y`.
#'
#'          Note that, if using non-standard evaluation for `y`, `weights`, or `init_score` (specifying
#'          them as column names from `X`), then `X` will be subsetted, and any additional parameters
#'          passed that correspond to column indices (such as per-column `max_bin` or
#'          `categorical_features`) will be applied on the subsetted data.
#'
#'          Supports dense matrices from base R (class `matrix`, will be casted to `double` storage
#'          mode if it isn't already) and sparse matrices in CSC format from the `Matrix` package
#'          (class `dgCMatrix`).
#' @param y,label Target / response variable to predict. May be passed as:\itemize{
#'                \item The name of a column from `X` / `data`, if it has column names. Will use non-standard
#'                      evaluation in order to try to determine if it matches with the name of a column in
#'                      `X` / `data` (i.e. will accept it as the name of a column without putting quotes
#'                      around it), and can also be passed as a character.
#'                 \item A vector with the number of entries matching to the number of rows in `X` / `data`.
#'                 }
#'                 If passing `objective="auto"`, the optimization objective will be determined according to
#'                 the type / class of this variable.
#'
#'                 If `y` is passed as a factor, then `num_class` in `params` will be set automatically
#'                 according to its levels.
#'
#'                 Passing `y` as a factor will also make \link{predict.lgb.Booster} use its levels in the
#'                 outputs from predictions when appropriate.
#' @param weights Sample / observation weights for rows in `X` / `data`. Same format as
#'                `y` (i.e. accepts non-standard evaluation for column names, and accepts numeric vectors).
#' @param init_score Initial values for each observation from which the boosting process will
#'                   be started (e.g. as the result of some previous model). If not passing it (the default),
#'                   will start from a blank state.
#' @param objective Optimization objective (e.g. `"regression"`, `"binary"`, etc.).
#'                  For a list of accepted objectives, see
#'                  \href{https://lightgbm.readthedocs.io/en/latest/Parameters.html}{
#'                  the "Parameters" section of the documentation}.
#'
#'                  If passing `"auto"`, will be deduced from the type of `y` / `label`:\itemize{
#'                  \item If `y` is not a factor, will set the objective to `"regression"`.
#'                  \item If `y` is a factor with two classes, will set the objective to `"binary"`.
#'                  \item If `y` is a factor with more than two classes, will set the objective to `"multiclass"`.
#'                  }
#'
#'                  If `y` is a factor, then it will automatically set parameter `num_classes` based on
#'                  its number of levels, overriding any such entry in `params` if it is present there.
#' @param nthreads Number of parallel threads to use. For best speed, this should be set to the number of
#'                 physical cores in the CPU - in a typical x86-64 machine, this corresponds to half the
#'                 number of maximum threads (e.g. `nthreads = max(parallel::detectCores() / 2L, 1L)` as
#'                 a shorthand for the optimal value).
#'
#'                 Be aware that using too many threads can result in speed degradation in smaller datasets
#'                 (see the parameters documentation for more details).
#'
#'                 If passing zero, will use the default number of threads configured for OpenMP.
#'
#'                 This parameter overrides `num_threads` in `params` if it exists there.
#' @param dataset_params Extra parameters to pass to \link{lgb.Dataset} once it comes the
#'                       time to convert the dataset to this library's internal format.
#'
#'                       For a list of the accepted parameters, see
#'                       \href{https://lightgbm.readthedocs.io/en/latest/Parameters.html#io-parameters}{
#'                       the "I/O Parameters" section of the documentation}.
#' @param save_name File name to use when writing the trained model to disk. Should end in ".model".
#'                  If passing `NULL`, will not save the trained model to disk.
#' @param ... Additional arguments passed to \code{\link{lgb.train}}. For example:
#'     \itemize{
#'        \item{\code{valids}: a list of \code{lgb.Dataset} objects, used for validation}
#'        \item{\code{eval}: evaluation function, can be (a list of) character or custom eval function}
#'        \item{\code{record}: Boolean, TRUE will record iteration message to \code{booster$record_evals}}
#'        \item{\code{colnames}: feature names, if not null, will use this to overwrite the names in dataset}
#'        \item{\code{categorical_feature}: categorical features. This can either be a character vector of feature
#'              names or an integer vector with the indices of the features (e.g. \code{c(1L, 10L)} to
#'              say "the first and tenth columns"). This parameter is not supported in the `formula` and
#'              `data.frame` interfaces.}
#'        \item{\code{reset_data}: Boolean, setting it to TRUE (not the default value) will transform the booster model
#'                          into a predictor model which frees up memory and the original datasets}
#'     }
#' @inheritSection lgb_shared_params Early Stopping
#' @return A trained \code{lgb.Booster} model object.
#' @importFrom utils head
#' @importFrom parallel detectCores
#' @examples
#' library(lightgbm)
#' data("iris")
#' model <- lightgbm(Species ~ ., data = iris, verbose = -1L, nthreads = 1L)
#' pred <- predict(model, iris, type = "class")
#' all(pred == iris$Species)
#'
#' model <- lightgbm(iris, Species, verbose = -1L, nthreads = 1L)
#' head(predict(model, iris, type = "score"))
#'
#' model <- lightgbm(as.matrix(iris[, -5L]), iris$Species, verbose = -1L, nthreads = 1L)
#' head(predict(model, iris, type = "raw"))
#' @export
lightgbm <- function(...) {
  UseMethod("lightgbm")
}

#' @rdname lightgbm
#' @export
lightgbm.formula <- function(formula,
                             data,
                             weights = NULL,
                             init_score = NULL,
                             objective = "auto",
                             nrounds = 100L,
                             nthreads = parallel::detectCores(),
                             params = list(),
                             dataset_params = list(),
                             verbose = 1L,
                             eval_freq = 1L,
                             early_stopping_rounds = NULL,
                             save_name = NULL,
                             serializable = TRUE,
                             ...
                             ) {
  data_processor_outputs <- new.env()
  data_processor <- DataProcessor$new(
    data_processor_outputs
    , data
    , dataset_params
    , model_formula = formula
    , label = NULL
    , weights = weights
    , init_score = init_score
  )
  return(
    lightgbm_internal(
      data_processor_outputs = data_processor_outputs
      , data_processor = data_processor
      , objective = objective
      , nthreads = nthreads
      , params = params
      , nrounds = nrounds
      , verbose = verbose
      , eval_freq = eval_freq
      , early_stopping_rounds = early_stopping_rounds
      , save_name = save_name
      , serializable = serializable
      , ...
    )
  )
}

#' @rdname lightgbm
#' @export
lightgbm.data.frame <- function(data,
                                label,
                                weights = NULL,
                                init_score = NULL,
                                objective = "auto",
                                nrounds = 100L,
                                nthreads = parallel::detectCores(),
                                params = list(),
                                dataset_params = list(),
                                verbose = 1L,
                                eval_freq = 1L,
                                early_stopping_rounds = NULL,
                                save_name = NULL,
                                serializable = TRUE,
                                ...) {
  if (!is.null(params$categorical_feature) || !is.null(dataset_params$categorical_feature)) {
    stop("'categorical_feature' is not supported for 'data.frame' inputs in 'lightgbm()'.")
  }
  data_processor_outputs <- new.env()
  data_processor <- DataProcessor$new(
    data_processor_outputs
    , as.data.frame(data)
    , dataset_params
    , model_formula = NULL
    , label = label
    , weights = weights
    , init_score = init_score
  )
  return(
    lightgbm_internal(
      data_processor_outputs = data_processor_outputs
      , data_processor = data_processor
      , objective = objective
      , nthreads = nthreads
      , params = params
      , nrounds = nrounds
      , verbose = verbose
      , eval_freq = eval_freq
      , early_stopping_rounds = early_stopping_rounds
      , save_name = save_name
      , serializable = serializable
      , ...
    )
  )
}

#' @rdname lightgbm
#' @export
lightgbm.matrix <- function(X,
                            y,
                            weights = NULL,
                            init_score = NULL,
                            objective = "auto",
                            nrounds = 100L,
                            nthreads = parallel::detectCores(),
                            params = list(),
                            dataset_params = list(),
                            verbose = 1L,
                            eval_freq = 1L,
                            early_stopping_rounds = NULL,
                            save_name = NULL,
                            serializable = TRUE,
                            ...) {
  data_processor_outputs <- new.env()
  data_processor <- DataProcessor$new(
    data_processor_outputs
    , X
    , dataset_params
    , model_formula = NULL
    , label = y
    , weights = weights
    , init_score = init_score
  )
  return(
    lightgbm_internal(
      data_processor_outputs = data_processor_outputs
      , data_processor = data_processor
      , objective = objective
      , nthreads = nthreads
      , params = params
      , nrounds = nrounds
      , verbose = verbose
      , eval_freq = eval_freq
      , early_stopping_rounds = early_stopping_rounds
      , save_name = save_name
      , serializable = serializable
      , ...
    )
  )
}

#' @rdname lightgbm
#' @export
lightgbm.dgCMatrix <- function(X,
                               y,
                               weights = NULL,
                               init_score = NULL,
                               objective = "auto",
                               nrounds = 100L,
                               nthreads = parallel::detectCores(),
                               params = list(),
                               dataset_params = list(),
                               verbose = 1L,
                               eval_freq = 1L,
                               early_stopping_rounds = NULL,
                               save_name = NULL,
                               serializable = TRUE,
                               ...) {
  data_processor_outputs <- new.env()
  data_processor <- DataProcessor$new(
    data_processor_outputs
    , X
    , dataset_params
    , model_formula = NULL
    , label = y
    , weights = weights
    , init_score = init_score
  )
  return(
    lightgbm_internal(
      data_processor_outputs = data_processor_outputs
      , data_processor = data_processor
      , objective = objective
      , nthreads = nthreads
      , params = params
      , nrounds = nrounds
      , verbose = verbose
      , eval_freq = eval_freq
      , early_stopping_rounds = early_stopping_rounds
      , save_name = save_name
      , serializable = serializable
      , ...
    )
  )
}

lightgbm_internal <- function(data_processor_outputs,
                              data_processor,
                              objective,
                              nthreads,
                              params = list(),
                              nrounds = 100L,
                              verbose = 1L,
                              eval_freq = 1L,
                              early_stopping_rounds = NULL,
                              save_name = "lightgbm.model",
                              serializable = TRUE,
                              ...) {
  if (objective == "auto") {
    objective <- data_processor_outputs$objective
  }
  if (objective %in% c("multiclass", "multiclassova") && NROW(data_processor$label_levels)) {
    if (!is.null(params$num_class)) {
      warning("'num_class' is overriden when using 'lightgbm()' interface with factors.")
    }
    params$num_class <- length(data_processor$label_levels)
  }
  params$num_threads <- nthreads

  # validate inputs early to avoid unnecessary computation
  if (nrounds <= 0L) {
    stop("nrounds should be greater than zero")
  }

  dtrain <- data_processor_outputs$dataset

  train_args <- list(
    "params" = params
    , "data" = dtrain
    , "nrounds" = nrounds
    , "obj" = objective
    , "verbose" = verbose
    , "eval_freq" = eval_freq
    , "early_stopping_rounds" = early_stopping_rounds
    , "serializable" = serializable
  )
  train_args <- append(train_args, list(...))

  if (! "valids" %in% names(train_args)) {
    train_args[["valids"]] <- list()
  }

  # Set validation as oneself
  if (verbose > 0L) {
    train_args[["valids"]][["train"]] <- dtrain
  }

  # Train a model using the regular way
  bst <- do.call(
    what = lgb.train
    , args = train_args
  )
  bst$data_processor <- data_processor

  # Store model under a specific name
  if (!is.null(save_name)) {
    bst$save_model(filename = save_name)
  }

  return(bst)
}

#' @name agaricus.train
#' @title Training part from Mushroom Data Set
#' @description This data set is originally from the Mushroom data set,
#'              UCI Machine Learning Repository.
#'              This data set includes the following fields:
#'
#'               \itemize{
#'                   \item{\code{label}: the label for each record}
#'                   \item{\code{data}: a sparse Matrix of \code{dgCMatrix} class, with 126 columns.}
#'                }
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
#' @usage data(agaricus.train)
#' @format A list containing a label vector, and a dgCMatrix object with 6513
#' rows and 127 variables
NULL

#' @name agaricus.test
#' @title Test part from Mushroom Data Set
#' @description This data set is originally from the Mushroom data set,
#'              UCI Machine Learning Repository.
#'              This data set includes the following fields:
#'
#'              \itemize{
#'                  \item{\code{label}: the label for each record}
#'                  \item{\code{data}: a sparse Matrix of \code{dgCMatrix} class, with 126 columns.}
#'              }
#' @references
#' https://archive.ics.uci.edu/ml/datasets/Mushroom
#'
#' Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository
#' [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
#' School of Information and Computer Science.
#'
#' @docType data
#' @keywords datasets
#' @usage data(agaricus.test)
#' @format A list containing a label vector, and a dgCMatrix object with 1611
#' rows and 126 variables
NULL

#' @name bank
#' @title Bank Marketing Data Set
#' @description This data set is originally from the Bank Marketing data set,
#'              UCI Machine Learning Repository.
#'
#'              It contains only the following: bank.csv with 10% of the examples and 17 inputs,
#'              randomly selected from 3 (older version of this dataset with less inputs).
#'
#' @references
#' http://archive.ics.uci.edu/ml/datasets/Bank+Marketing
#'
#' S. Moro, P. Cortez and P. Rita. (2014)
#' A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems
#'
#' @docType data
#' @keywords datasets
#' @usage data(bank)
#' @format A data.table with 4521 rows and 17 variables
NULL

# Various imports
#' @import methods
#' @importFrom Matrix Matrix
#' @importFrom R6 R6Class
#' @useDynLib lib_lightgbm , .registration = TRUE
NULL

# Suppress false positive warnings from R CMD CHECK about
# "unrecognized global variable"
globalVariables(c(
    "."
    , ".N"
    , ".SD"
    , "abs_contribution"
    , "bar_color"
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
