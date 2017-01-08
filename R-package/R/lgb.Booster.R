Booster <- R6Class(
  "lgb.Booster",
  cloneable=FALSE,
  public = list(
    best_iter = -1,
    record_evals = list(),
    finalize = function() {
      if(!lgb.is.null.handle(private$handle)){
        print("free booster handle")
        lgb.call("LGBM_BoosterFree_R", ret=NULL, private$handle)
        private$handle <- NULL
      }
    }, 
    initialize = function(params = list(),
                          train_set = NULL,
                          modelfile = NULL,
                          ...) {
      params <- append(params, list(...))
      params_str <- lgb.params2str(params)
      handle <- lgb.new.handle()
      if (!is.null(train_set)) {
        if (!lgb.check.r6.class(train_set, "lgb.Dataset")) {
          stop("lgb.Booster: Only can use lgb.Dataset as training data")
        }
        handle <-
          lgb.call("LGBM_BoosterCreate_R", ret=handle, train_set$.__enclos_env__$private$get_handle(), params_str)
        private$train_set <- train_set
        private$num_dataset <- 1
        private$init_predictor <- train_set$.__enclos_env__$private$predictor
        if (!is.null(private$init_predictor)) {
          lgb.call("LGBM_BoosterMerge_R", ret=NULL,
                handle,
                private$init_predictor$.__enclos_env__$private$handle)
        }
        private$is_predicted_cur_iter <-
          c(private$is_predicted_cur_iter, FALSE)
      } else if (!is.null(modelfile)) {
        if (!is.character(modelfile)) {
          stop("lgb.Booster: Only can use string as model file path")
        }
        handle <-
          lgb.call("LGBM_BoosterCreateFromModelfile_R",
            ret=handle,
            lgb.c_str(modelfile))
      } else {
        stop(
          "lgb.Booster: Need at least one training dataset or model file to create booster instance"
        )
      }
      class(handle) <- "lgb.Booster.handle"
      private$handle <- handle
      private$num_class <- as.integer(1)
      private$num_class <-
        lgb.call("LGBM_BoosterGetNumClasses_R", ret=private$num_class, private$handle)
    },
    set_train_data_name = function(name) {
      private$name_train_set <- name
      return(self)
    },
    add_valid = function(data, name) {
      if (!lgb.check.r6.class(data, "lgb.Dataset")) {
        stop("lgb.Booster.add_valid: Only can use lgb.Dataset as validation data")
      }
      if (!identical(data$.__enclos_env__$private$predictor, private$init_predictor)) {
        stop(
          "lgb.Booster.add_valid: Add validation data failed, you should use same predictor for these data"
        )
      }
      if(!is.character(name)){
        stop("only can use character as data name")
      }
      lgb.call("LGBM_BoosterAddValidData_R", ret=NULL, private$handle, data$.__enclos_env__$private$get_handle())
      private$valid_sets <- c(private$valid_sets, data)
      private$name_valid_sets <- c(private$name_valid_sets, name)
      private$num_dataset <- private$num_dataset + 1
      private$is_predicted_cur_iter <-
        c(private$is_predicted_cur_iter, FALSE)
      return(self)
    },
    reset_parameter = function(params, ...) {
      params <- append(params, list(...))
      params_str <- algb.params2str(params)
      lgb.call("LGBM_BoosterResetParameter_R", ret=NULL,
            private$handle,
            params_str)
      return(self)
    },
    update = function(train_set = NULL, fobj = NULL) {
      if (!is.null(train_set)) {
        if (!lgb.check.r6.class(train_set, "lgb.Dataset")) {
          stop("lgb.Booster.update: Only can use lgb.Dataset as training data")
        }
        if (!identical(train_set$predictor, private$init_predictor)) {
          stop(
            "lgb.Booster.update: Change train_set failed, you should use same predictor for these data"
          )
        }
        lgb.call("LGBM_BoosterResetTrainingData_R", ret=NULL,
              private$handle,
              train_set$.__enclos_env__$private$get_handle())
        private$train_set = train_set
      }
      if (is.null(fobj)) {
        ret <-
          lgb.call("LGBM_BoosterUpdateOneIter_R", ret=NULL, private$handle)
      } else {
        if (typeof(fobj) != 'closure') {
          stop("lgb.Booster.update: fobj should be a function")
        }
        gpair <- fobj(private$inner_predict(1), private$train_set)
        ret <-
          lgb.call(
            "LGBM_BoosterUpdateOneIterCustom_R", ret=NULL,
            private$handle,
            gpair$grad,
            gpair$hess,
            length(gpair$grad)
          )
      }
      for (i in 1:length(private$is_predicted_cur_iter)) {
        private$is_predicted_cur_iter[[i]] <- FALSE
      }
      return(ret)
    },
    rollback_one_iter = function() {
      lgb.call("LGBM_BoosterRollbackOneIter_R", ret=NULL, private$handle)
      for (i in 1:length(private$is_predicted_cur_iter)) {
        private$is_predicted_cur_iter[[i]] <- FALSE
      }
      return(self)
    },
    current_iter = function() {
      cur_iter <- as.integer(0)
      return(lgb.call("LGBM_BoosterGetCurrentIteration_R",  ret=cur_iter, private$handle))
    },
    eval = function(data, name, feval = NULL) {
      if (!lgb.check.r6.class(data, "lgb.Dataset")) {
        stop("lgb.Booster.eval: only can use lgb.Dataset to eval")
      }
      data_idx <- 0
      if (identical(data, private$train_set)) {
        data_idx <- 1
      } else {
        if(length(private$valid_sets) > 0){
          for (i in 1:length(private$valid_sets)) {
            if (identical(data, private$valid_sets[[i]])) {
              data_idx <- i + 1
              break
            }
          }
        }
      }
      if (data_idx == 0) {
        self$add_valid(data, name)
        data_idx <- private$num_dataset
      }
      return(private$inner_eval(name, data_idx, feval))
    },
    eval_train = function(feval = NULL) {
      return(private$inner_eval(private$name_train_set, 1, feval))
    },
    eval_valid = function(feval = NULL) {
      ret = list()
      if(length(private$valid_sets) <= 0) return(ret)
      for (i in 1:length(private$valid_sets)) {
        ret <-
          append(ret, private$inner_eval(private$name_valid_sets[[i]], i + 1, feval))
      }
      return(ret)
    },
    save_model = function(filename, num_iteration = NULL) {
      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }
      lgb.call(
        "LGBM_BoosterSaveModel_R",
        ret = NULL,
        private$handle,
        as.integer(num_iteration),
        lgb.c_str(filename)
      )
      return(self)
    },
    dump_model = function(num_iteration = NULL) {
      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }
      return(
        lgb.call.return.str(
          "LGBM_BoosterDumpModel_R",
          private$handle,
          as.integer(num_iteration)
        )
      )
    },
    predict = function(data,
                        num_iteration = NULL,
                        rawscore = FALSE,
                        predleaf = FALSE,
                        header = FALSE,
                        reshape = FALSE) {
      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }
      predictor <- Predictor$new(private$handle)
      return(predictor$predict(data, num_iteration, rawscore, predleaf, header, reshape))
    },
    to_predictor = function() {
      Predictor$new(private$handle)
    }
  ),
  private = list(
    handle = NULL,
    train_set = NULL,
    name_train_set = "training",
    valid_sets = list(),
    name_valid_sets = list(),
    predict_buffer = list(),
    is_predicted_cur_iter = list(),
    num_class = 1,
    num_dataset = 0,
    init_predictor = NULL,
    eval_names = NULL,
    higher_better_inner_eval = NULL,
    inner_predict = function(idx) {
      data_name <- private$name_train_set
      if(idx > 1){
        data_name <- private$name_valid_sets[[idx - 1]]
      }
      if (idx > private$num_dataset) {
        stop("data_idx should not be greater than num_dataset")
      }
      if (is.null(private$predict_buffer[[data_name]])) {
        npred <- as.integer(0)
        npred <-
          lgb.call("LGBM_BoosterGetNumPredict_R",
                ret = npred,
                private$handle,
                as.integer(idx - 1))
        private$predict_buffer[[data_name]] <- rep(0.0, npred)
      }
      if (!private$is_predicted_cur_iter[[idx]]) {
        private$predict_buffer[[data_name]] <- 
          lgb.call(
            "LGBM_BoosterGetPredict_R",
            ret=private$predict_buffer[[data_name]],
            private$handle,
            as.integer(idx - 1)
          )
        private$is_predicted_cur_iter[[idx]] <- TRUE
      }
      return(private$predict_buffer[[data_name]])
    },
    get_eval_info = function() {
      if (is.null(private$eval_names)) {
        names <-
          lgb.call.return.str("LGBM_BoosterGetEvalNames_R", private$handle)
        if(nchar(names) > 0){
          names <- strsplit(names, "\t")[[1]]
          private$eval_names <- names
          private$higher_better_inner_eval <-
            rep(FALSE, length(names))
          for (i in 1:length(names)) {
            if (startsWith(names[i], "auc") |
                startsWith(names[i], "ndcg")) {
              private$higher_better_inner_eval[i] <- TRUE
            }
          }
          
        }
      }
      return(private$eval_names)
    },
    inner_eval = function(data_name, data_idx, feval = NULL) {
      if (data_idx > private$num_dataset) {
        stop("data_idx should not be greater than num_dataset")
      }
      private$get_eval_info()
      ret <- list()
      if (length(private$eval_names) > 0) {
        tmp_vals <- rep(0.0, length(private$eval_names))
        tmp_vals <-
          lgb.call("LGBM_BoosterGetEval_R", ret=tmp_vals,
                private$handle,
                as.integer(data_idx - 1))
        for (i in 1:length(private$eval_names)) {
          res <- list()
          res$data_name <- data_name
          res$name <- private$eval_names[i]
          res$value <- tmp_vals[i]
          res$higher_better <- private$higher_better_inner_eval[i]
          ret <- append(ret, list(res))
        }
      }
      if (!is.null(feval)) {
        if (typeof(feval) != 'closure') {
          stop("lgb.Booster.eval: feval should be a function")
        }
        data <- private$train_set
        if (data_idx > 1) {
          data <- private$valid_sets[[data_idx - 1]]
        }
        res <- feval(private$inner_predict(data_idx), data)
        res$data_name <- data_name
        ret <- append(ret, list(res))
      }
      return(ret)
    }
  )
)

# internal helper method
lgb.is.Booster <- function(x){
  if(lgb.check.r6.class(x, "lgb.Booster")){
    return(TRUE)
  } else{
    return(FALSE)
  }
}

#' Predict method for LightGBM model
#' 
#' Predicted values based on class \code{lgb.Booster}
#' 
#' @param object Object of class \code{lgb.Booster}
#' @param data a \code{matrix} object, a \code{dgCMatrix} object or a character representing a filename
#' @param num_iteration number of iteration want to predict with, NULL or <= 0 means use best iteration
#' @param rawscore whether the prediction should be returned in the for of original untransformed 
#'        sum of predictions from boosting iterations' results. E.g., setting \code{rawscore=TRUE} for 
#'        logistic regression would result in predictions for log-odds instead of probabilities.
#' @param predleaf whether predict leaf index instead. 
#' @param header only used for prediction for text file. True if text file has header
#' @param reshape whether to reshape the vector of predictions to a matrix form when there are several 
#'        prediction outputs per case. 

#' @return 
#' For regression or binary classification, it returns a vector of length \code{nrows(data)}.
#' For multiclass classification, either a \code{num_class * nrows(data)} vector or 
#' a \code{(nrows(data), num_class)} dimension matrix is returned, depending on 
#' the \code{reshape} value.
#' 
#' When \code{predleaf = TRUE}, the output is a matrix object with the 
#' number of columns corresponding to the number of trees.
#' @examples
#' library(lightgbm)
#' data(agaricus.train, package='lightgbm')
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label=train$label)
#' data(agaricus.test, package='lightgbm')
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label=test$label)
#' params <- list(objective="regression", metric="l2")
#' valids <- list(test=dtest)
#' model <- lgb.train(params, dtrain, 100, valids, min_data=1, learning_rate=1, early_stopping_rounds=10)
#' preds <- predict(model, test$data)
#' 
#' @rdname predict.lgb.Booster
#' @export
predict.lgb.Booster <- function(object, 
                        data,
                        num_iteration = NULL,
                        rawscore = FALSE,
                        predleaf = FALSE,
                        header = FALSE,
                        reshape = FALSE) {
  if(!lgb.is.Booster(object)){
    stop("predict.lgb.Booster: should input lgb.Booster object")
  }
  object$predict(data, num_iteration, rawscore, predleaf, header, reshape)
}

#' Load LightGBM model
#' 
#' Load LightGBM model from saved model file
#' 
#' @param filename path of model file
#' 
#' @return booster
#' @examples
#' library(lightgbm)
#' data(agaricus.train, package='lightgbm')
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label=train$label)
#' data(agaricus.test, package='lightgbm')
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label=test$label)
#' params <- list(objective="regression", metric="l2")
#' valids <- list(test=dtest)
#' model <- lgb.train(params, dtrain, 100, valids, min_data=1, learning_rate=1, early_stopping_rounds=10)
#' lgb.save(model, "model.txt")
#' load_booster <- lgb.load("model.txt")
#' @rdname lgb.load 
#' @export
lgb.load <- function(filename){
  if(!is.character(filename)){
    stop("lgb.load: filename should be character")
  }
  Booster$new(modelfile=filename)
}

#' Save LightGBM model
#' 
#' Save LightGBM model
#' 
#' @param booster Object of class \code{lgb.Booster}
#' @param filename saved filename
#' @param num_iteration number of iteration want to predict with, NULL or <= 0 means use best iteration
#' 
#' @return booster
#' @examples
#' library(lightgbm)
#' data(agaricus.train, package='lightgbm')
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label=train$label)
#' data(agaricus.test, package='lightgbm')
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label=test$label)
#' params <- list(objective="regression", metric="l2")
#' valids <- list(test=dtest)
#' model <- lgb.train(params, dtrain, 100, valids, min_data=1, learning_rate=1, early_stopping_rounds=10)
#' lgb.save(model, "model.txt")
#' @rdname lgb.save 
#' @export
lgb.save <- function(booster, filename, num_iteration=NULL){
  if(!lgb.is.Booster(booster)){
    stop("lgb.save: should input lgb.Booster object")
  }
  if(!is.character(filename)){
    stop("lgb.save: filename should be character")
  }
  booster$save_model(filename, num_iteration)
}

#' Dump LightGBM model to json
#' 
#' Dump LightGBM model to json
#' 
#' @param booster Object of class \code{lgb.Booster}
#' @param num_iteration number of iteration want to predict with, NULL or <= 0 means use best iteration
#' 
#' @return json format of model
#' @examples
#' library(lightgbm)
#' data(agaricus.train, package='lightgbm')
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label=train$label)
#' data(agaricus.test, package='lightgbm')
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label=test$label)
#' params <- list(objective="regression", metric="l2")
#' valids <- list(test=dtest)
#' model <- lgb.train(params, dtrain, 100, valids, min_data=1, learning_rate=1, early_stopping_rounds=10)
#' json_model <- lgb.dump(model)
#' @rdname lgb.dump 
#' @export
lgb.dump <- function(booster, num_iteration=NULL){
  if(!lgb.is.Booster(booster)){
    stop("lgb.dump: should input lgb.Booster object")
  }
  booster$dump_model(num_iteration)
}

#' Get record evaluation result from booster
#' 
#' Get record evaluation result from booster
#' @param booster Object of class \code{lgb.Booster}
#' @param data_name name of dataset
#' @param eval_name name of evaluation
#' @param iters iterations, NULL will return all
#' @param is_err TRUE will return evaluation error instead
#' @return vector of evaluation result
#' 
#' @rdname lgb.get.eval.result
#' @export
lgb.get.eval.result <- function(booster, data_name, eval_name, iters=NULL, is_err=FALSE){
  if(!lgb.is.Booster(booster)){
    stop("lgb.get.eval.result: only can use booster to get eval result")
  }
  if(!is.character(data_name) | !is.character(eval_name)){
    stop("lgb.get.eval.result: data_name and eval_name should be character")
  }
  if(is.null(booster$record_evals[[data_name]])){
    stop("lgb.get.eval.result: wrong data name")
  }
  if(is.null(booster$record_evals[[data_name]][[eval_name]])){
    stop("lgb.get.eval.result: wrong eval name")
  }
  result <- booster$record_evals[[data_name]][[eval_name]]$eval
  if(is_err){
    result <- booster$record_evals[[data_name]][[eval_name]]$eval_err
  }
  if(is.null(iters)){
    return(as.numeric(result))
  }
  iters <- as.integer(iters)
  delta <- booster$record_evals$start_iter - 1
  iters <- iters - delta
  return(as.numeric(result[iters]))
}

