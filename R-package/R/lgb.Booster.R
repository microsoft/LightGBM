Booster <- R6Class(
  "lgb.Booster",
  cloneable=FALSE,
  public = list(
    best_iter = -1,
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
        private$predict_buffer <- c(private$predict_buffer, NULL)
        private$is_predicted_cur_iter <-
          c(private$is_predicted_cur_iter, FALSE)
      } else if (!is.null(modelfile)) {
        if (!is.character(modelfile)) {
          stop("lgb.Booster: Only can use string as model file path")
          handle <-
            lgb.call("LGBM_BoosterCreateFromModelfile_R",
              ret=handle,
              lgb.c_str(modelfile))
        }
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
      lgb.call("LGBM_BoosterAddValidData_R", ret=NULL, private$handle, data$.__enclos_env__$private$get_handle())
      private$valid_sets <- c(private$valid_sets, data)
      private$name_valid_sets <- c(private$name_valid_sets, name)
      private$num_dataset <- private$num_dataset + 1
      private$predict_buffer <- c(private$predict_buffer, NULL)
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
        private$predict_buffer[1] <- NULL
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
        private$is_predicted_cur_iter[i] <- FALSE
      }
      return(ret)
    },
    rollback_one_iter = function() {
      lgb.call("LGBM_BoosterRollbackOneIter_R", ret=NULL, private$handle)
      for (i in 1:length(private$is_predicted_cur_iter)) {
        private$is_predicted_cur_iter[i] <- FALSE
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
        for (i in 1:length(private$valid_sets)) {
          if (identical(data, private$valid_sets[i])) {
            data_idx <- i + 1
            break
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
      for (i in 1:length(private$valid_sets)) {
        ret <-
          c(ret,
            private$inner_eval(private$name_valid_sets[i], i + 1, feval))
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
        num_iteration,
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
          num_iteration
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
      if (idx > private$num_dataset) {
        stop("data_idx should not be greater than num_dataset")
      }
      if (is.null(private$predict_buffer[idx])) {
        npred <- as.integer(0)
        npred <-
          lgb.call("LGBM_BoosterGetNumPredict_R",
                ret = npred,
                private$handle,
                idx - 1)
        private$predict_buffer[idx] <- rep(0.0, npred)
      }
      if (!private$is_predicted_cur_iter[idx]) {
        private$predict_buffer[idx] <- 
          lgb.call(
            "LGBM_BoosterGetPredict_R",
            ret=private$predict_buffer[idx],
            private$handle,
            idx - 1
          )
        private$is_predicted_cur_iter[idx] <- TRUE
      }
      return(private$predict_buffer[idx])
    },
    get_eval_info = function() {
      if (is.null(private$eval_names)) {
        names <-
          lgb.call.return.str("LGBM_BoosterGetEvalNames_R", private$handle)
        private$eval_names <- as.list(strsplit(names, "\t"))
        if (!is.null(private$eval_names)) {
          private$higher_better_inner_eval <-
            rep(FALSE, length(private$eval_names))
          for (i in 1:length(private$eval_names)) {
            if (startsWith(private$eval_names[i], "auc")
                | startsWith(private$eval_names[i], "ndcg")) {
              private$higher_better_inner_eval[i] <- TRUE
            }
          }
        }
      }
      if (is.null(private$eval_names)) {
        private$eval_names <- list()
      }
      return(private$eval_names)
    },
    inner_eval = function(data_name, data_idx, feval = NULL) {
      if (idx > private$num_dataset) {
        stop("data_idx should not be greater than num_dataset")
      }
      private$get_eval_info()
      ret <- list()
      if (length(private$eval_names) > 0) {
        tmp_res <- rep(0.0, length(private$eval_names))
        tmp_res <-
          lgb.call("LGBM_BoosterGetEval_R", ret=tmp_res,
                private$handle,
                data_idx - 1)
        for (i in 1:length(tmp_res)) {
          ret <-
            c(
              ret,
              c(
                data_name,
                private$eval_names[i],
                tmp_res[i],
                private$higher_better_inner_eval[i]
              )
            )
        }
      }
      if (!is.null(feval)) {
        if (typeof(feval) != 'closure') {
          stop("lgb.Booster.eval: feval should be a function")
        }
        data <- private$train_set
        if (data_idx > 1) {
          data <- private$valid_sets[data_idx - 1]
        }
        res <- feval(private$inner_predict(data_idx), data)
        for (i in 1:length(res)) {
          ret <-
            c(ret,
              c(data_name, res[i]$name, res[i]$value, res[i]$higher_better))
        }
      }
      return(ret)
    }
  )
)

#' Predict method for LightGBM model
#' 
#' Predicted values based on either lightgbm model or model handle object.
#' 
#' @param booster Object of class \code{lgb.Booster}
#' @param data a \code{matrix} object, a \code{dgCMatrix} object or a character representing a filename
#' @param num_iteration number of iteration want to predict with, <= 0 means use best iteration
#' @param rawscore whether the prediction should be returned in the for of original untransformed 
#'        sum of predictions from boosting iterations' results. E.g., setting \code{rawscore=TRUE} for 
#'        logistic regression would result in predictions for log-odds instead of probabilities.
#' @param predleaf whether predict leaf index instead. 
#' @param header only used for prediction for text file. True if text file has header
#' @param reshape whether to reshape the vector of predictions to a matrix form when there are several 
#'        prediction outputs per case.
#' 
#' @details  
#' Note that \code{ntreelimit} is not necessarily equal to the number of boosting iterations
#' and it is not necessarily equal to the number of trees in a model.
#' E.g., in a random forest-like model, \code{ntreelimit} would limit the number of trees.
#' But for multiclass classification, there are multiple trees per iteration, 
#' but \code{ntreelimit} limits the number of boosting iterations.
#' 
#' Also note that \code{ntreelimit} would currently do nothing for predictions from gblinear, 
#' since gblinear doesn't keep its boosting history. 
#' 
#' One possible practical applications of the \code{predleaf} option is to use the model 
#' as a generator of new features which capture non-linearity and interactions, 
#' e.g., as implemented in \code{\link{xgb.create.features}}. 
#' 
#' @return 
#' For regression or binary classification, it returns a vector of length \code{nrows(data)}.
#' For multiclass classification, either a \code{num_class * nrows(data)} vector or 
#' a \code{(nrows(data), num_class)} dimension matrix is returned, depending on 
#' the \code{reshape} value.
#' 
#' When \code{predleaf = TRUE}, the output is a matrix object with the 
#' number of columns corresponding to the number of trees.
#' 
#'
#' @rdname predict.lgb.Booster
#' @export
predict.lgb.Booster <- function(booster, 
                        data,
                        num_iteration = NULL,
                        rawscore = FALSE,
                        predleaf = FALSE,
                        header = FALSE,
                        reshape = FALSE) {
  booster$predict(data, num_iteration, rawscore, predleaf, header, reshape)
}

#' Save LightGBM model
#' 
#' Save LightGBM model
#' 
#' @param booster Object of class \code{lgb.Booster}
#' @param filename saved filename
#' @param num_iteration number of iteration want to predict with, <= 0 means use best iteration
#' 
#' @return booster
#' 
#'
#' @rdname lgb.save 
#' @export
lgb.save <- function(booster, filename, num_iteration=NULL){
  booster$save_model(booster, filename, num_iteration)
}

#' Dump LightGBM model to json
#' 
#' Dump LightGBM model to json
#' 
#' @param booster Object of class \code{lgb.Booster}
#' @param num_iteration number of iteration want to predict with, <= 0 means use best iteration
#' 
#' @return json format of model
#' 
#'
#' @rdname lgb.dump 
#' @export
lgb.dump <- function(booster, num_iteration=NULL){
  booster$dump_model(booster, num_iteration)
}