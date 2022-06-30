DataProcessor <- R6::R6Class(
  classname = "lgb.DataProcessor",
  public = list(
    factor_levels = NULL,

    process_y = function(y, objective, params) {

      if (is.character(y)) {
        y <- factor(y)
      }

      if (is.factor(y)) {

        if (is.ordered(y)) {
          warning("Passed labels as ordered factor. Order of levels is ignored.")
        }

        self$factor_levels <- levels(y)
        if (length(self$factor_levels) <= 1L) {
          stop("Labels to predict is a factor with <2 possible values.")
        }

        y <- as.numeric(y) - 1.0
        out <- list(y = y)
        if (length(self$factor_levels) == 2L) {
          if (objective == "auto") {
            objective <- "binary"
          }
          if (objective != "binary") {
            stop("Two-level factors as labels only allowed for objective='binary' or objective='auto'.")
          }
        } else {
          if (objective == "auto") {
            objective <- "multiclass"
          }
          if (!(objective %in% .MULTICLASS_OBJECTIVES)) {
            stop(
              sprintf(
                "Factors with >2 levels as labels only allowed for multi-class objectives. Got: %s (allowed: %s)"
                , objective
                , toString(.MULTICLASS_OBJECTIVES)
              )
            )
          }
          if ("num_class" %in% names(params)) {
            warning("'num_class' was passed as parameter, but it is set automatically when passing factors as labels.")
          }
          params$num_class <- length(self$factor_levels)

        }
        out$objective <- objective
        out$params <- params
        return(out)

      } else {

        y <- as.numeric(y)
        if (objective == "auto") {
          objective <- "regression"
        }
        out <- list(
          y = y,
          objective = objective,
          params = params
        )
        return(out)

      }
    },

    process_predictions = function(pred, type) {
      if (NROW(self$factor_levels)) {
        if (type == "class") {
          pred <- as.integer(pred) + 1L
          attributes(pred)$levels <- self$factor_levels
          attributes(pred)$class <- "factor"
        } else if (type %in% c("response", "raw")) {
          if (is.matrix(pred) && ncol(pred) == length(self$factor_levels)) {
            colnames(pred) <- self$factor_levels
          }
        }
      }

      return(pred)
    }
  )
)
