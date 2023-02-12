DataProcessor <- R6::R6Class(
  classname = "lgb.DataProcessor",
  public = list(
    factor_levels = NULL,

    process_label = function(label, objective, params) {

      if (is.character(label)) {
        label <- factor(label)
      }

      if (is.factor(label)) {

        if (is.ordered(label)) {
          warning("Passed labels as ordered factor. Order of levels is ignored.")
        }

        self$factor_levels <- levels(label)
        if (length(self$factor_levels) <= 1L) {
          stop("Labels to predict is a factor with <2 possible values.")
        }

        label <- as.numeric(label) - 1.0
        out <- list(label = label)
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
          if (!(objective %in% .MULTICLASS_OBJECTIVES())) {
            stop(
              sprintf(
                "Factors with >2 levels as labels only allowed for multi-class objectives. Got: %s (allowed: %s)"
                , objective
                , toString(.MULTICLASS_OBJECTIVES())
              )
            )
          }
          aliases_num_class <- .PARAMETER_ALIASES()$num_class
          present_num_class <- intersect(aliases_num_class, names(params))
          data_num_class <- length(self$factor_levels)
          if (length(present_num_class)) {
            for (num_class_alias in present_num_class) {
              if (params[[num_class_alias]] != data_num_class) {
                warning("'num_class' was passed as parameter, but it is set automatically when passing factors as labels.")
                break
              }
            }
            params <- params[setdiff(names(params), present_num_class)]
          }

          params$num_class <- data_num_class

        }
        out$objective <- objective
        out$params <- params
        return(out)

      } else {

        label <- as.numeric(label)
        if (objective == "auto") {
          objective <- "regression"
        }
        out <- list(
          label = label
          , objective = objective
          , params = params
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
