#' @importFrom data.table is.data.table
#' @importFrom methods as
#' @importClassesFrom Matrix sparseVector sparseMatrix CsparseMatrix dgCMatrix

DataProcessor <- R6::R6Class(
  classname = "lgb.DataProcessor",
  public = list(
    ncols = NULL,
    colnames = NULL,
    factor_levels = NULL,
    formula = NULL,
    formula_terms = NULL,
    formula_predict = NULL,
    label_levels = NULL,
    initialize = function(env_out,
                          data,
                          params,
                          model_formula = NULL,
                          label = NULL,
                          weights = NULL,
                          init_score = NULL) {

      if (!is.null(model_formula)) {

        if (!is.data.frame(data)) {
          stop("'lightgbm()' formula interface is only supported for 'data.frame' inputs.")
        }
        self$formula <- model_formula
        formula_terms <- as.character(model_formula)
        formula_terms[3L] <- paste0(formula_terms[3L], "-1")
        model_formula <- paste0(formula_terms[2L], formula_terms[1L], formula_terms[3L])
        model_formula <- as.formula(model_formula)
        self$formula_terms <- terms(model_formula, data = data)
        self$formula_predict <- delete.response(self$formula_terms)
        model_frame <- model.frame(model_formula, data, na.action = NULL)
        label <- model.response(model_frame, type = "any")
        data <- model.matrix(self$formula_predict, data = model_frame)

      } else {

        self$colnames <- colnames(data)

        if (NROW(self$colnames)) {

          label_nse <- substitute(label)
          label_nse <- eval.parent(substitute(substitute(label_nse)), n = 2L)
          label_nse <- deparse1(label_nse, collapse = "")
          if (label_nse != "NULL" && label_nse %in% self$colnames) {
            self$colnames <- self$colnames[self$colnames != label_nse]
            if (data.table::is.data.table(data)) {
              label <- data[, label_nse, with = FALSE, drop = TRUE]
            } else {
              label <- data[, label_nse, drop = TRUE]
            }
          } else if (is.character(label) && NROW(label) == 1L && label %in% self$colnames) {
            self$colnames <- self$colnames[self$colnames != label]
            if (data.table::is.data.table(data)) {
              label <- data[, label, with=FALSE, drop=TRUE]
            } else {
              label <- data[, label, drop=TRUE]
            }
          }

          weights_nse <- substitute(weights)
          weights_nse <- eval.parent(substitute(substitute(weights_nse)), n = 2L)
          weights_nse <- deparse1(weights_nse, collapse = "")
          if (weights_nse != "NULL" && weights_nse %in% self$colnames) {
            self$colnames <- self$colnames[self$colnames != weights_nse]
            if (data.table::is.data.table(data)) {
              weights <- data[, weights_nse, with = FALSE, drop = TRUE]
            } else {
              weights <- data[, weights_nse, drop = TRUE]
            }
          } else if (is.character(weights) && NROW(weights) == 1L && weights %in% self$colnames) {
            self$colnames <- self$colnames[self$colnames != weights]
            if (data.table::is.data.table(data)) {
              weights <- data[, weights, with = FALSE, drop = TRUE]
            } else {
              weights <- data[, weights, drop = TRUE]
            }
          }

          init_score_nse <- substitute(init_score)
          init_score_nse <- eval.parent(substitute(substitute(init_score_nse)), n = 2L)
          init_score_nse <- deparse1(init_score_nse, collapse = "")
          if (init_score_nse != "NULL" && init_score_nse %in% self$colnames) {
            self$colnames <- self$colnames[self$colnames != init_score_nse]
            if (data.table::is.data.table(data)) {
              init_score <- data[, init_score_nse, with = FALSE, drop = TRUE]
            } else {
              init_score <- data[, init_score_nse, drop = TRUE]
            }
          } else if (is.character(init_score) && NROW(init_score) == 1L && init_score %in% self$colnames) {
            self$colnames <- self$colnames[self$colnames != init_score]
            if (data.table::is.data.table(data)) {
              init_score <- data[, init_score, with = FALSE, drop = TRUE]
            } else {
              init_score <- data[, init_score, drop = TRUE]
            }
          }

          if (length(self$colnames) < ncol(data)) {
            if (data.table::is.data.table(data)) {
              data <- data[, self$colnames, with = FALSE, drop = FALSE]
            } else {
              data <- data[, self$colnames, drop = FALSE]
            }
          }
        } else {
          self$colnames <- NULL
        }

        self$ncols <- ncol(data)

        if (is.data.frame(data)) {

          supported_types <- c("numeric", "integer", "factor", "character", "Date", "POSIXct")
          coltype_is_supported <- sapply(data, function(col) inherits(col, supported_types))
          if (!all(coltype_is_supported)) {
            unsupported_types <- unique(unlist(lapply(
              data
              , function(col) if (inherits(col, supported_types)) NULL else type(col)
            )))
            stop(sprintf("Error: 'lightgbm()' received 'data' with unsupported column types: %s"
                         , paste(head(unsupported_types, 5L)), collapse = ", "))
          }

          data <- data.table::as.data.table(data)

          cols_char <- names(data)[sapply(data, is.character)]
          if (NROW(cols_char)) {
            suppressWarnings(data[, (cols_char) := lapply(.SD, factor), .SDcols = cols_char])
          }

          cols_factors <- names(data)[sapply(data, is.factor)]
          if (NROW(cols_factors)) {
            has_ordered_factor <- any(sapply(data, is.ordered))
            if (has_ordered_factor) {
              warning(paste0("'lighgbm()' was passed data with ordered factors."
                             , "The order in factor levels is ignored."))
            }
            self$factor_levels <- lapply(data[, cols_factors, with = FALSE, drop = FALSE], levels)
            data[
              , (cols_factors) := lapply(.SD, function(x) {
                x <- as.numeric(x) - 1.0
                x[is.na(x)] <- -1.0
                return(x)
              })
              , .SDcols = cols_factors
            ]

            params$categorical_feature <- which(names(data) %in% cols_factors)
          } else {
            params$categorical_feature <- NULL
          }

          data <- as.matrix(data, drop = FALSE)
        }
      }

      if (is.character(label)) {
        label <- factor(label)
      }
      if (!is.factor(label)) {
        label <- as.numeric(label)
        env_out$objective <- "regression"
      } else {
        self$label_levels <- levels(label)
        if (length(levels(label)) <= 1L) {
          stop("Labels to predict is a factor with <2 possible values.")
        } else if (length(levels(label)) == 2L) {
          env_out$objective <- "binary"
        } else {
          env_out$objective <- "multiclass"
        }
        label <- as.numeric(label) - 1.0
      }

      if (!is.numeric(label)) {
        label <- as.numeric(label)
      }
      if (length(label) != nrow(data)) {
        stop("Labels to predict must have length equal to the number of rows in 'X'/'data'.")
      }

      if (!is.null(weights)) {
        weights <- as.numeric(weights)
        if (length(weights) != nrow(data)) {
          stop("'weights' must have length equal to the number of rows in 'X'/'data'.")
        }
      }
      if (!is.null(init_score)) {
        init_score <- as.numeric(init_score)
        if (length(weights) != nrow(data)) {
          stop("'init_score' must have length equal to the number of rows in 'X'/'data'.")
        }
      }

      dataset <- lgb.Dataset(
        data = data
        , label = label
        , weight = weights
        , init_score = init_score
        , params = params
      )
      env_out$dataset <- dataset
    },

    process_new_data = function(data) {
      if (!is.null(self$formula_predict)) {

        data <- model.matrix(self$formula_predict, data = data)

      } else {

        if (is.null(dim(data))) {
          if (inherits(data, "sparseVector")) {
            data <- t(as(data, "CsparseMatrix"))
            if (!inherits(data, "dgCMatrix")) {
              data <- as(data, "dgCMatrix")
            }
          } else {
            data <- matrix(data, nrow = 1L)
          }
        }

        if (ncol(data) < self$ncols) {
          stop(sprintf("New data has fewer columns than expected (%d vs %d)"
                       , ncol(data), self$ncols))
        }

        if (NROW(self$colnames)) {
          if (data.table::is.data.table(data)) {
            data <- data[, self$colnames, with = FALSE, drop = FALSE]
          } else {
            data <- data[, self$colnames, drop = FALSE]
          }
        } else {
          if (ncol(data) > self$ncols) {
            if (data.table::is.data.table(data)) {
              data <- data[, 1L:self$ncols, with = FALSE, drop = FALSE]
            } else {
              data <- data[, 1L:self$ncols, drop = FALSE]
            }
          }
        }

        if (NROW(self$factor_levels)) {
          if (!is.data.frame(data)) {
            stop(paste0("When calling 'lightgbm()' on a 'data.frame' with factor columns,"
                        , "new data to predict on must also be passed as 'data.frame'."))
          }
          data <- as.data.table(data)
          cols_cat <- names(self$factor_levels)
          data[
            , (cols_cat) := mapply(
              factor
              , .SD
              , self$factor_levels
              , SIMPLIFY = FALSE
            )
            , .SDcols = cols_cat
          ][
            , (cols_cat) := lapply(.SD, function(x) {
              x <- as.numeric(x) - 1.0
              x[is.na(x)] <- -1.0
              return(x)
            })
            , .SDcols = cols_cat
          ]
        }
      }

      if (is.data.frame(data)) {
        data <- as.matrix(data, drop = FALSE)
      }

      return(data)
    },

    process_predictions = function(pred, is_contrib = FALSE) {
      if (!is_contrib && NROW(self$label_levels)) {
        if (is.matrix(pred) && ncol(pred) == length(self$label_levels)) {
          colnames(pred) <- self$label_levels
        }
      }
      if (is_contrib) {
        if (NROW(self$colnames) && ncol(pred) == NROW(self$colnames) + 1L) {
          colnames(pred) <- c(self$colnames, "(Intercept)")
        } else if (!is.null(self$formula_terms)) {
          term_labels <- attributes(self$formula_terms)$term.labels
          if (length(term_labels) + 1L == ncol(pred)) {
            colnames(pred) <- c(term_labels, "(Intercept)")
          }
        }
      }
      return(pred)
    }
  )
)
