lgb.is.Booster <- function(x) {
  return(all(c("R6", "lgb.Booster") %in% class(x)))
}

lgb.is.Dataset <- function(x) {
  return(all(c("R6", "lgb.Dataset") %in% class(x)))
}

lgb.is.Predictor <- function(x) {
  return(all(c("R6", "lgb.Predictor") %in% class(x)))
}

lgb.is.null.handle <- function(x) {
  if (is.null(x)) {
    return(TRUE)
  }
  return(
    isTRUE(.Call(LGBM_HandleIsNull_R, x))
  )
}

lgb.params2str <- function(params) {

  # Check for a list as input
  if (!identical(class(params), "list")) {
    stop("params must be a list")
  }

  # Split parameter names
  names(params) <- gsub("\\.", "_", names(params))

  # Setup temporary variable
  ret <- list()

  # Perform key value join
  for (key in names(params)) {

    # If a parameter has multiple values, join those values together with commas.
    # trimws() is necessary because format() will pad to make strings the same width
    val <- paste0(
      trimws(
        format(
          x = params[[key]]
          , scientific = FALSE
        )
      )
      , collapse = ","
    )
    if (nchar(val) <= 0L) next # Skip join

    # Join key value
    pair <- paste0(c(key, val), collapse = "=")
    ret <- c(ret, pair)

  }

  # Check ret length
  if (length(ret) == 0L) {
    return("")
  }

  return(paste0(ret, collapse = " "))

}

lgb.check_interaction_constraints <- function(interaction_constraints, column_names) {

  # Convert interaction constraints to feature numbers
  string_constraints <- list()

  if (!is.null(interaction_constraints)) {

    if (!methods::is(interaction_constraints, "list")) {
        stop("interaction_constraints must be a list")
    }
    if (!all(sapply(interaction_constraints, function(x) {is.character(x) || is.numeric(x)}))) {
        stop("every element in interaction_constraints must be a character vector or numeric vector")
    }

    for (constraint in interaction_constraints) {

      # Check for character name
      if (is.character(constraint)) {

          constraint_indices <- as.integer(match(constraint, column_names) - 1L)

          # Provided indices, but some indices are not existing?
          if (sum(is.na(constraint_indices)) > 0L) {
            stop(
              "supplied an unknown feature in interaction_constraints "
              , sQuote(constraint[is.na(constraint_indices)])
            )
          }

        } else {

          # Check that constraint indices are at most number of features
          if (max(constraint) > length(column_names)) {
            stop(
              "supplied a too large value in interaction_constraints: "
              , max(constraint)
              , " but only "
              , length(column_names)
              , " features"
            )
          }

          # Store indices as [0, n-1] indexed instead of [1, n] indexed
          constraint_indices <- as.integer(constraint - 1L)

        }

        # Convert constraint to string
        constraint_string <- paste0("[", paste0(constraint_indices, collapse = ","), "]")
        string_constraints <- append(string_constraints, constraint_string)
    }

  }

  return(string_constraints)

}

lgb.check.obj <- function(params, obj) {

  # List known objectives in a vector
  OBJECTIVES <- c(
    "regression"
    , "regression_l1"
    , "regression_l2"
    , "mean_squared_error"
    , "mse"
    , "l2_root"
    , "root_mean_squared_error"
    , "rmse"
    , "mean_absolute_error"
    , "mae"
    , "quantile"
    , "huber"
    , "fair"
    , "poisson"
    , "binary"
    , "lambdarank"
    , "multiclass"
    , "softmax"
    , "multiclassova"
    , "multiclass_ova"
    , "ova"
    , "ovr"
    , "xentropy"
    , "cross_entropy"
    , "xentlambda"
    , "cross_entropy_lambda"
    , "mean_absolute_percentage_error"
    , "mape"
    , "gamma"
    , "tweedie"
    , "rank_xendcg"
    , "xendcg"
    , "xe_ndcg"
    , "xe_ndcg_mart"
    , "xendcg_mart"
  )

  # Check whether the objective is empty or not, and take it from params if needed
  if (!is.null(obj)) {
    params$objective <- obj
  }

  # Check whether the objective is a character
  if (is.character(params$objective)) {

    # If the objective is a character, check if it is a known objective
    if (!(params$objective %in% OBJECTIVES)) {

      stop("lgb.check.obj: objective name error should be one of (", paste0(OBJECTIVES, collapse = ", "), ")")

    }

  } else if (!is.function(params$objective)) {

    stop("lgb.check.obj: objective should be a character or a function")

  }

  return(params)

}

# [description]
#     Take any character values from eval and store them in params$metric.
#     This has to account for the fact that `eval` could be a character vector,
#     a function, a list of functions, or a list with a mix of strings and
#     functions
lgb.check.eval <- function(params, eval) {

  if (is.null(params$metric)) {
    params$metric <- list()
  } else if (is.character(params$metric)) {
    params$metric <- as.list(params$metric)
  }

  # if 'eval' is a character vector or list, find the character
  # elements and add them to 'metric'
  if (!is.function(eval)) {
    for (i in seq_along(eval)) {
      element <- eval[[i]]
      if (is.character(element)) {
        params$metric <- append(params$metric, element)
      }
    }
  }

  # If more than one character metric was given, then "None" should
  # not be included
  if (length(params$metric) > 1L) {
    params$metric <- Filter(
        f = function(metric) {
          !(metric %in% .NO_METRIC_STRINGS())
        }
        , x = params$metric
    )
  }

  # duplicate metrics should be filtered out
  params$metric <- as.list(unique(unlist(params$metric)))

  return(params)
}


# [description]
#
#     Resolve differences between passed-in keyword arguments, parameters,
#     and parameter aliases. This function exists because some functions in the
#     package take in parameters through their own keyword arguments other than
#     the `params` list.
#
#     If the same underlying parameter is provided multiple
#     ways, the first item in this list is used:
#
#         1. the main (non-alias) parameter found in `params`
#         2. the first alias of that parameter found in `params`
#         3. the keyword argument passed in
#
#     For example, "num_iterations" can also be provided to lgb.train()
#     via keyword "nrounds". lgb.train() will choose one value for this parameter
#     based on the first match in this list:
#
#         1. params[["num_iterations]]
#         2. the first alias of "num_iterations" found in params
#         3. the nrounds keyword argument
#
#     If multiple aliases are found in `params` for the same parameter, they are
#     all removed before returning `params`.
#
# [return]
#     params with num_iterations set to the chosen value, and other aliases
#     of num_iterations removed
lgb.check.wrapper_param <- function(main_param_name, params, alternative_kwarg_value) {

  aliases <- .PARAMETER_ALIASES()[[main_param_name]]
  aliases_provided <- names(params)[names(params) %in% aliases]
  aliases_provided <- aliases_provided[aliases_provided != main_param_name]

  # prefer the main parameter
  if (!is.null(params[[main_param_name]])) {
    for (param in aliases_provided) {
      params[[param]] <- NULL
    }
    return(params)
  }

  # if the main parameter wasn't proovided, prefer the first alias
  if (length(aliases_provided) > 0L) {
    first_param <- aliases_provided[1L]
    params[[main_param_name]] <- params[[first_param]]
    for (param in aliases_provided) {
      params[[param]] <- NULL
    }
    return(params)
  }

  # if not provided in params at all, use the alternative value provided
  # through a keyword argument from lgb.train(), lgb.cv(), etc.
  params[[main_param_name]] <- alternative_kwarg_value
  return(params)
}
