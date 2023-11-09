.is_Booster <- function(x) {
  return(all(c("R6", "lgb.Booster") %in% class(x)))  # nolint: class_equals
}

.is_Dataset <- function(x) {
  return(all(c("R6", "lgb.Dataset") %in% class(x)))  # nolint: class_equals
}

.is_Predictor <- function(x) {
  return(all(c("R6", "lgb.Predictor") %in% class(x)))  # nolint: class_equals
}

.is_null_handle <- function(x) {
  if (is.null(x)) {
    return(TRUE)
  }
  return(
    isTRUE(.Call(LGBM_HandleIsNull_R, x))
  )
}

.params2str <- function(params) {

  if (!identical(class(params), "list")) {
    stop("params must be a list")
  }

  names(params) <- gsub(".", "_", names(params), fixed = TRUE)
  param_names <- names(params)
  ret <- list()

  # Perform key value join
  for (i in seq_along(params)) {

    # If a parameter has multiple values, join those values together with commas.
    # trimws() is necessary because format() will pad to make strings the same width
    val <- paste0(
      trimws(
        format(
          x = unname(params[[i]])
          , scientific = FALSE
        )
      )
      , collapse = ","
    )
    if (nchar(val) <= 0L) next # Skip join

    # Join key value
    pair <- paste0(c(param_names[[i]], val), collapse = "=")
    ret <- c(ret, pair)

  }

  if (length(ret) == 0L) {
    return("")
  }

  return(paste0(ret, collapse = " "))

}

.check_interaction_constraints <- function(interaction_constraints, column_names) {

  # Convert interaction constraints to feature numbers
  string_constraints <- list()

  if (!is.null(interaction_constraints)) {

    if (!methods::is(interaction_constraints, "list")) {
        stop("interaction_constraints must be a list")
    }
    constraint_is_character_or_numeric <- sapply(
        X = interaction_constraints
        , FUN = function(x) {
            return(is.character(x) || is.numeric(x))
        }
    )
    if (!all(constraint_is_character_or_numeric)) {
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


# [description]
#     Take any character values from eval and store them in params$metric.
#     This has to account for the fact that `eval` could be a character vector,
#     a function, a list of functions, or a list with a mix of strings and
#     functions
.check_eval <- function(params, eval) {

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
#         2. the alias with the highest priority found in `params`
#         3. the keyword argument passed in
#
#     For example, "num_iterations" can also be provided to lgb.train()
#     via keyword "nrounds". lgb.train() will choose one value for this parameter
#     based on the first match in this list:
#
#         1. params[["num_iterations]]
#         2. the highest priority alias of "num_iterations" found in params
#         3. the nrounds keyword argument
#
#     If multiple aliases are found in `params` for the same parameter, they are
#     all removed before returning `params`.
#
# [return]
#     params with num_iterations set to the chosen value, and other aliases
#     of num_iterations removed
.check_wrapper_param <- function(main_param_name, params, alternative_kwarg_value) {

  aliases <- .PARAMETER_ALIASES()[[main_param_name]]
  aliases_provided <- aliases[aliases %in% names(params)]
  aliases_provided <- aliases_provided[aliases_provided != main_param_name]

  # prefer the main parameter
  if (!is.null(params[[main_param_name]])) {
    for (param in aliases_provided) {
      params[[param]] <- NULL
    }
    return(params)
  }

  # if the main parameter wasn't provided, prefer the first alias
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

#' @importFrom parallel detectCores
.get_default_num_threads <- function() {
  if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {  # nolint: undesirable_function
    return(RhpcBLASctl::get_num_cores())
  } else {
    msg <- "Optional package 'RhpcBLASctl' not found."
    cores <- 0L
    if (Sys.info()["sysname"] != "Linux") {
      cores <- parallel::detectCores(logical = FALSE)
      if (is.na(cores) || cores < 0L) {
        cores <- 0L
      }
    }
    if (cores == 0L) {
      msg <- paste(msg, "Will use default number of OpenMP threads.", sep = " ")
    } else {
      msg <- paste(msg, "Detection of CPU cores might not be accurate.", sep = " ")
    }
    warning(msg)
    return(cores)
  }
}

.equal_or_both_null <- function(a, b) {
  if (is.null(a)) {
    if (!is.null(b)) {
      return(FALSE)
    }
    return(TRUE)
  } else {
    if (is.null(b)) {
      return(FALSE)
    }
    return(a == b)
  }
}
