#' Data preparator for LightGBM datasets with rules (integer)
#'
#' Attempts to prepare a clean dataset to prepare to put in a \code{lgb.Dataset}. Factors and characters are converted to numeric (specifically: integer). In addition, keeps rules created so you can convert other datasets using this converter. This is useful if you have a specific need for integer dataset instead of numeric dataset. Note that there are programs which do not support integer-only input. Consider this as a half memory technique which is dangerous, especially for LightGBM.
#'
#' @param data A data.frame or data.table to prepare.
#' @param rules A set of rules from the data preparator, if already used.
#'
#' @return A list with the cleaned dataset (\code{data}) and the rules (\code{rules}). The data must be converted to a matrix format (\code{as.matrix}) for input in \code{lgb.Dataset}.
#'
#' @examples
#' library(lightgbm)
#' data(iris)
#'
#' str(iris)
#'
#' new_iris <- lgb.prepare_rules2(data = iris) # Autoconverter
#' str(new_iris$data)
#'
#' data(iris) # Erase iris dataset
#' iris$Species[1] <- "NEW FACTOR" # Introduce junk factor (NA)
#'
#' # Use conversion using known rules
#' # Unknown factors become 0, excellent for sparse datasets
#' newer_iris <- lgb.prepare_rules2(data = iris, rules = new_iris$rules)
#'
#' # Unknown factor is now zero, perfect for sparse datasets
#' newer_iris$data[1, ] # Species became 0 as it is an unknown factor
#'
#' newer_iris$data[1, 5] <- 1 # Put back real initial value
#'
#' # Is the newly created dataset equal? YES!
#' all.equal(new_iris$data, newer_iris$data)
#'
#' # Can we test our own rules?
#' data(iris) # Erase iris dataset
#'
#' # We remapped values differently
#' personal_rules <- list(Species = c("setosa" = 3L,
#'                                    "versicolor" = 2L,
#'                                    "virginica" = 1L))
#' newest_iris <- lgb.prepare_rules2(data = iris, rules = personal_rules)
#' str(newest_iris$data) # SUCCESS!
#'
#' @importFrom data.table set
#' @export
lgb.prepare_rules2 <- function(data, rules = NULL) {

  # data.table not behaving like data.frame
  if (inherits(data, "data.table")) {

    # Must use existing rules
    if (!is.null(rules)) {

      # Loop through rules
      for (i in names(rules)) {

        data.table::set(data, j = i, value = unname(rules[[i]][data[[i]]]))
        data[[i]][is.na(data[[i]])] <- 0L # Overwrite NAs by 0s as integer

      }

    } else {

      # Get data classes
      list_classes <- vapply(data, class, character(1))

      # Map characters/factors
      is_fix <- which(list_classes %in% c("character", "factor"))
      rules <- list()

      # Need to create rules?
      if (length(is_fix) > 0) {

        # Go through all characters/factors
        for (i in is_fix) {

          # Store column elsewhere
          mini_data <- data[[i]]

          # Get unique values
          if (is.factor(mini_data)) {
            mini_unique <- levels(mini_data) # Factor
            mini_numeric <- seq_along(mini_unique) # Respect ordinal if needed
          } else {
            mini_unique <- as.factor(unique(mini_data)) # Character
            mini_numeric <- as.integer(mini_unique) # No respect of ordinality
          }

          # Create rules
          indexed <- colnames(data)[i] # Index value
          rules[[indexed]] <- mini_numeric # Numeric content
          names(rules[[indexed]]) <- mini_unique # Character equivalent

          # Apply to real data column
          data.table::set(data, j = i, value = unname(rules[[indexed]][mini_data]))

        }

      }

    }

  } else {

    # Must use existing rules
    if (!is.null(rules)) {

      # Loop through rules
      for (i in names(rules)) {

        data[[i]] <- unname(rules[[i]][data[[i]]])
        data[[i]][is.na(data[[i]])] <- 0L # Overwrite NAs by 0s as integer

      }

    } else {

      # Default routine (data.frame)
      if (inherits(data, "data.frame")) {

        # Get data classes
        list_classes <- vapply(data, class, character(1))

        # Map characters/factors
        is_fix <- which(list_classes %in% c("character", "factor"))
        rules <- list()

        # Need to create rules?
        if (length(is_fix) > 0) {

          # Go through all characters/factors
          for (i in is_fix) {

            # Store column elsewhere
            mini_data <- data[[i]]

            # Get unique values
            if (is.factor(mini_data)) {
              mini_unique <- levels(mini_data) # Factor
              mini_numeric <- seq_along(mini_unique) # Respect ordinal if needed
            } else {
              mini_unique <- as.factor(unique(mini_data)) # Character
              mini_numeric <- as.integer(mini_unique) # No respect of ordinality
            }

            # Create rules
            indexed <- colnames(data)[i] # Index value
            rules[[indexed]] <- mini_numeric # Numeric content
            names(rules[[indexed]]) <- mini_unique # Character equivalent

            # Apply to real data column
            data[[i]] <- unname(rules[[indexed]][mini_data])

          }

        }

      } else {

        # What do you think you are doing here? Throw error.
        stop("lgb.prepare: you provided ", paste(class(data), collapse = " & "), " but data should have class data.frame")

      }

    }

  }

  return(list(data = data, rules = rules))

}
