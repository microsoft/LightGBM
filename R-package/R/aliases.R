# Central location for parameter aliases.
# See https://lightgbm.readthedocs.io/en/latest/Parameters.html#core-parameters

# [description] List of respected parameter aliases. Wrapped in a function to take advantage of
#               lazy evaluation (so it doesn't matter what order R sources files during installation).
# [return] A named list, where each key is a main LightGBM parameter and each value is a character
#          vector of corresponding aliases.
.PARAMETER_ALIASES <- function() {
    return(list(
        "boosting" = c(
            "boosting"
            , "boost"
            , "boosting_type"
        )
        , "early_stopping_round" = c(
            "early_stopping_round"
            , "early_stopping_rounds"
            , "early_stopping"
            , "n_iter_no_change"
        )
        , "num_iterations" = c(
            "num_iterations"
            , "num_iteration"
            , "n_iter"
            , "num_tree"
            , "num_trees"
            , "num_round"
            , "num_rounds"
            , "num_boost_round"
            , "n_estimators"
        )
    ))
}
