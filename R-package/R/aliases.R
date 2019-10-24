# Central location for paramter aliases.
# See https://lightgbm.readthedocs.io/en/latest/Parameters.html#core-parameters

# [description] List of respected parameter aliases. Wrapped in a function to take advantage of
#               lazy evaluation (so it doesn't matter what order R sources files during installation).
# [return] A named list, where each key is a main LightGBM parameter and  each value is a character
#          vector of corresponding aliases.
.PARAMETER_ALIASES <- function(){
    return(list(
        "boosting" = c(
            "boosting"
            , "boost"
            , "boosting_type"
        )
        , "metric" = c(
            "metric"
            , "metrics"
            , "metric_types"
        )
        , "num_class" = c(
            "num_class"
            , "num_classes"
        )
    ))
}
