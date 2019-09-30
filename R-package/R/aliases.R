# Central location for paramter aliases.
# See https://lightgbm.readthedocs.io/en/latest/Parameters.html#core-parameters

# [description] List of respected parameter aliases. Wrapped in a function to take advantage of
#               lazy evaluation (so it doesn't matter what order R sources files during installation).
# [return] A named list, where each key is a main LightGBM parameter and  each value is a character
#          vector of corresponding aliases.
.PARAMETER_ALIASES <- function(){
    return(list(
        "boosting" = c(
            "boosting_type"
            , "boost"
        )
    ))
}
