# [description] List of metrics known to LightGBM. The most up to date list can be found
#               at https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric-parameters
#
# [return] A named logical vector, where each key is a metric name and each value is a boolean.
#          TRUE if higher values of the metric are desirable, FALSE if lower values are desirable.
.METRICS_HIGHER_BETTER <- function() {
    return(c(
        "l1" = FALSE
        , "l2" = FALSE
        , "mae" = FALSE
        , "mape" = FALSE
        , "mean_absolute_percentage_error" = FALSE
        , "mse" = FALSE
        , "mean_squared_error" = FALSE
        , "mean_absolute_error" = FALSE
        , "regression_l1" = FALSE
        , "regression_l2" = FALSE
        , "regression" = FALSE
        , "rmse" = FALSE
        , "root_mean_squared_error" = FALSE
        , "quantile" = FALSE
        , "huber" = FALSE
        , "fair" = FALSE
        , "poisson" = FALSE
        , "gamma" = FALSE
        , "gamma_deviance" = FALSE
        , "tweedie" = FALSE
        , "ndcg" = TRUE
        , "lambdarank" = TRUE
        , "rank_xendcg" = TRUE
        , "xendcg" = TRUE
        , "xe_ndcg" = TRUE
        , "xe_ndcg_mart" = TRUE
        , "xendcg_mart" = TRUE
        , "map" = TRUE
        , "mean_average_precision" = TRUE
        , "auc" = TRUE
        , "binary_logloss" = FALSE
        , "binary_error" = FALSE
        , "auc_mu" = TRUE
        , "multi_logloss" = FALSE
        , "multiclass" = FALSE
        , "softmax" = FALSE
        , "multiclassova" = FALSE
        , "multiclass_ova" = FALSE
        , "ova" = FALSE
        , "ovr" = FALSE
        , "multi_error" = FALSE
        , "cross_entropy" = FALSE
        , "xentropy" = FALSE
        , "cross_entropy_lambda" = FALSE
        , "xentlambda" = FALSE
        , "kullback_leibler" = FALSE
        , "kldiv" = FALSE
    ))
}
