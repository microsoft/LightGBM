#' Plot feature importance as a bar graph
#'
#' Plot previously calculated feature importance: Gain, Cover and Frequency, as a bar graph.
#'
#' @param tree_imp a \code{data.table} returned by \code{\link{lgb.importance}}.
#' @param top_n maximal number of top features to include into the plot.
#' @param measure the name of importance measure to plot, can be "Gain", "Cover" or "Frequency".
#' @param left_margin (base R barplot) allows to adjust the left margin size to fit feature names.
#' @param cex (base R barplot) passed as \code{cex.names} parameter to \code{barplot}.
#'
#' @details
#' The graph represents each feature as a horizontal bar of length proportional to the defined importance of a feature.
#' Features are shown ranked in a decreasing importance order.
#'
#' @return
#' The \code{lgb.plot.importance} function creates a \code{barplot}
#' and silently returns a processed data.table with \code{top_n} features sorted by defined importance.
#'
#' @examples
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#'
#' params <- list(
#'     objective = "binary"
#'     , learning_rate = 0.01
#'     , num_leaves = 63
#'     , max_depth = -1
#'     , min_data_in_leaf = 1
#'     , min_sum_hessian_in_leaf = 1
#' )
#'
#' model <- lgb.train(params, dtrain, 10)
#'
#' tree_imp <- lgb.importance(model, percentage = TRUE)
#' lgb.plot.importance(tree_imp, top_n = 10, measure = "Gain")
#' @importFrom graphics barplot par
#' @export
lgb.plot.importance <- function(tree_imp,
                                top_n = 10,
                                measure = "Gain",
                                left_margin = 10,
                                cex = NULL) {

  # Check for measurement (column names) correctness
  measure <- match.arg(measure, choices = c("Gain", "Cover", "Frequency"), several.ok = FALSE)

  # Get top N importance (defaults to 10)
  top_n <- min(top_n, nrow(tree_imp))

  # Parse importance
  tree_imp <- tree_imp[order(abs(get(measure)), decreasing = TRUE),][seq_len(top_n),]

  # Attempt to setup a correct cex
  if (is.null(cex)) {
    cex <- 2.5 / log2(1 + top_n)
  }

  # Refresh plot
  op <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(op))

  graphics::par(
    mar = c(
      op$mar[1]
      , left_margin
      , op$mar[3]
      , op$mar[4]
    )
  )

  # Do plot
  tree_imp[.N:1,
           graphics::barplot(
               height = get(measure),
               names.arg = Feature,
               horiz = TRUE,
               border = NA,
               main = "Feature Importance",
               xlab = measure,
               cex.names = cex,
               las = 1
           )]

  # Return invisibly
  invisible(tree_imp)

}
