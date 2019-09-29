#' Plot feature contribution as a bar graph
#'
#' Plot previously calculated feature contribution as a bar graph.
#'
#' @param tree_interpretation_dt a \code{data.table} returned by \code{\link{lgb.interprete}}.
#' @param top_n maximal number of top features to include into the plot.
#' @param cols the column numbers of layout, will be used only for multiclass classification feature contribution.
#' @param left_margin (base R barplot) allows to adjust the left margin size to fit feature names.
#' @param cex (base R barplot) passed as \code{cex.names} parameter to \code{barplot}.
#'
#' @details
#' The graph represents each feature as a horizontal bar of length proportional to the defined contribution of a feature.
#' Features are shown ranked in a decreasing contribution order.
#'
#' @return
#' The \code{lgb.plot.interpretation} function creates a \code{barplot}.
#'
#' @examples
#' library(lightgbm)
#' Sigmoid <- function(x) {1 / (1 + exp(-x))}
#' Logit <- function(x) {log(x / (1 - x))}
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' setinfo(dtrain, "init_score", rep(Logit(mean(train$label)), length(train$label)))
#' data(agaricus.test, package = "lightgbm")
#' test <- agaricus.test
#'
#' params <- list(objective = "binary",
#'                learning_rate = 0.01, num_leaves = 63, max_depth = -1,
#'                min_data_in_leaf = 1, min_sum_hessian_in_leaf = 1)
#' model <- lgb.train(params, dtrain, 10)
#'
#' tree_interpretation <- lgb.interprete(model, test$data, 1:5)
#' lgb.plot.interpretation(tree_interpretation[[1]], top_n = 10)
#' @importFrom data.table setnames
#' @importFrom graphics barplot par
#' @export
lgb.plot.interpretation <- function(tree_interpretation_dt,
                                    top_n = 10,
                                    cols = 1,
                                    left_margin = 10,
                                    cex = NULL) {

  # Get number of columns
  num_class <- ncol(tree_interpretation_dt) - 1

  # Refresh plot
  op <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(op))

  # Do some magic plotting
  bottom_margin <- 3.0
  top_margin <- 2.0
  right_margin <- op$mar[4]

  graphics::par(
    mar = c(
      bottom_margin
      , left_margin
      , top_margin
      , right_margin
    )
  )

  # Check for number of classes
  if (num_class == 1) {

    # Only one class, plot straight away
    multiple.tree.plot.interpretation(tree_interpretation_dt,
                                      top_n = top_n,
                                      title = NULL,
                                      cex = cex)

  } else {

    # More than one class, shape data first
    layout_mat <- matrix(seq.int(to = cols * ceiling(num_class / cols)),
                         ncol = cols, nrow = ceiling(num_class / cols))

    # Shape output
    graphics::par(mfcol = c(nrow(layout_mat), ncol(layout_mat)))

    # Loop throughout all classes
    for (i in seq_len(num_class)) {

      # Prepare interpretation, perform T, get the names, and plot straight away
      plot_dt <- tree_interpretation_dt[, c(1, i + 1), with = FALSE]
      data.table::setnames(
        plot_dt
        , old = names(plot_dt)
        , new = c("Feature", "Contribution")
      )
      multiple.tree.plot.interpretation(
        plot_dt
        , top_n = top_n
        , title = paste("Class", i - 1)
        , cex = cex
      )

    }
  }
}

#' @importFrom graphics barplot
multiple.tree.plot.interpretation <- function(tree_interpretation,
                                              top_n,
                                              title,
                                              cex) {

  # Parse tree
  tree_interpretation <- tree_interpretation[order(abs(Contribution), decreasing = TRUE),][seq_len(min(top_n, .N)),]

  # Attempt to setup a correct cex
  if (is.null(cex)) {
    cex <- 2.5 / log2(1 + top_n)
  }

  # Do plot
  tree_interpretation[.N:1,
                      graphics::barplot(
                          height = Contribution,
                          names.arg = Feature,
                          horiz = TRUE,
                          col = ifelse(Contribution > 0, "firebrick", "steelblue"),
                          border = NA,
                          main = title,
                          cex.names = cex,
                          las = 1
                      )]

  # Return invisibly
  return(invisible(NULL))

}
