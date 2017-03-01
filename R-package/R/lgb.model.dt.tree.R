#' Parse a LightGBM model json dump
#'
#' Parse a LightGBM model json dump into a \code{data.table} structure.
#'
#' @param model object of class \code{lgb.Booster}
#'
#' @return
#' A \code{data.table} with detailed information about model trees' nodes and leafs.
#'
#' The columns of the \code{data.table} are:
#'
#' \itemize{
#'  \item \code{tree_index}: ID of a tree in a model (integer)
#'  \item \code{split_index}: ID of a node in a tree (integer)
#'  \item \code{split_feature}: for a node, it's a feature name (character);
#'                              for a leaf, it simply labels it as \code{'NA'}
#'  \item \code{node_parent}: ID of the parent node for current node (integer)
#'  \item \code{leaf_index}: ID of a leaf in a tree (integer)
#'  \item \code{leaf_parent}: ID of the parent node for current leaf (integer)
#'  \item \code{split_gain}: Split gain of a node
#'  \item \code{threshold}: Spliting threshold value of a node
#'  \item \code{decision_type}: Decision type of a node
#'  \item \code{internal_value}: Node value
#'  \item \code{internal_count}: The number of observation collected by a node
#'  \item \code{leaf_value}: Leaf value
#'  \item \code{leaf_count}: The number of observation collected by a leaf
#' }
#'
#' @examples
#'
#' data(agaricus.train, package = 'lightgbm')
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#'
#' params = list(objective = "binary",
#'               learning_rate = 0.01, num_leaves = 63, max_depth = -1,
#'               min_data_in_leaf = 1, min_sum_hessian_in_leaf = 1)
#'               model <- lgb.train(params, dtrain, 20)
#' model <- lgb.train(params, dtrain, 20)
#'
#' tree_dt <- lgb.model.dt.tree(model)
#'
#' @importFrom magrittr %>%
#' @importFrom data.table :=
#' @export

lgb.model.dt.tree <- function(model, num_iteration = NULL) {
  json_model <- lgb.dump(model, num_iteration = num_iteration)
  parsed_json_model <- jsonlite::fromJSON(json_model,
                                          simplifyVector = TRUE,
                                          simplifyDataFrame = FALSE,
                                          simplifyMatrix = FALSE,
                                          flatten = FALSE)
  tree_list <- lapply(parsed_json_model$tree_info, single.tree.parse)
  tree_dt <- data.table::rbindlist(tree_list, use.names = TRUE)
  tree_dt[, split_feature := Lookup(split_feature,
                                    seq(0, parsed_json_model$max_feature_idx, by = 1),
                                    parsed_json_model$feature_names)]
  return(tree_dt)
}

single.tree.parse <- function(lgb_tree) {
  single_tree_dt <- data.table::data.table(tree_index = integer(0),
                                           split_index = integer(0), split_feature = integer(0), node_parent = integer(0),
                                           leaf_index = integer(0), leaf_parent = integer(0),
                                           split_gain = numeric(0), threshold = numeric(0), decision_type = character(0),
                                           internal_value = integer(0), internal_count = integer(0),
                                           leaf_value = integer(0), leaf_count = integer(0))
  pre_order_traversal <- function(tree_node_leaf, parent_index = NA) {
    if (!is.null(tree_node_leaf$split_index)) {
      single_tree_dt <<- data.table::rbindlist(l = list(single_tree_dt,
                                                        c(tree_node_leaf[c("split_index", "split_feature",
                                                                           "split_gain", "threshold", "decision_type",
                                                                           "internal_value", "internal_count")],
                                                          "node_parent" = parent_index)),
                                               use.names = TRUE, fill = TRUE)
      pre_order_traversal(tree_node_leaf$left_child, parent_index = tree_node_leaf$split_index)
      pre_order_traversal(tree_node_leaf$right_child, parent_index = tree_node_leaf$split_index)
    } else if (!is.null(tree_node_leaf$leaf_index)) {
      single_tree_dt <<- data.table::rbindlist(l = list(single_tree_dt,
                                                        tree_node_leaf[c("leaf_index", "leaf_parent",
                                                                         "leaf_value", "leaf_count")]),
                                               use.names = TRUE, fill = TRUE)
    }
  }
  pre_order_traversal(lgb_tree$tree_structure)
  single_tree_dt[, tree_index := lgb_tree$tree_index]
  return(single_tree_dt)
}

Lookup <- function(key, key_lookup, value_lookup, missing = NA) {
  match(key, key_lookup) %>%
    magrittr::extract(value_lookup, .) %>%
    magrittr::inset(. , is.na(.), missing)
}
