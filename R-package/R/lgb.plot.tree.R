#' @name lgb.plot.tree
#' @title Plot a single LightGBM tree.
#' @description The \code{lgb.plot.tree} function creates a DiagrammeR plot of a single LightGBM tree.
#' @param model a \code{lgb.Booster} object.
#' @param tree an integer specifying the tree to plot. This is 1-based, so e.g. a value of '7' means 'the 7th tree' (tree_index=6 in LightGBM's underlying representation).
#' @param rules a list of rules to replace the split values with feature levels.
#'
#' @return
#' The \code{lgb.plot.tree} function creates a DiagrammeR plot.
#'
#' @details
#' The \code{lgb.plot.tree} function creates a DiagrammeR plot of a single LightGBM tree. The tree is extracted from the model and displayed as a directed graph. The nodes are labelled with the feature, split value, gain, cover and value. The edges are labelled with the decision type and split value.
#'
#' @examples
#' \donttest{
#' # EXAMPLE: use the LightGBM example dataset to build a model with a single tree
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' data(agaricus.test, package = "lightgbm")
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
#' # define model parameters and build a single tree
#' params <- list(
#'     objective = "regression",
#'     min_data = 1L,
#' )
#' valids <- list(test = dtest)
#' model <- lgb.train(
#'     params = params,
#'     data = dtrain,
#'     nrounds = 1L,
#'     valids = valids,
#'     early_stopping_rounds = 1L
#' )
#' # plot the tree and compare to the tree table
#' # trees start from 0 in lgb.model.dt.tree
#' tree_table <- lgb.model.dt.tree(model)
#' lgb.plot.tree(model, 0)
#' }
#'
#' @export
lgb.plot.tree <- function(model = NULL, tree = NULL, rules = NULL) {
    # check model is lgb.Booster
    if (!.is_Booster(x = model)) {
        stop("lgb.plot.tree: model should be an ", sQuote("lgb.Booster"))
    }
    # check DiagrammeR is available
    if (!requireNamespace("DiagrammeR", quietly = TRUE)) {
        stop("lgb.plot.tree: DiagrammeR package is required",
            call. = FALSE
        )
    }
    # tree must be numeric
    if (!inherits(tree, "numeric")) {
        stop("lgb.plot.tree: Has to be an integer numeric")
    }
    # tree must be integer
    if (tree %% 1 != 0) {
        stop("lgb.plot.tree: Has to be an integer numeric")
    }
    # extract data.table model structure
    modelDT <- lgb.model.dt.tree(model)
    # check that tree is less than or equal to the maximum tree index in the model
    if (tree > max(modelDT$tree_index) || tree < 1) {
        warning("lgb.plot.tree: Value of 'tree' should be between 1 and the total number of trees in the model (", max(modelDT$tree_index), "). Got: ", tree, ".")
        stop("lgb.plot.tree: Invalid tree number")
    }
    # filter modelDT to just the rows for the selected tree
    modelDT <- modelDT[tree_index == tree, ]
    # change the column names to shorter more diagram friendly versions
    data.table::setnames(modelDT
    , old = c("tree_index", "split_feature", "threshold", "split_gain")
    , new = c("Tree", "Feature", "Split", "Gain"))
    # assign leaf_value to the Value column in modelDT
    modelDT[, Value := leaf_value]
    # assign new values if NA
    modelDT[is.na(Value), Value := internal_value]
    modelDT[is.na(Gain), Gain := leaf_value]
    modelDT[is.na(Feature), Feature := "Leaf"]
    # assign internal_count to Cover, and if Feature is "Leaf", assign leaf_count to Cover
    modelDT[, Cover := internal_count][Feature == "Leaf", Cover := leaf_count]
    # remove unnecessary columns
    modelDT[, c("leaf_count", "internal_count", "leaf_value", "internal_value") := NULL]
    # assign split_index to Node
    modelDT[, Node := split_index]
    # find the maximum value of Node, if Node is NA, assign max_node + leaf_index + 1 to Node
    max_node <- max(modelDT[["Node"]], na.rm = TRUE)
    modelDT[is.na(Node), Node := max_node + leaf_index + 1]
    # adding ID column
    modelDT[, ID := paste(Tree, Node, sep = "-")]
    # remove unnecessary columns
    modelDT[, c("depth", "leaf_index") := NULL]
    modelDT[, parent := node_parent][is.na(parent), parent := leaf_parent]
    modelDT[, c("node_parent", "leaf_parent", "split_index") := NULL]
    # assign the IDs of the matching parent nodes to Yes and No
    modelDT[, Yes := modelDT$ID[match(modelDT$Node, modelDT$parent)]]
    modelDT <- modelDT[nrow(modelDT):1, ]
    modelDT[, No := modelDT$ID[match(modelDT$Node, modelDT$parent)]]
    # which way do the NA's go (this path will get a thicker arrow)
    # for categorical features, NA gets put into the zero group
    modelDT[default_left == TRUE, Missing := Yes]
    modelDT[default_left == FALSE, Missing := No]
    modelDT[.zero_present(Split), Missing := Yes]
    # create the label text
    modelDT[, label := paste0(
        Feature
        , "\nCover: "
        , Cover
        , ifelse(Feature == "Leaf", "", "\nGain: "), ifelse(Feature == "Leaf"
        , ""
        , round(Gain, 4))
        , "\nValue: "
        , round(Value, 4)
    )]
    # style the nodes - same format as xgboost
    modelDT[Node == 0, label := paste0("Tree ", Tree, "\n", label)]
    modelDT[, shape := "rectangle"][Feature == "Leaf", shape := "oval"]
    modelDT[, filledcolor := "Beige"][Feature == "Leaf", filledcolor := "Khaki"]
    # in order to draw the first tree on top:
    modelDT <- modelDT[order(-Tree)]
    nodes <- DiagrammeR::create_node_df(
        n         = nrow(modelDT)
        , ID        = modelDT$ID
        , label     = modelDT$label
        , fillcolor = modelDT$filledcolor
        , shape     = modelDT$shape
        , data      = modelDT$Feature
        , fontcolor = "black"
    )
    # round the edge labels to 4 s.f. if they are numeric
    # as otherwise get too many decimal places and the diagram looks bad
    # would rather not use suppressWarnings
    numeric_idx <- suppressWarnings(!is.na(as.numeric(modelDT[["Split"]])))
    modelDT[numeric_idx, Split := round(as.numeric(Split), 4)]
    # replace indices with feature levels if rules supplied
    
    if (!is.null(rules)) {
        for (f in names(rules)) {
            modelDT[Feature == f & decision_type == "==", Split := .levels.to.names(Split, f, rules)]
        }
    }
    # replace long split names with a message
    modelDT[nchar(Split) > 500, Split := "Split too long to render"]
    # create the edge labels
    edges <- DiagrammeR::create_edge_df(
        from = match(modelDT[Feature != "Leaf", c(ID)] %>% rep(2), modelDT$ID),
        to = match(modelDT[Feature != "Leaf", c(Yes, No)], modelDT$ID),
        label = modelDT[Feature != "Leaf", paste(decision_type, Split)] %>%
            c(rep("", nrow(modelDT[Feature != "Leaf"]))),
        style = modelDT[Feature != "Leaf", ifelse(Missing == Yes, "bold", "solid")] %>%
            c(modelDT[Feature != "Leaf", ifelse(Missing == No, "bold", "solid")]),
        rel = "leading_to"
    )
    # create the graph
    graph <- DiagrammeR::create_graph(
        nodes_df = nodes
        , edges_df = edges
        , attr_theme = NULL
    )
    graph <- DiagrammeR::add_global_graph_attrs(
        graph = graph
        , attr_type = "graph"
        , attr = c("layout", "rankdir")
        , value = c("dot", "LR")
        )
    graph <- DiagrammeR::add_global_graph_attrs(
        graph = graph
        , attr_type = "node"
        , attr = c("color", "style", "fontname")
        , value = c("DimGray", "filled", "Helvetica")
    )
    graph <- DiagrammeR::add_global_graph_attrs(
        graph = graph
        , attr_type = "edge"
        , attr = c("color", "arrowsize", "arrowhead", "fontname")
        , value = c("DimGray", "1.5", "vee", "Helvetica")
    )
    # render the graph
    DiagrammeR::render_graph(graph)
    return(invisible(NULL))
}

.zero_present <- function(x) {
    sapply(strsplit(as.character(x), "||", fixed = TRUE), function(el) {
        any(el == "0")
    })
    return(invisible(NULL))
}

.levels.to.names <- function(x, feature_name, rules) {
    lvls <- sort(rules[[feature_name]])
    result <- strsplit(x, "||", fixed = TRUE)
    result <- lapply(result, as.numeric)
    result <- lapply(result, .levels_to_names)
    result <- lapply(result, paste, collapse = "\n")
    result <- as.character(result)
    return(invisible(NULL))
}

.levels_to_names <- function(x) {
    names(lvls)[as.numeric(x)]
    return(invisible(NULL))
}