#' @name lgb.plot.tree
#' @title Plot a single LightGBM tree using DiagrammeR.
#' @description The \code{lgb.plot.tree} function creates a DiagrammeR plot of a single LightGBM tree.
#' @param model a \code{lgb.Booster} object.
#' @param tree an integer specifying the tree to plot.
#' @param rules a list of rules to replace the split values with feature levels.
#'
#' @return
#' The \code{lgb.plot.tree} function creates a DiagrammeR plot.
#'
#' @details
#' The \code{lgb.plot.tree} function creates a DiagrammeR plot of a single LightGBM tree. The tree is extracted from the model and displayed as a directed graph. The nodes are labelled with the feature, split value, gain, cover and value. The edges are labelled with the decision type and split value. The nodes are styled with a rectangle shape and filled with a beige colour. Leaf nodes are styled with an oval shape and filled with a khaki colour. The graph is rendered using the dot layout with a left-to-right rank direction. The nodes are coloured dim gray with a filled style and a Helvetica font. The edges are coloured dim gray with a solid style, a 1.5 arrow size, a vee arrowhead and a Helvetica font.
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
#'     metric = "l2",
#'     min_data = 1L,
#'     learning_rate = 1.0
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

# function to plot a single LightGBM tree using DiagrammeR
lgb.plot.tree <- function(model = NULL, tree = NULL, rules = NULL) {
    # check model is lgb.Booster
    if (!inherits(model, "lgb.Booster")) {
        stop("model: Has to be an object of class lgb.Booster")
    }
    # check DiagrammeR is available
    if (!requireNamespace("DiagrammeR", quietly = TRUE)) {
        stop("DiagrammeR package is required for lgb.plot.tree",
            call. = FALSE
        )
    }
    # tree must be numeric
    if (!inherits(tree, "numeric")) {
        stop("tree: Has to be an integer numeric")
    }
    # tree must be integer
    if (tree %% 1 != 0) {
        stop("tree: Has to be an integer numeric")
    }
    # extract data.table model structure
    dt <- lgb.model.dt.tree(model)
    # check that tree is less than or equal to the maximum tree index in the model
    if (tree > max(dt$tree_index)) {
        stop("tree: has to be less than the number of trees in the model")
    }
    # filter dt to just the rows for the selected tree
    dt <- dt[tree_index == tree, ]
    # change the column names to shorter more diagram friendly versions
    data.table::setnames(dt, old = c("tree_index", "split_feature", "threshold", "split_gain"), new = c("Tree", "Feature", "Split", "Gain"))
    dt[, Value := 0.0]
    dt[, Value := leaf_value]
    dt[is.na(Value), Value := internal_value]
    dt[is.na(Gain), Gain := leaf_value]
    dt[is.na(Feature), Feature := "Leaf"]
    dt[, Cover := internal_count][Feature == "Leaf", Cover := leaf_count]
    dt[, c("leaf_count", "internal_count", "leaf_value", "internal_value") := NULL]
    dt[, Node := split_index]
    max_node <- max(dt[["Node"]], na.rm = TRUE)
    dt[is.na(Node), Node := max_node + leaf_index + 1]
    dt[, ID := paste(Tree, Node, sep = "-")]
    dt[, c("depth", "leaf_index") := NULL]
    dt[, parent := node_parent][is.na(parent), parent := leaf_parent]
    dt[, c("node_parent", "leaf_parent", "split_index") := NULL]
    dt[, Yes := dt$ID[match(dt$Node, dt$parent)]]
    dt <- dt[nrow(dt):1, ]
    dt[, No := dt$ID[match(dt$Node, dt$parent)]]
    # which way do the NA's go (this path will get a thicker arrow)
    # for categorical features, NA gets put into the zero group
    dt[default_left == TRUE, Missing := Yes]
    dt[default_left == FALSE, Missing := No]
    zero_present <- function(x) {
        sapply(strsplit(as.character(x), "||", fixed = TRUE), function(el) {
            any(el == "0")
        })
    }
    dt[zero_present(Split), Missing := Yes]
    # dt[, c('parent', 'default_left') := NULL]
    # data.table::setcolorder(dt, c('Tree','Node','ID','Feature','decision_type','Split','Yes','No','Missing','Gain','Cover','Value'))
    # create the label text
    dt[, label := paste0(
        Feature,
        "\nCover: ", Cover,
        ifelse(Feature == "Leaf", "", "\nGain: "), ifelse(Feature == "Leaf", "", round(Gain, 4)),
        "\nValue: ", round(Value, 4)
    )]
    # style the nodes - same format as xgboost
    dt[Node == 0, label := paste0("Tree ", Tree, "\n", label)]
    dt[, shape := "rectangle"][Feature == "Leaf", shape := "oval"]
    dt[, filledcolor := "Beige"][Feature == "Leaf", filledcolor := "Khaki"]
    # in order to draw the first tree on top:
    dt <- dt[order(-Tree)]
    nodes <- DiagrammeR::create_node_df(
        n         = nrow(dt),
        ID        = dt$ID,
        label     = dt$label,
        fillcolor = dt$filledcolor,
        shape     = dt$shape,
        data      = dt$Feature,
        fontcolor = "black"
    )
    # round the edge labels to 4 s.f. if they are numeric
    # as otherwise get too many decimal places and the diagram looks bad
    # would rather not use suppressWarnings
    numeric_idx <- suppressWarnings(!is.na(as.numeric(dt[["Split"]])))
    dt[numeric_idx, Split := round(as.numeric(Split), 4)]
    # replace indices with feature levels if rules supplied
    levels.to.names <- function(x, feature_name, rules) {
        lvls <- sort(rules[[feature_name]])
        result <- strsplit(x, "||", fixed = TRUE)
        result <- lapply(result, as.numeric)
        levels_to_names <- function(x) {
            names(lvls)[as.numeric(x)]
        }
        result <- lapply(result, levels_to_names)
        result <- lapply(result, paste, collapse = "\n")
        result <- as.character(result)
    }
    if (!is.null(rules)) {
        for (f in names(rules)) {
            dt[Feature == f & decision_type == "==", Split := levels.to.names(Split, f, rules)]
        }
    }
    # replace long split names with a message
    dt[nchar(Split) > 500, Split := "Split too long to render"]
    # create the edge labels
    edges <- DiagrammeR::create_edge_df(
        from = match(dt[Feature != "Leaf", c(ID)] %>% rep(2), dt$ID),
        to = match(dt[Feature != "Leaf", c(Yes, No)], dt$ID),
        label = dt[Feature != "Leaf", paste(decision_type, Split)] %>%
            c(rep("", nrow(dt[Feature != "Leaf"]))),
        style = dt[Feature != "Leaf", ifelse(Missing == Yes, "bold", "solid")] %>%
            c(dt[Feature != "Leaf", ifelse(Missing == No, "bold", "solid")]),
        rel = "leading_to"
    )
    # create the graph
    graph <- DiagrammeR::create_graph(
        nodes_df = nodes,
        edges_df = edges,
        attr_theme = NULL
    ) %>%
        DiagrammeR::add_global_graph_attrs(
            attr_type = "graph",
            attr = c("layout", "rankdir"),
            value = c("dot", "LR")
        ) %>%
        DiagrammeR::add_global_graph_attrs(
            attr_type = "node",
            attr = c("color", "style", "fontname"),
            value = c("DimGray", "filled", "Helvetica")
        ) %>%
        DiagrammeR::add_global_graph_attrs(
            attr_type = "edge",
            attr = c("color", "arrowsize", "arrowhead", "fontname"),
            value = c("DimGray", "1.5", "vee", "Helvetica")
        )
    # render the graph
    DiagrammeR::render_graph(graph)
}
