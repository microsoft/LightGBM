NROUNDS <- 10L
MAX_DEPTH <- 3L
N <- nrow(iris)
FEATURES <- colnames(iris[2L:5L])

model_reg <- lgb.train(
  params = list(
    objective = "regression"
    ,  num_threads = .LGB_MAX_THREADS
    , max.depth = MAX_DEPTH
  )
  , data = lgb.Dataset(data.matrix(iris[FEATURES]), label = iris[, 1L])
  , verbose = .LGB_VERBOSITY
  , nrounds = NROUNDS
)

df <- as.data.frame(lgb.model.dt.tree(model_reg))
df_list <- split(df, f = df$tree_index, drop = TRUE)

df_leaf <- df[!is.na(df$leaf_index), ]
df_internal <- df[is.na(df$leaf_index), ]

test_that("lgb.model.dt.tree() returns the right number of trees", {
  expect_equal(length(unique(df$tree_index)), NROUNDS)
})

test_that("Tree index from lgb.model.dt.tree() is in 0:(NROUNS-1)", {
  expect_equal(unique(df$tree_index), (0L:(NROUNDS - 1L)))
})

test_that("Depth calculated from lgb.model.dt.tree() respects max.depth", {
  expect_true(max(df$depth) <= MAX_DEPTH)
})

test_that("Each tree from lgb.model.dt.tree() has single root node", {
  expect_equal(
    unname(sapply(df_list, function(df) sum(df$depth == 0L)))
    , rep(1L, NROUNDS)
  )
})

test_that("Each tree from lgb.model.dt.tree() has two depth 1 nodes", {
  expect_equal(
    unname(sapply(df_list, function(df) sum(df$depth == 1L)))
    , rep(2L, NROUNDS)
  )
})

test_that("leaves from lgb.model.dt.tree() do not have split info", {
  internal_node_cols <- c(
    "split_index"
    , "split_feature"
    , "split_gain"
    , "threshold"
    , "decision_type"
    , "default_left"
    , "internal_value"
    , "internal_count"
  )
  expect_true(all(is.na(df_leaf[internal_node_cols])))
})

test_that("leaves from lgb.model.dt.tree() have valid leaf info", {
  expect_true(all(df_leaf$leaf_index %in% 0L:(2.0^MAX_DEPTH - 1.0)))
  expect_true(all(is.finite(df_leaf$leaf_value)))
  expect_true(all(df_leaf$leaf_count > 0L & df_leaf$leaf_count <= N))
})

test_that("non-leaves from lgb.model.dt.tree() do not have leaf info", {
  leaf_node_cols <- c("leaf_index", "leaf_parent", "leaf_value", "leaf_count")
  expect_true(all(is.na(df_internal[leaf_node_cols])))
})

test_that("non-leaves from lgb.model.dt.tree() have valid split info", {
  expect_true(
    all(
      sapply(
        split(df_internal, df_internal$tree_index),
        function(x) all(x$split_index %in% 0L:(nrow(x) - 1L))
      )
    )
  )

  expect_true(all(df_internal$split_feature %in% FEATURES))

  num_cols <- c("split_gain", "threshold", "internal_value")
  expect_true(all(is.finite(unlist(df_internal[, num_cols]))))

  # range of decision type?
  expect_true(all(df_internal$default_left %in% c(TRUE, FALSE)))

  counts <- df_internal$internal_count
  expect_true(all(counts > 1L & counts <= N))
})
