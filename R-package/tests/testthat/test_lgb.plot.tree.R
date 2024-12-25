NROUNDS <- 10L
MAX_DEPTH <- 3L
N <- nrow(iris)
X <- data.matrix(iris[2L:4L])
FEAT <- colnames(X)
NCLASS <- nlevels(iris[, 5L])

model_reg <- lgb.train(
  params = list(
    objective = "regression"
    , num_threads = .LGB_MAX_THREADS
    , max.depth = MAX_DEPTH
  )
  , data = lgb.Dataset(X, label = iris[, 1L])
  , verbose = .LGB_VERBOSITY
  , nrounds = NROUNDS
)

model_binary <- lgb.train(
  params = list(
    objective = "binary"
    , num_threads = .LGB_MAX_THREADS
    , max.depth = MAX_DEPTH
  )
  , data = lgb.Dataset(X, label = iris[, 5L] == "setosa")
  , verbose = .LGB_VERBOSITY
  , nrounds = NROUNDS
)

model_multiclass <- lgb.train(
  params = list(
    objective = "multiclass"
    , num_threads = .LGB_MAX_THREADS
    , max.depth = MAX_DEPTH
    , num_classes = NCLASS
  )
  , data = lgb.Dataset(X, label = as.integer(iris[, 5L]) - 1L)
  , verbose = .LGB_VERBOSITY
  , nrounds = NROUNDS
)

model_rank <- lgb.train(
  params = list(
    objective = "lambdarank"
    , num_threads = .LGB_MAX_THREADS
    , max.depth = MAX_DEPTH
    , lambdarank_truncation_level = 3L
  )
  , data = lgb.Dataset(
    X
    , label = as.integer(iris[, 1L] > 5.8)
    , group = rep(10L, times = 15L)
  )
  , verbose = .LGB_VERBOSITY
  , nrounds = NROUNDS
)

models <- list(
  reg = model_reg
  , bin = model_binary
  , multi = model_multiclass
  , rank = model_rank
)

for (model_name in names(models)){
  model <- models[[model_name]]
  expected_n_trees <- NROUNDS
  if (model_name == "multi") {
    expected_n_trees <- NROUNDS * NCLASS
  }
  df <- as.data.frame(lgb.model.dt.tree(model))
  df_list <- split(df, f = df$tree_index, drop = TRUE)
  df_leaf <- df[!is.na(df$leaf_index), ]
  df_internal <- df[is.na(df$leaf_index), ]

  test_that("lgb.plot.tree fails when a non existing tree is selected", {
    expect_error({
      lgb.plot.tree(model, 0)
    }, regexp = "lgb.plot.tree: Value of 'tree' should be between 1 and the total number of trees in the model")
  })
  test_that("lgb.plot.tree fails when a non existing tree is selected", {
    expect_error({
      lgb.plot.tree(model, 999)
    }, regexp = "lgb.plot.tree: Value of 'tree' should be between 1 and the total number of trees in the model")
  })
  test_that("lgb.plot.tree fails when a non numeric tree is selected", {
    expect_error({
      lgb.plot.tree(model, "a")
    }, regexp = "lgb.plot.tree: Has to be an integer numeric")
  })
  test_that("lgb.plot.tree fails when a non integer tree is selected", {
    expect_error({
      lgb.plot.tree(model, 1.5)
    }, regexp = "lgb.plot.tree: Has to be an integer numeric")
  })
  test_that("lgb.plot.tree fails when a non lgb.Booster model is passed", {
    expect_error({
      lgb.plot.tree(1, 0)
    }, regexp = "lgb.plot.tree: model should be an 'lgb.Booster'")
  })
}

