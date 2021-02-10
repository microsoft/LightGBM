# We are going to look at how iterating too much might generate observation instability.
# Obviously, we are in a controlled environment, without issues (real rules).
# Do not do this in a real scenario.

library(lightgbm)

# define helper functions for creating plots

# output of `RColorBrewer::brewer.pal(10, "RdYlGn")`, hardcooded here to avoid a dependency
.diverging_palette <- c(
  "#A50026", "#D73027", "#F46D43", "#FDAE61", "#FEE08B"
  , "#D9EF8B", "#A6D96A", "#66BD63", "#1A9850", "#006837"
)

.prediction_depth_plot <- function(df) {
  plot(
    x = df$X
    , y = df$Y
    , type = "p"
    , main = "Prediction Depth"
    , xlab = "Leaf Bin"
    , ylab = "Prediction Probability"
    , pch = 19L
    , col = .diverging_palette[df$binned + 1L]
  )
  legend(
    "topright"
    , title = "bin"
    , legend = sort(unique(df$binned))
    , pch = 19L
    , col = .diverging_palette[sort(unique(df$binned + 1L))]
    , cex = 0.7
  )
}

.prediction_depth_spread_plot <- function(df) {
  plot(
    x = df$binned
    , xlim = c(0L, 9L)
    , y = df$Z
    , type = "p"
    , main = "Prediction Depth Spread"
    , xlab = "Leaf Bin"
    , ylab = "Logloss"
    , pch = 19L
    , col = .diverging_palette[df$binned + 1L]
  )
  legend(
    "topright"
    , title = "bin"
    , legend = sort(unique(df$binned))
    , pch = 19L
    , col = .diverging_palette[sort(unique(df$binned + 1L))]
    , cex = 0.7
  )
}

.depth_density_plot <- function(df) {
  plot(
    x = density(df$Y)
    , xlim = c(min(df$Y), max(df$Y))
    , type = "p"
    , main = "Depth Density"
    , xlab = "Prediction Probability"
    , ylab = "Bin Density"
    , pch = 19L
    , col = .diverging_palette[df$binned + 1L]
  )
  legend(
    "topright"
    , title = "bin"
    , legend = sort(unique(df$binned))
    , pch = 19L
    , col = .diverging_palette[sort(unique(df$binned + 1L))]
    , cex = 0.7
  )
}

# load some data
data(agaricus.train, package = "lightgbm")
train <- agaricus.train
dtrain <- lgb.Dataset(train$data, label = train$label)
data(agaricus.test, package = "lightgbm")
test <- agaricus.test
dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)

# setup parameters and we train a model
params <- list(objective = "regression", metric = "l2")
valids <- list(test = dtest)
model <- lgb.train(
    params
    , dtrain
    , 50L
    , valids
    , min_data = 1L
    , learning_rate = 0.1
    , bagging_fraction = 0.1
    , bagging_freq = 1L
    , bagging_seed = 1L
)

# We create a data.frame with the following structure:
# X = average leaf of the observation throughout all trees
# Y = prediction probability (clamped to [1e-15, 1-1e-15])
# Z = logloss
# binned = binned quantile of average leaf
new_data <- data.frame(
    X = rowMeans(predict(
        model
        , agaricus.test$data
        , predleaf = TRUE
    ))
    , Y = pmin(
        pmax(
            predict(model, agaricus.test$data)
            , 1e-15
        )
        , 1.0 - 1e-15
    )
)
new_data$Z <- -1.0 * (agaricus.test$label * log(new_data$Y) + (1L - agaricus.test$label) * log(1L - new_data$Y))
new_data$binned <- .bincode(
    x = new_data$X
    , breaks = quantile(
        x = new_data$X
        , probs = seq_len(9L) / 10.0
    )
    , right = TRUE
    , include.lowest = TRUE
)
new_data$binned[is.na(new_data$binned)] <- 0L

# We can check the binned content
table(new_data$binned)

# We can plot the binned content
# On the second plot, we clearly notice the lower the bin (the lower the leaf value), the higher the loss
# On the third plot, it is smooth!
.prediction_depth_plot(df = new_data)
.prediction_depth_spread_plot(df = new_data)
.depth_density_plot(df = new_data)

# Now, let's show with other parameters
model2 <- lgb.train(
    params
    , dtrain
    , 100L
    , valids
    , min_data = 1L
    , learning_rate = 1.0
)

# We create the data structure, but for model2
new_data2 <- data.frame(
    X = rowMeans(predict(
        model2
        , agaricus.test$data
        , predleaf = TRUE
    ))
    , Y = pmin(
        pmax(
            predict(
                model2
                , agaricus.test$data
            )
            , 1e-15
        )
      , 1.0 - 1e-15
     )
)
new_data2$Z <- -1.0 * (agaricus.test$label * log(new_data2$Y) + (1L - agaricus.test$label) * log(1L - new_data2$Y))
new_data2$binned <- .bincode(
    x = new_data2$X
    , breaks = quantile(
        x = new_data2$X
        , probs = seq_len(9L) / 10.0
    )
    , right = TRUE
    , include.lowest = TRUE
)
new_data2$binned[is.na(new_data2$binned)] <- 0L

# We can check the binned content
table(new_data2$binned)

# We can plot the binned content
# On the second plot, we clearly notice the lower the bin (the lower the leaf value), the higher the loss
# On the third plot, it is clearly not smooth! We are severely overfitting the data, but the rules are
# real thus it is not an issue
# However, if the rules were not true, the loss would explode.
.prediction_depth_plot(df = new_data2)
.prediction_depth_spread_plot(df = new_data2)
.depth_density_plot(df = new_data2)

# Now, try with very severe overfitting
model3 <- lgb.train(
    params
    , dtrain
    , 1000L
    , valids
    , min_data = 1L
    , learning_rate = 1.0
)

# We create the data structure, but for model3
new_data3 <- data.frame(
    X = rowMeans(predict(
        model3
        , agaricus.test$data
        , predleaf = TRUE
    ))
    , Y = pmin(
        pmax(
            predict(
                model3
                , agaricus.test$data
            )
            , 1e-15
        )
        , 1.0 - 1e-15
    )
)
new_data3$Z <- -1.0 * (agaricus.test$label * log(new_data3$Y) + (1L - agaricus.test$label) * log(1L - new_data3$Y))
new_data3$binned <- .bincode(
    x = new_data3$X
    , breaks = quantile(
        x = new_data3$X
        , probs = seq_len(9L) / 10.0
    )
    , right = TRUE
    , include.lowest = TRUE
)
new_data3$binned[is.na(new_data3$binned)] <- 0L

# We can check the binned content
table(new_data3$binned)

# We can plot the binned content
# On the third plot, it is clearly not smooth! We are severely overfitting the data, but the rules
# are real thus it is not an issue.
# However, if the rules were not true, the loss would explode. See the sudden spikes?
.depth_density_plot(df = new_data3)

# Compare with our second model, the difference is severe. This is smooth.
.depth_density_plot(df = new_data2)
