# We are going to look at how iterating too much might generate observation instability.
# Obviously, we are in a controlled environment, without issues (real rules).
# Do not do this in a real scenario.

# First, we load our libraries
library(lightgbm)
library(ggplot2)

# Second, we load our data
data(agaricus.train, package = "lightgbm")
train <- agaricus.train
dtrain <- lgb.Dataset(train$data, label = train$label)
data(agaricus.test, package = "lightgbm")
test <- agaricus.test
dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)

# Third, we setup parameters and we train a model
params <- list(objective = "regression", metric = "l2")
valids <- list(test = dtest)
model <- lgb.train(params,
                   dtrain,
                   50,
                   valids,
                   min_data = 1,
                   learning_rate = 0.1,
                   bagging_fraction = 0.1,
                   bagging_freq = 1,
                   bagging_seed = 1)

# We create a data.frame with the following structure:
# X = average leaf of the observation throughout all trees
# Y = prediction probability (clamped to [1e-15, 1-1e-15])
# Z = logloss
# binned = binned quantile of average leaf
new_data <- data.frame(X = rowMeans(predict(model,
                                            agaricus.test$data,
                                            predleaf = TRUE)),
                       Y = pmin(pmax(predict(model,
                                             agaricus.test$data), 1e-15), 1 - 1e-15))
new_data$Z <- -(agaricus.test$label * log(new_data$Y) + (1 - agaricus.test$label) * log(1 - new_data$Y))
new_data$binned <- .bincode(x = new_data$X,
                            breaks = quantile(x = new_data$X,
                                              probs = (1:9)/10),
                            right = TRUE,
                            include.lowest = TRUE)
new_data$binned[is.na(new_data$binned)] <- 0
new_data$binned <- as.factor(new_data$binned)

# We can check the binned content
table(new_data$binned)

# We can plot the binned content
# On the second plot, we clearly notice the lower the bin (the lower the leaf value), the higher the loss
# On the third plot, it is smooth!
ggplot(data = new_data, mapping = aes(x = X, y = Y, color = binned)) + geom_point() + theme_bw() + labs(title = "Prediction Depth", x = "Leaf Bin", y = "Prediction Probability")
ggplot(data = new_data, mapping = aes(x = binned, y = Z, fill = binned, group = binned)) + geom_boxplot() + theme_bw() + labs(title = "Prediction Depth Spread", x = "Leaf Bin", y = "Logloss")
ggplot(data = new_data, mapping = aes(x = Y, y = ..count.., fill = binned)) + geom_density(position = "fill") + theme_bw() + labs(title = "Depth Density", x = "Prediction Probability", y = "Bin Density")


# Now, let's show with other parameters
model2 <- lgb.train(params,
                    dtrain,
                    100,
                    valids,
                    min_data = 1,
                    learning_rate = 1)

# We create the data structure, but for model2
new_data2 <- data.frame(X = rowMeans(predict(model2,
                                             agaricus.test$data,
                                             predleaf = TRUE)),
                        Y = pmin(pmax(predict(model2,
                                              agaricus.test$data), 1e-15), 1 - 1e-15))
new_data2$Z <- -(agaricus.test$label * log(new_data2$Y) + (1 - agaricus.test$label) * log(1 - new_data2$Y))
new_data2$binned <- .bincode(x = new_data2$X,
                             breaks = quantile(x = new_data2$X,
                                               probs = (1:9)/10),
                             right = TRUE,
                             include.lowest = TRUE)
new_data2$binned[is.na(new_data2$binned)] <- 0
new_data2$binned <- as.factor(new_data2$binned)

# We can check the binned content
table(new_data2$binned)

# We can plot the binned content
# On the second plot, we clearly notice the lower the bin (the lower the leaf value), the higher the loss
# On the third plot, it is clearly not smooth! We are severely overfitting the data, but the rules are real thus it is not an issue
# However, if the rules were not true, the loss would explode.
ggplot(data = new_data2, mapping = aes(x = X, y = Y, color = binned)) + geom_point() + theme_bw() + labs(title = "Prediction Depth", x = "Leaf Bin", y = "Prediction Probability")
ggplot(data = new_data2, mapping = aes(x = binned, y = Z, fill = binned, group = binned)) + geom_boxplot() + theme_bw() + labs(title = "Prediction Depth Spread", x = "Leaf Bin", y = "Logloss")
ggplot(data = new_data2, mapping = aes(x = Y, y = ..count.., fill = binned)) + geom_density(position = "fill") + theme_bw() + labs(title = "Depth Density", x = "Prediction Probability", y = "Bin Density")


# Now, try with very severe overfitting
model3 <- lgb.train(params,
                    dtrain,
                    1000,
                    valids,
                    min_data = 1,
                    learning_rate = 1)

# We create the data structure, but for model3
new_data3 <- data.frame(X = rowMeans(predict(model3,
                                             agaricus.test$data,
                                             predleaf = TRUE)),
                        Y = pmin(pmax(predict(model3,
                                              agaricus.test$data), 1e-15), 1 - 1e-15))
new_data3$Z <- -(agaricus.test$label * log(new_data3$Y) + (1 - agaricus.test$label) * log(1 - new_data3$Y))
new_data3$binned <- .bincode(x = new_data3$X,
                             breaks = quantile(x = new_data3$X,
                                               probs = (1:9)/10),
                             right = TRUE,
                             include.lowest = TRUE)
new_data3$binned[is.na(new_data3$binned)] <- 0
new_data3$binned <- as.factor(new_data3$binned)

# We can check the binned content
table(new_data3$binned)

# We can plot the binned content
# On the third plot, it is clearly not smooth! We are severely overfitting the data, but the rules are real thus it is not an issue.
# However, if the rules were not true, the loss would explode. See the sudden spikes?
ggplot(data = new_data3, mapping = aes(x = Y, y = ..count.., fill = binned)) + geom_density(position = "fill") + theme_bw() + labs(title = "Depth Density", x = "Prediction Probability", y = "Bin Density")

# Compare with our second model, the difference is severe. This is smooth.
ggplot(data = new_data2, mapping = aes(x = Y, y = ..count.., fill = binned)) + geom_density(position = "fill") + theme_bw() + labs(title = "Depth Density", x = "Prediction Probability", y = "Bin Density")
