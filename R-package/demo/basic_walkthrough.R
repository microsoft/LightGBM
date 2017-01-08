require(lightgbm)
require(methods)

# we load in the agaricus dataset
# In this example, we are aiming to predict whether a mushroom is edible
data(agaricus.train, package='lightgbm')
data(agaricus.test, package='lightgbm')
train <- agaricus.train
test <- agaricus.test
# the loaded data is stored in sparseMatrix, and label is a numeric vector in {0,1}
class(train$label)
class(train$data)

#-------------Basic Training using lightgbm-----------------
# this is the basic usage of lightgbm you can put matrix in data field
# note: we are putting in sparse matrix here, lightgbm naturally handles sparse input
# use sparse matrix when your feature is sparse(e.g. when you are using one-hot encoding vector)
print("Training lightgbm with sparseMatrix")
bst <- lightgbm(data = train$data, label = train$label, num_leaves = 4, learning_rate = 1, nrounds = 2,
                objective = "binary")
# alternatively, you can put in dense matrix, i.e. basic R-matrix
print("Training lightgbm with Matrix")
bst <- lightgbm(data = as.matrix(train$data), label = train$label, num_leaves = 4, learning_rate = 1, nrounds = 2,
                objective = "binary")

# you can also put in lgb.Dataset object, which stores label, data and other meta datas needed for advanced features
print("Training lightgbm with lgb.Dataset")
dtrain <- lgb.Dataset(data = train$data, label = train$label)
bst <- lightgbm(data = dtrain, num_leaves = 4, learning_rate = 1, nrounds = 2, 
               objective = "binary")

# Verbose = 0,1,2
print("Train lightgbm with verbose 0, no message")
bst <- lightgbm(data = dtrain, num_leaves = 4, learning_rate = 1, nrounds = 2,
                objective = "binary", verbose = 0)
print("Train lightgbm with verbose 1, print evaluation metric")
bst <- lightgbm(data = dtrain, num_leaves = 4, learning_rate = 1, nrounds = 2,
               nthread = 2, objective = "binary", verbose = 1)
print("Train lightgbm with verbose 2, also print information about tree")
bst <- lightgbm(data = dtrain, num_leaves = 4, learning_rate = 1, nrounds = 2,
               nthread = 2, objective = "binary", verbose = 2)

# you can also specify data as file path to a LibSVM/TCV/CSV format input
# since we do not have this file with us, the following line is just for illustration
# bst <- lightgbm(data = 'agaricus.train.svm', num_leaves = 4, learning_rate = 1, nrounds = 2,objective = "binary")

#--------------------basic prediction using lightgbm--------------
# you can do prediction using the following line
# you can put in Matrix, sparseMatrix, or lgb.Dataset 
pred <- predict(bst, test$data)
err <- mean(as.numeric(pred > 0.5) != test$label)
print(paste("test-error=", err))

#-------------------save and load models-------------------------
# save model to binary local file
lgb.save(bst, "lightgbm.model")
# load binary model to R
bst2 <- lgb.load("lightgbm.model")
pred2 <- predict(bst2, test$data)
# pred2 should be identical to pred
print(paste("sum(abs(pred2-pred))=", sum(abs(pred2-pred))))


#----------------Advanced features --------------
# to use advanced features, we need to put data in lgb.Dataset
dtrain <- lgb.Dataset(data = train$data, label=train$label, free_raw_data=FALSE)
dtest <- lgb.Dataset(data = test$data, label=test$label, free_raw_data=FALSE)
#---------------Using valids----------------
# valids is a list of lgb.Dataset, each of them is tagged with name
valids <- list(train=dtrain, test=dtest)
# to train with valids, use lgb.train, which contains more advanced features
# valids allows us to monitor the evaluation result on all data in the list 
print("Train lightgbm using lgb.train with valids")
bst <- lgb.train(data=dtrain, num_leaves=4, learning_rate=1, nrounds=2, valids=valids,
                 nthread = 2, objective = "binary")
# we can change evaluation metrics, or use multiple evaluation metrics
print("train lightgbm using lgb.train with valids, watch logloss and error")
bst <- lgb.train(data=dtrain, num_leaves=4, learning_rate=1, nrounds=2, valids=valids,
                 eval = c("binary_error","binary_logloss"),
                 nthread = 2, objective = "binary")

# lgb.Dataset can also be saved using lgb.Dataset.save
lgb.Dataset.save(dtrain, "dtrain.buffer")
# to load it in, simply call lgb.Dataset
dtrain2 <- lgb.Dataset("dtrain.buffer")
bst <- lgb.train(data=dtrain2, num_leaves=4, learning_rate=1, nrounds=2, valids=valids,
                 nthread = 2, objective = "binary")
# information can be extracted from lgb.Dataset using getinfo
label = getinfo(dtest, "label")
pred <- predict(bst, test$data)
err <- as.numeric(sum(as.integer(pred > 0.5) != label))/length(label)
print(paste("test-error=", err))


