This is a page contains all parameters in LightGBM.

***List of other Helpful Links***
* [Parameters](./Parameters.md)
* [Python API Reference](./Python-API.md)

## Convert parameters from XGBoost

LightGBM uses [leaf-wise](https://github.com/Microsoft/LightGBM/wiki/Features#optimization-in-accuracy) tree growth algorithm. But other popular tools, e.g. XGBoost, use depth-wise tree growth. So LightGBM use ```num_leaves``` to control complexity of tree model, and other tools usually use ```max_depth```. Following table is the correspond between leaves and depths. The relation is ```num_leaves = 2^(max_depth) ```.

| max_depth | num_leaves |
| --------- | ---------- |
| 1 | 2 |
| 2 | 4 |
| 3 | 8 |
| 7 | 128 |
| 10 | 1024 |   

## For faster speed

* Use bagging by set ```bagging_fraction``` and ```bagging_freq``` 
* Use feature sub-sampling by set ```feature_fraction```
* Use small ```max_bin```
* Use ```save_binary``` to speed up data loading in future learning
* Use parallel learning, refer to [parallel learning guide](./Parallel-Learning-Guide.md).

## For better accuracy

* Use large ```max_bin``` (may slower)
* Use small ```learning_rate``` with large ```num_iterations```
* Use large ```num_leave```(may over-fitting)
* Use bigger training data
* Try ```dart```

## Deal with over-fitting

* Use small ```max_bin```
* Use small ```num_leaves```
* Use ```min_data_in_leaf``` and ```min_sum_hessian_in_leaf```
* Use bagging by set ```bagging_fraction``` and ```bagging_freq``` 
* Use feature sub-sampling by set ```feature_fraction```
* Use bigger training data
* Try ```lambda_l1```, ```lambda_l2``` and ```min_gain_to_split``` to regularization
* Try ```max_depth``` to avoid growing deep tree 
