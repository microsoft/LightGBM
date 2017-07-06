# Parameters Tuning

This is a page contains all parameters in LightGBM.

***List of other Helpful Links***
* [Parameters](./Parameters.md)
* [Python API Reference](./Python-API.md)

## Tune parameters for the leaf-wise(best-first) tree

LightGBM uses the [leaf-wise](https://github.com/Microsoft/LightGBM/wiki/Features#optimization-in-accuracy) tree growth algorithm, while many other popular tools use depth-wise tree growth. Compared with depth-wise growth, the leaf-wise algorithm can convenge much faster. However, the leaf-wise growth may be over-fitting if not used with the appropriate parameters. 

To get good results using a leaf-wise tree, these are some important parameters:

1. ```num_leaves```. This is the main parameter to control the complexity of the tree model. Theoretically, we can set ```num_leaves = 2^(max_depth) ``` to convert from depth-wise tree. However, this simple conversion is not good in practice. The reason is, when number of leaves are the same, the leaf-wise tree is much deeper than depth-wise tree. As a result, it may be over-fitting. Thus, when trying to tune the ```num_leaves```, we should let it be smaller than ```2^(max_depth)```. For example, when the ```max_depth=6``` the depth-wise tree can get good accuracy, but setting ```num_leaves``` to ```127``` may cause over-fitting, and setting it to ```70``` or ```80``` may get better accuracy than depth-wise. Actually, the concept ```depth``` can be forgotten in leaf-wise tree, since it doesn't have a correct mapping from ```leaves``` to ```depth```. 

2. ```min_data_in_leaf```. This is a very important parameter to deal with over-fitting in leaf-wise tree. Its value depends on the number of training data and ```num_leaves```. Setting it to a large value can avoid growing too deep a tree, but may cause under-fitting. In practice, setting it to hundreds or thousands is enough for a large dataset. 

3. ```max_depth```. You also can use ```max_depth``` to limit the tree depth explicitly. 


## For faster speed

* Use bagging by setting ```bagging_fraction``` and ```bagging_freq``` 
* Use feature sub-sampling by setting ```feature_fraction```
* Use small ```max_bin```
* Use ```save_binary``` to speed up data loading in future learning
* Use parallel learning, refer to [parallel learning guide](./Parallel-Learning-Guide.md).

## For better accuracy

* Use large ```max_bin``` (may be slower)
* Use small ```learning_rate``` with large ```num_iterations```
* Use large ```num_leaves```(may cause over-fitting)
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
