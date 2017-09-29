# Advanced Topics

## Missing value handle

* LightGBM enables the missing value handle by default, you can disable it by set ```use_missing=false```.
* LightGBM uses NA (NAN) to represent the missing value by default, you can change it to use zero by set ```zero_as_missing=true```.
* When ```zero_as_missing=false``` (default), the unshown value in sparse matrices (and LightSVM) is treated as zeros. 
* When ```zero_as_missing=true```, NA and zeros (including unshown value in sparse matrices (and LightSVM)) are treated as missing. 

## Categorical feature support

* LightGBM can offer a good accuracy when using native categorical features. Not like simply one-hot coding, LightGBM can find the optimal split of categorical features. Such a optimal split can provide the much better accuracy than one-hot coding solution. 
* Use `categorical_feature` to specific the categorical features. Refer to the parameter `categorical_feature` in [Parameters](./Parameters.md).
* Need to convert to `int` type first, and only support non-negative numbers. It is better to convert into continues ranges.
* Use `max_cat_group`, `cat_smooth_ratio` to deal with over-fitting (when #data is small or #category is large).
* For categocal features with high cardinality (#categoriy is large), it is better to convert it to numerical features. 

## LambdaRank 

* The label should be `int` type, and larger number represent the higher relevance (e.g. 0:bad, 1:fair, 2:good, 3:perfect).
* Use `label_gain` to set the gain(weight) of `int` label.
* Use `max_position` to set the NDCG optimization position.

## Parameters Tuning

* Refer to [Parameters tuning](./Parameters-tuning.md).

## GPU support

* Refer to [GPU Tutorial](./GPU-Tutorial.md) and [GPU Targets](./GPU-Targets.md).

## Parallel Learning 

* Refer to [Parallel Learning Guide](./Parallel-Learning-Guide.rst).
