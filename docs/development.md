Development Guide
==================

Algorithms
----------

Refer to [Features](https://github.com/Microsoft/LightGBM/wiki/Features) to get important algorithms used in LightGBM.

Classes And Code Structure 
--------------------------

### Important Classes

| Class | description |
| ----- | --------- |
| `Application` | The entrance of application, including training and prediction logic |
| `Bin` | Data structure used for store feature discrete values(converted from float values) | 
| `Boosting` | Boosting interface, current implementation is GBDT and DART |
| `Config` | Store parameters and configurations|
| `Dataset` | Store information of dataset |
| `DatasetLoader` | Used to construct dataset | 
| `Feature` | Store One column feature |
| `Metric` | Evaluation metrics |
| `Network` | Newwork interfaces and communication algorithms |
| `ObjectiveFunction` | Objective function used to train |
| `Tree` | Store information of tree model |
| `TreeLearner` | Used to learn trees | 

### Code Structure

| Path | description |
| ----- | --------- |
| ./include | header files |
| ./include/utils | some common functions |
| ./src/application | Implementations of training and prediction logic |
| ./src/boosting | Implementations of Boosting |
| ./src/io | Implementations of IO relatived classes, including  `Bin`, `Config`, `Dataset`, `DatasetLoader`, `Feature` and `Tree`|
| ./src/metric | Implementations of metrics |
| ./src/network | Implementations of network functions |
| ./src/objective | Implementations of objective functions |
| ./src/treelearner | Implementations of tree learners |

### API Documents

LightGBM support use [doxygen](http://www.stack.nl/~dimitri/doxygen/) to generate documents for classes and functions.

C API
-----
Refere to the comments in [c_api.h](https://github.com/Microsoft/LightGBM/blob/master/include/LightGBM/c_api.h).

High level Language package
---------------------------

Follow the implementation of [python-package](https://github.com/Microsoft/LightGBM/tree/master/python-package/lightgbm).

Ask Questions
-------------
Feel free to open [issues](https://github.com/Microsoft/LightGBM/issues) if you met problems.



