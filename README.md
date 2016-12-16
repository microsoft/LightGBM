LightGBM, Light Gradient Boosting Machine
=========================================
[![Build Status](https://travis-ci.org/Microsoft/LightGBM.svg?branch=master)](https://travis-ci.org/Microsoft/LightGBM)

LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages:

- Faster training speed and higher efficiency
- Lower memory usage
- Better accuracy
- Parallel learning supported
- Capable of handling large-scale data

For more details, please refer to [Features](https://github.com/Microsoft/LightGBM/wiki/Features).

[Experiments](https://github.com/Microsoft/LightGBM/wiki/Experiments#comparison-experiment) on public datasets show that LightGBM can outperform other existing boosting framework on both efficiency and accuracy, with significant lower memory consumption. What's more, the [experiments](https://github.com/Microsoft/LightGBM/wiki/Experiments#parallel-experiment) show that LightGBM can achieve a linear speed-up by using multiple machines for training in specific settings.

News
----
12/05/2016 : **Categorical Features as input directly**(without one-hot coding). Experiment on [Expo data](http://stat-computing.org/dataexpo/2009/) shows about 8x speed-up with same accuracy compared with one-hot coding (refer to [categorical log]( https://github.com/guolinke/boosting_tree_benchmarks/blob/master/lightgbm/lightgbm_dataexpo_speed.log) and [one-hot log]( https://github.com/guolinke/boosting_tree_benchmarks/blob/master/lightgbm/lightgbm_dataexpo_onehot_speed.log)).
For the setting details, please refer to [IO Parameters](./docs/Parameters.md#io-parameters).

12/02/2016 : Release [**python-package**](./python-package) beta version, welcome to have a try and provide issues and feedback.

Get Started And Documents
-------------------------
To get started, please follow the [Installation Guide](https://github.com/Microsoft/LightGBM/wiki/Installation-Guide) and [Quick Start](https://github.com/Microsoft/LightGBM/wiki/Quick-Start).

* [**Wiki**](https://github.com/Microsoft/LightGBM/wiki)
* [**Installation Guide**](https://github.com/Microsoft/LightGBM/wiki/Installation-Guide)
* [**Quick Start**](https://github.com/Microsoft/LightGBM/wiki/Quick-Start)
* [**Examples**](https://github.com/Microsoft/LightGBM/tree/master/examples)
* [**Features**](https://github.com/Microsoft/LightGBM/wiki/Features)
* [**Parallel Learning Guide**](https://github.com/Microsoft/LightGBM/wiki/Parallel-Learning-Guide)
* [**Configuration**](https://github.com/Microsoft/LightGBM/wiki/Configuration)
* [**Document Indexer**](https://github.com/Microsoft/LightGBM/blob/master/docs/Readme.md)

How to Contribute
-----------------

LightGBM has been developed and used by many active community members. Your help is very valuable to make it better for everyone.

- Check out [call for contributions](https://github.com/Microsoft/LightGBM/issues?q=is%3Aissue+is%3Aopen+label%3Acall-for-contribution) to see what can be improved, or open an issue if you want something.
- Contribute to the [tests](https://github.com/Microsoft/LightGBM/tree/master/tests) to make it more reliable. 
- Contribute to the [documents](https://github.com/Microsoft/LightGBM/tree/master/docs) to make it clearly for everyone.
- Contribute to the [examples](https://github.com/Microsoft/LightGBM/tree/master/examples) to share your experience with other users.
- Check out [Development Guide](./docs/development.md).
- Open issue if you met problems during development.

Microsoft Open Source Code of Conduct
------------
This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
