LightGBM, Light Gradient Boosting Machine
=========================================

[![Build Status](https://travis-ci.org/Microsoft/LightGBM.svg?branch=master)](https://travis-ci.org/Microsoft/LightGBM)
[![Windows Build Status](https://ci.appveyor.com/api/projects/status/1ys5ot401m0fep6l/branch/master?svg=true)](https://ci.appveyor.com/project/guolinke/lightgbm/branch/master)
[![Documentation Status](https://readthedocs.org/projects/lightgbm/badge/?version=latest)](https://lightgbm.readthedocs.io/)
[![GitHub Issues](https://img.shields.io/github/issues/Microsoft/LightGBM.svg)](https://github.com/Microsoft/LightGBM/issues)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Microsoft/LightGBM/blob/master/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/lightgbm.svg)](https://pypi.python.org/pypi/lightgbm)
[![PyPI Version](https://badge.fury.io/py/lightgbm.svg)](https://badge.fury.io/py/lightgbm)
[![Join the chat at https://gitter.im/Microsoft/LightGBM](https://badges.gitter.im/Microsoft/LightGBM.svg)](https://gitter.im/Microsoft/LightGBM?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Slack](https://lightgbm-slack-autojoin.herokuapp.com/badge.svg)](https://lightgbm-slack-autojoin.herokuapp.com/)

LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages:

- Faster training speed and higher efficiency
- Lower memory usage
- Better accuracy
- Parallel and GPU learning supported
- Capable of handling large-scale data

For more details, please refer to [Features](https://github.com/Microsoft/LightGBM/blob/master/docs/Features.rst).

[Comparison experiments](https://github.com/Microsoft/LightGBM/blob/master/docs/Experiments.rst#comparison-experiment) on public datasets show that LightGBM can outperform existing boosting frameworks on both efficiency and accuracy, with significantly lower memory consumption. What's more, the [parallel experiments](https://github.com/Microsoft/LightGBM/blob/master/docs/Experiments.rst#parallel-experiment) show that LightGBM can achieve a linear speed-up by using multiple machines for training in specific settings.

News
----

08/15/2017 : Optimal split for categorical features.

07/13/2017 : [Gitter](https://gitter.im/Microsoft/LightGBM) is avaiable.

06/20/2017 : Python-package is on [PyPI](https://pypi.python.org/pypi/lightgbm) now.

06/09/2017 : [LightGBM Slack team](https://lightgbm.slack.com) is available.

05/03/2017 : LightGBM v2 stable release.

04/10/2017 : LightGBM supports GPU-accelerated tree learning now. Please read our [GPU Tutorial](./docs/GPU-Tutorial.rst) and [Performance Comparison](./docs/GPU-Performance.rst).

02/20/2017 : Update to LightGBM v2.

02/12/2017: LightGBM v1 stable release.

01/08/2017 : Release [**R-package**](https://github.com/Microsoft/LightGBM/tree/master/R-package) beta version, welcome to have a try and provide feedback.

12/05/2016 : **Categorical Features as input directly** (without one-hot coding). 

12/02/2016 : Release [**Python-package**](https://github.com/Microsoft/LightGBM/tree/master/python-package) beta version, welcome to have a try and provide feedback.

More detailed update logs : [Key Events](https://github.com/Microsoft/LightGBM/blob/master/docs/Key-Events.md).


External (unofficial) Repositories
----------------------------------

Julia Package: https://github.com/Allardvm/LightGBM.jl

JPMML: https://github.com/jpmml/jpmml-lightgbm


Get Started and Documentation
-----------------------------

Install by following the [guide](https://github.com/Microsoft/LightGBM/blob/master/docs/Installation-Guide.rst) for the command line program, [Python-package](https://github.com/Microsoft/LightGBM/tree/master/python-package) or [R-package](https://github.com/Microsoft/LightGBM/tree/master/R-package). Then please see the [Quick Start](https://github.com/Microsoft/LightGBM/blob/master/docs/Quick-Start.rst) guide.

Our primary documentation is at https://lightgbm.readthedocs.io/ and is generated from this repository.

Next you may want to read:

* [**Examples**](https://github.com/Microsoft/LightGBM/tree/master/examples) showing command line usage of common tasks
* [**Features**](https://github.com/Microsoft/LightGBM/blob/master/docs/Features.rst) and algorithms supported by LightGBM
* [**Parameters**](https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst) is an exhaustive list of customization you can make
* [**Parallel Learning**](https://github.com/Microsoft/LightGBM/blob/master/docs/Parallel-Learning-Guide.rst) and [**GPU Learning**](https://github.com/Microsoft/LightGBM/blob/master/docs/GPU-Tutorial.rst) can speed up computation
* [**Laurae++ interactive documentation**](https://sites.google.com/view/lauraepp/parameters) is a detailed guide for hyperparameters

Documentation for contributors:

* [**How we update readthedocs.io**](https://github.com/Microsoft/LightGBM/blob/master/docs/README.rst)
* Check out the [**Development Guide**](https://github.com/Microsoft/LightGBM/blob/master/docs/Development-Guide.rst).

Support
-------

* Ask a question [on Stack Overflow with the `lightgbm` tag](https://stackoverflow.com/questions/ask?tags=lightgbm), we monitor this for new questions.
* Discuss on the [LightGBM Gitter](https://gitter.im/Microsoft/LightGBM).
* Discuss on the [LightGBM Slack team](https://lightgbm.slack.com).
  * Use [this invite link](https://lightgbm-slack-autojoin.herokuapp.com/) to join the team.
* Open **bug reports** and **feature requests** (not questions) on [GitHub issues](https://github.com/Microsoft/LightGBM/issues).

How to Contribute
-----------------

LightGBM has been developed and used by many active community members. Your help is very valuable to make it better for everyone.

- Check out [call for contributions](https://github.com/Microsoft/LightGBM/issues?q=is%3Aissue+is%3Aopen+label%3Acall-for-contribution) to see what can be improved, or open an issue if you want something.
- Contribute to the [tests](https://github.com/Microsoft/LightGBM/tree/master/tests) to make it more reliable.
- Contribute to the [documents](https://github.com/Microsoft/LightGBM/tree/master/docs) to make it clearer for everyone.
- Contribute to the [examples](https://github.com/Microsoft/LightGBM/tree/master/examples) to share your experience with other users.
- Open issue if you met problems during development.

Microsoft Open Source Code of Conduct
-------------------------------------

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

Reference Paper
---------------

Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree). In Advances in Neural Information Processing Systems (NIPS), pp. 3149-3157. 2017.
