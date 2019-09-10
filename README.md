LightGBM, Light Gradient Boosting Machine
=========================================

[![Azure Pipelines Build Status](https://lightgbm-ci.visualstudio.com/lightgbm-ci/_apis/build/status/Microsoft.LightGBM?branchName=master)](https://lightgbm-ci.visualstudio.com/lightgbm-ci/_build/latest?definitionId=1)
[![Appveyor Build Status](https://ci.appveyor.com/api/projects/status/1ys5ot401m0fep6l/branch/master?svg=true)](https://ci.appveyor.com/project/guolinke/lightgbm/branch/master)
[![Travis Build Status](https://travis-ci.org/microsoft/LightGBM.svg?branch=master)](https://travis-ci.org/microsoft/LightGBM)
[![Documentation Status](https://readthedocs.org/projects/lightgbm/badge/?version=latest)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/microsoft/LightGBM/blob/master/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/lightgbm.svg)](https://pypi.org/project/lightgbm)
[![PyPI Version](https://img.shields.io/pypi/v/lightgbm.svg)](https://pypi.org/project/lightgbm)
[![Join Gitter at https://gitter.im/Microsoft/LightGBM](https://badges.gitter.im/Microsoft/LightGBM.svg)](https://gitter.im/Microsoft/LightGBM?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Slack](https://lightgbm-slack-autojoin.herokuapp.com/badge.svg)](https://lightgbm-slack-autojoin.herokuapp.com)

LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages:

- Faster training speed and higher efficiency.
- Lower memory usage.
- Better accuracy.
- Support of parallel and GPU learning.
- Capable of handling large-scale data.

For further details, please refer to [Features](https://github.com/microsoft/LightGBM/blob/master/docs/Features.rst).

Benefitting from these advantages, LightGBM is being widely-used in many [winning solutions](https://github.com/microsoft/LightGBM/blob/master/examples/README.md#machine-learning-challenge-winning-solutions) of machine learning competitions.

[Comparison experiments](https://github.com/microsoft/LightGBM/blob/master/docs/Experiments.rst#comparison-experiment) on public datasets show that LightGBM can outperform existing boosting frameworks on both efficiency and accuracy, with significantly lower memory consumption. What's more, [parallel experiments](https://github.com/microsoft/LightGBM/blob/master/docs/Experiments.rst#parallel-experiment) show that LightGBM can achieve a linear speed-up by using multiple machines for training in specific settings.

Get Started and Documentation
-----------------------------

Our primary documentation is at https://lightgbm.readthedocs.io/ and is generated from this repository. If you are new to LightGBM, follow [the installation instructions](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html) on that site.

Next you may want to read:

* [**Examples**](https://github.com/microsoft/LightGBM/tree/master/examples) showing command line usage of common tasks.
* [**Features**](https://github.com/microsoft/LightGBM/blob/master/docs/Features.rst) and algorithms supported by LightGBM.
* [**Parameters**](https://github.com/microsoft/LightGBM/blob/master/docs/Parameters.rst) is an exhaustive list of customization you can make.
* [**Parallel Learning**](https://github.com/microsoft/LightGBM/blob/master/docs/Parallel-Learning-Guide.rst) and [**GPU Learning**](https://github.com/microsoft/LightGBM/blob/master/docs/GPU-Tutorial.rst) can speed up computation.
* [**Laurae++ interactive documentation**](https://sites.google.com/view/lauraepp/parameters) is a detailed guide for hyperparameters.

Documentation for contributors:

* [**How we update readthedocs.io**](https://github.com/microsoft/LightGBM/blob/master/docs/README.rst).
* Check out the [**Development Guide**](https://github.com/microsoft/LightGBM/blob/master/docs/Development-Guide.rst).


News
----

08/15/2017 : Optimal split for categorical features.

07/13/2017 : [Gitter](https://gitter.im/Microsoft/LightGBM) is available.

06/20/2017 : Python-package is on [PyPI](https://pypi.org/project/lightgbm) now.

06/09/2017 : [LightGBM Slack team](https://lightgbm.slack.com) is available.

05/03/2017 : LightGBM v2 stable release.

04/10/2017 : LightGBM supports GPU-accelerated tree learning now. Please read our [GPU Tutorial](./docs/GPU-Tutorial.rst) and [Performance Comparison](./docs/GPU-Performance.rst).

02/20/2017 : Update to LightGBM v2.

02/12/2017 : LightGBM v1 stable release.

01/08/2017 : Release [**R-package**](https://github.com/microsoft/LightGBM/tree/master/R-package) beta version, welcome to have a try and provide feedback.

12/05/2016 : **Categorical Features as input directly** (without one-hot coding). 

12/02/2016 : Release [**Python-package**](https://github.com/microsoft/LightGBM/tree/master/python-package) beta version, welcome to have a try and provide feedback.

More detailed update logs : [Key Events](https://github.com/microsoft/LightGBM/blob/master/docs/Key-Events.md).

External (Unofficial) Repositories
----------------------------------

Julia-package: https://github.com/Allardvm/LightGBM.jl

JPMML (Java PMML converter): https://github.com/jpmml/jpmml-lightgbm

Treelite (model compiler for efficient deployment): https://github.com/dmlc/treelite

leaves (Go model applier): https://github.com/dmitryikh/leaves

ONNXMLTools (ONNX converter): https://github.com/onnx/onnxmltools

SHAP (model output explainer): https://github.com/slundberg/shap

MMLSpark (LightGBM on Spark): https://github.com/Azure/mmlspark

ML.NET (.NET/C#-package): https://github.com/dotnet/machinelearning

LightGBM.NET (.NET/C#-package): https://github.com/rca22/LightGBM.Net

Dask-LightGBM (distributed and parallel Python-package): https://github.com/dask/dask-lightgbm

Ruby gem: https://github.com/ankane/lightgbm

Support
-------

* Ask a question [on Stack Overflow with the `lightgbm` tag](https://stackoverflow.com/questions/ask?tags=lightgbm), we monitor this for new questions.
* Discuss on the [LightGBM Gitter](https://gitter.im/Microsoft/LightGBM).
* Discuss on the [LightGBM Slack team](https://lightgbm.slack.com).
  * Use [this invite link](https://lightgbm-slack-autojoin.herokuapp.com/) to join the team.
* Open **bug reports** and **feature requests** (not questions) on [GitHub issues](https://github.com/microsoft/LightGBM/issues).

How to Contribute
-----------------

LightGBM has been developed and used by many active community members. Your help is very valuable to make it better for everyone.

- Contribute to the [tests](https://github.com/microsoft/LightGBM/tree/master/tests) to make it more reliable.
- Contribute to the [documentation](https://github.com/microsoft/LightGBM/tree/master/docs) to make it clearer for everyone.
- Contribute to the [examples](https://github.com/microsoft/LightGBM/tree/master/examples) to share your experience with other users.
- Look for [issues with tag "help wanted"](https://github.com/microsoft/LightGBM/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) and submit pull requests to address them.
- Add your stories and experience to [Awesome LightGBM](https://github.com/microsoft/LightGBM/blob/master/examples/README.md). If LightGBM helped you in a machine learning competition or some research application, we want to hear about it!
- [Open an issue](https://github.com/microsoft/LightGBM/issues) to report problems or recommend new features.

Microsoft Open Source Code of Conduct
-------------------------------------

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

Reference Papers
----------------

Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu. "[LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)". Advances in Neural Information Processing Systems 30 (NIPS 2017), pp. 3149-3157.

Qi Meng, Guolin Ke, Taifeng Wang, Wei Chen, Qiwei Ye, Zhi-Ming Ma, Tie-Yan Liu. "[A Communication-Efficient Parallel Algorithm for Decision Tree](http://papers.nips.cc/paper/6380-a-communication-efficient-parallel-algorithm-for-decision-tree)". Advances in Neural Information Processing Systems 29 (NIPS 2016), pp. 1279-1287.

Huan Zhang, Si Si and Cho-Jui Hsieh. "[GPU Acceleration for Large-scale Tree Boosting](https://arxiv.org/abs/1706.08359)". SysML Conference, 2018.

**Note**: If you use LightGBM in your GitHub projects, please add `lightgbm` in the `requirements.txt`.

License
-------

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/microsoft/LightGBM/blob/master/LICENSE) for additional details.
