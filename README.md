LightGBM, Light Gradient Boosting Machine
=========================================

[![GitHub Actions Build Status](https://github.com/microsoft/LightGBM/workflows/GitHub%20Actions/badge.svg?branch=master)](https://github.com/microsoft/LightGBM/actions)
[![Azure Pipelines Build Status](https://lightgbm-ci.visualstudio.com/lightgbm-ci/_apis/build/status/Microsoft.LightGBM?branchName=master)](https://lightgbm-ci.visualstudio.com/lightgbm-ci/_build/latest?definitionId=1)
[![Appveyor Build Status](https://ci.appveyor.com/api/projects/status/1ys5ot401m0fep6l/branch/master?svg=true)](https://ci.appveyor.com/project/guolinke/lightgbm/branch/master)
[![Travis Build Status](https://travis-ci.org/microsoft/LightGBM.svg?branch=master)](https://travis-ci.org/microsoft/LightGBM)
[![Documentation Status](https://readthedocs.org/projects/lightgbm/badge/?version=latest)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/github/license/microsoft/lightgbm.svg)](https://github.com/microsoft/LightGBM/blob/master/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/lightgbm.svg?logo=python&logoColor=white)](https://pypi.org/project/lightgbm)
[![PyPI Version](https://img.shields.io/pypi/v/lightgbm.svg?logo=pypi&logoColor=white)](https://pypi.org/project/lightgbm)
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

- [**Examples**](https://github.com/microsoft/LightGBM/tree/master/examples) showing command line usage of common tasks.
- [**Features**](https://github.com/microsoft/LightGBM/blob/master/docs/Features.rst) and algorithms supported by LightGBM.
- [**Parameters**](https://github.com/microsoft/LightGBM/blob/master/docs/Parameters.rst) is an exhaustive list of customization you can make.
- [**Parallel Learning**](https://github.com/microsoft/LightGBM/blob/master/docs/Parallel-Learning-Guide.rst) and [**GPU Learning**](https://github.com/microsoft/LightGBM/blob/master/docs/GPU-Tutorial.rst) can speed up computation.
- [**Laurae++ interactive documentation**](https://sites.google.com/view/lauraepp/parameters) is a detailed guide for hyperparameters.
- [**Optuna Hyperparameter Tuner**](https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258) provides automated tuning for LightGBM hyperparameters.

Documentation for contributors:

- [**How we update readthedocs.io**](https://github.com/microsoft/LightGBM/blob/master/docs/README.rst).
- Check out the [**Development Guide**](https://github.com/microsoft/LightGBM/blob/master/docs/Development-Guide.rst).

News
----

Please refer to changelogs at [GitHub releases](https://github.com/microsoft/LightGBM/releases) page.

Some old update logs are available at [Key Events](https://github.com/microsoft/LightGBM/blob/master/docs/Key-Events.md) page.

External (Unofficial) Repositories
----------------------------------

Optuna (hyperparameter optimization framework): https://github.com/optuna/optuna

Julia-package: https://github.com/IQVIA-ML/LightGBM.jl

JPMML (Java PMML converter): https://github.com/jpmml/jpmml-lightgbm

Treelite (model compiler for efficient deployment): https://github.com/dmlc/treelite

cuML Forest Inference Library (GPU-accelerated inference): https://github.com/rapidsai/cuml

m2cgen (model appliers for various languages): https://github.com/BayesWitnesses/m2cgen

leaves (Go model applier): https://github.com/dmitryikh/leaves

ONNXMLTools (ONNX converter): https://github.com/onnx/onnxmltools

SHAP (model output explainer): https://github.com/slundberg/shap

MMLSpark (LightGBM on Spark): https://github.com/Azure/mmlspark

Kubeflow Fairing (LightGBM on Kubernetes): https://github.com/kubeflow/fairing

ML.NET (.NET/C#-package): https://github.com/dotnet/machinelearning

LightGBM.NET (.NET/C#-package): https://github.com/rca22/LightGBM.Net

Dask-LightGBM (distributed and parallel Python-package): https://github.com/dask/dask-lightgbm

Ruby gem: https://github.com/ankane/lightgbm

Support
-------

- Ask a question [on Stack Overflow with the `lightgbm` tag](https://stackoverflow.com/questions/ask?tags=lightgbm), we monitor this for new questions.
- Discuss on the [LightGBM Gitter](https://gitter.im/Microsoft/LightGBM).
- Discuss on the [LightGBM Slack team](https://lightgbm.slack.com).
  - Use [this invite link](https://lightgbm-slack-autojoin.herokuapp.com/) to join the team.
- Open **bug reports** and **feature requests** (not questions) on [GitHub issues](https://github.com/microsoft/LightGBM/issues).

How to Contribute
-----------------

Check [CONTRIBUTING](https://github.com/microsoft/LightGBM/blob/master/CONTRIBUTING.md) page.

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
