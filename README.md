<img src=https://github.com/microsoft/LightGBM/blob/master/docs/logo/LightGBM_logo_black_text.svg width=300 />

Light Gradient Boosting Machine
===============================

[![Python-package GitHub Actions Build Status](https://github.com/microsoft/LightGBM/workflows/Python-package/badge.svg?branch=master)](https://github.com/microsoft/LightGBM/actions)
[![R-package GitHub Actions Build Status](https://github.com/microsoft/LightGBM/workflows/R-package/badge.svg?branch=master)](https://github.com/microsoft/LightGBM/actions)
[![CUDA Version GitHub Actions Build Status](https://github.com/microsoft/LightGBM/workflows/CUDA%20Version/badge.svg?branch=master)](https://github.com/microsoft/LightGBM/actions)
[![Static Analysis GitHub Actions Build Status](https://github.com/microsoft/LightGBM/workflows/Static%20Analysis/badge.svg?branch=master)](https://github.com/microsoft/LightGBM/actions)
[![Azure Pipelines Build Status](https://lightgbm-ci.visualstudio.com/lightgbm-ci/_apis/build/status/Microsoft.LightGBM?branchName=master)](https://lightgbm-ci.visualstudio.com/lightgbm-ci/_build/latest?definitionId=1)
[![Appveyor Build Status](https://ci.appveyor.com/api/projects/status/1ys5ot401m0fep6l/branch/master?svg=true)](https://ci.appveyor.com/project/guolinke/lightgbm/branch/master)
[![Documentation Status](https://readthedocs.org/projects/lightgbm/badge/?version=latest)](https://lightgbm.readthedocs.io/)
[![Link checks](https://github.com/microsoft/LightGBM/workflows/Link%20checks/badge.svg)](https://github.com/microsoft/LightGBM/actions?query=workflow%3A%22Link+checks%22)
[![License](https://img.shields.io/github/license/microsoft/lightgbm.svg)](https://github.com/microsoft/LightGBM/blob/master/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/lightgbm.svg?logo=python&logoColor=white)](https://pypi.org/project/lightgbm)
[![PyPI Version](https://img.shields.io/pypi/v/lightgbm.svg?logo=pypi&logoColor=white)](https://pypi.org/project/lightgbm)
[![CRAN Version](https://www.r-pkg.org/badges/version/lightgbm)](https://cran.r-project.org/package=lightgbm)

LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages:

- Faster training speed and higher efficiency.
- Lower memory usage.
- Better accuracy.
- Support of parallel, distributed, and GPU learning.
- Capable of handling large-scale data.

For further details, please refer to [Features](https://github.com/microsoft/LightGBM/blob/master/docs/Features.rst).

Benefiting from these advantages, LightGBM is being widely-used in many [winning solutions](https://github.com/microsoft/LightGBM/blob/master/examples/README.md#machine-learning-challenge-winning-solutions) of machine learning competitions.

[Comparison experiments](https://github.com/microsoft/LightGBM/blob/master/docs/Experiments.rst#comparison-experiment) on public datasets show that LightGBM can outperform existing boosting frameworks on both efficiency and accuracy, with significantly lower memory consumption. What's more, [distributed learning experiments](https://github.com/microsoft/LightGBM/blob/master/docs/Experiments.rst#parallel-experiment) show that LightGBM can achieve a linear speed-up by using multiple machines for training in specific settings.

Get Started and Documentation
-----------------------------

Our primary documentation is at https://lightgbm.readthedocs.io/ and is generated from this repository. If you are new to LightGBM, follow [the installation instructions](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html) on that site.

Next you may want to read:

- [**Examples**](https://github.com/microsoft/LightGBM/tree/master/examples) showing command line usage of common tasks.
- [**Features**](https://github.com/microsoft/LightGBM/blob/master/docs/Features.rst) and algorithms supported by LightGBM.
- [**Parameters**](https://github.com/microsoft/LightGBM/blob/master/docs/Parameters.rst) is an exhaustive list of customization you can make.
- [**Distributed Learning**](https://github.com/microsoft/LightGBM/blob/master/docs/Parallel-Learning-Guide.rst) and [**GPU Learning**](https://github.com/microsoft/LightGBM/blob/master/docs/GPU-Tutorial.rst) can speed up computation.
- [**Laurae++ interactive documentation**](https://sites.google.com/view/lauraepp/parameters) is a detailed guide for hyperparameters.
- [**FLAML**](https://www.microsoft.com/en-us/research/project/fast-and-lightweight-automl-for-large-scale-data/articles/flaml-a-fast-and-lightweight-automl-library/) provides automated tuning for LightGBM ([code examples](https://github.com/microsoft/FLAML/blob/main/notebook/flaml_lightgbm.ipynb)).
- [**Optuna Hyperparameter Tuner**](https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258) provides automated tuning for LightGBM hyperparameters ([code examples](https://github.com/optuna/optuna/tree/master/examples/lightgbm)).

Documentation for contributors:

- [**How we update readthedocs.io**](https://github.com/microsoft/LightGBM/blob/master/docs/README.rst).
- Check out the [**Development Guide**](https://github.com/microsoft/LightGBM/blob/master/docs/Development-Guide.rst).

News
----

Please refer to changelogs at [GitHub releases](https://github.com/microsoft/LightGBM/releases) page.

Some old update logs are available at [Key Events](https://github.com/microsoft/LightGBM/blob/master/docs/Key-Events.md) page.

External (Unofficial) Repositories
----------------------------------

FLAML (AutoML library for hyperparameter optimization): https://github.com/microsoft/FLAML

Optuna (hyperparameter optimization framework): https://github.com/optuna/optuna

Julia-package: https://github.com/IQVIA-ML/LightGBM.jl

JPMML (Java PMML converter): https://github.com/jpmml/jpmml-lightgbm

Treelite (model compiler for efficient deployment): https://github.com/dmlc/treelite

Hummingbird (model compiler into tensor computations): https://github.com/microsoft/hummingbird

cuML Forest Inference Library (GPU-accelerated inference): https://github.com/rapidsai/cuml

daal4py (Intel CPU-accelerated inference): https://github.com/IntelPython/daal4py

m2cgen (model appliers for various languages): https://github.com/BayesWitnesses/m2cgen

leaves (Go model applier): https://github.com/dmitryikh/leaves

ONNXMLTools (ONNX converter): https://github.com/onnx/onnxmltools

SHAP (model output explainer): https://github.com/slundberg/shap

dtreeviz (decision tree visualization and model interpretation): https://github.com/parrt/dtreeviz

MMLSpark (LightGBM on Spark): https://github.com/Azure/mmlspark

Kubeflow Fairing (LightGBM on Kubernetes): https://github.com/kubeflow/fairing

Kubeflow Operator (LightGBM on Kubernetes): https://github.com/kubeflow/xgboost-operator

ML.NET (.NET/C#-package): https://github.com/dotnet/machinelearning

LightGBM.NET (.NET/C#-package): https://github.com/rca22/LightGBM.Net

Ruby gem: https://github.com/ankane/lightgbm

LightGBM4j (Java high-level binding): https://github.com/metarank/lightgbm4j

lightgbm-rs (Rust binding): https://github.com/vaaaaanquish/lightgbm-rs

MLflow (experiment tracking, model monitoring framework): https://github.com/mlflow/mlflow

`{treesnip}` (R `{parsnip}`-compliant interface): https://github.com/curso-r/treesnip

`{mlr3learners.lightgbm}` (R `{mlr3}`-compliant interface): https://github.com/mlr3learners/mlr3learners.lightgbm

Support
-------

- Ask a question [on Stack Overflow with the `lightgbm` tag](https://stackoverflow.com/questions/ask?tags=lightgbm), we monitor this for new questions.
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
