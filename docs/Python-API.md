##Catalog

* [Data Structure API](Python-API.md#basic-data-structure-api)
    - [Dataset](Python-API.md#dataset)
    - [Booster](Python-API.md#booster)

* [Training API](Python-API.md#training-api)
    - [train](Python-API.md#trainparams-train_set-num_boost_round100-valid_setsnone-valid_namesnone-fobjnone-fevalnone-init_modelnone-feature_nameauto-categorical_featureauto-early_stopping_roundsnone-evals_resultnone-verbose_evaltrue-learning_ratesnone-callbacksnone)
    - [cv](Python-API.md#cvparams-train_set-num_boost_round10-nfold5-stratifiedfalse-shuffletrue-metricsnone-fobjnone-fevalnone-init_modelnone-feature_nameauto-categorical_featureauto-early_stopping_roundsnone-fpreprocnone-verbose_evalnone-show_stdvtrue-seed0-callbacksnone)

* [Scikit-learn API](Python-API.md#scikit-learn-api)
    - [Common Methods](Python-API.md#common-methods)
    - [Common Attributes](Python-API.md#common-attributes)
    - [LGBMClassifier](Python-API.md#lgbmclassifier)
    - [LGBMRegressor](Python-API.md#lgbmregressor)
    - [LGBMRanker](Python-API.md#lgbmranker)

* [Callbacks](Python-API.md#callbacks)
    - [Before iteration](Python-API.md#before-iteration)
        + [reset_parameter](Python-API.md#reset_parameterkwargs)
    - [After iteration](Python-API.md#after-iteration)
        + [print_evaluation](Python-API.md#print_evaluationperiod1-show_stdvtrue)
        + [record_evaluation](Python-API.md#record_evaluationeval_result)
        + [early_stopping](Python-API.md#early_stoppingstopping_rounds-verbosetrue)

* [Plotting](Python-API.md#plotting)

The methods of each Class is in alphabetical order.

----

##Basic Data Structure API

###Dataset

####__init__(data, label=None, max_bin=255, reference=None, weight=None, group=None, silent=False, feature_name='auto', categorical_feature='auto', params=None, free_raw_data=True)

    Parameters
    ----------
    data : str/numpy array/scipy.sparse
        Data source of Dataset.
        When data type is string, it represents the path of txt file
    label : list or numpy 1-D array, optional
        Label of the data
    max_bin : int, required
        Max number of discrete bin for features
    reference : Other Dataset, optional
        If this dataset validation, need to use training data as reference
    weight : list or numpy 1-D array, optional
        Weight for each instance.
    group : list or numpy 1-D array, optional
        Group/query size for dataset
    silent : boolean, optional
        Whether print messages during construction
    feature_name : list of str, or 'auto'
        Feature names
        If 'auto' and data is pandas DataFrame, use data columns name
    categorical_feature : list of str or int, or 'auto'
        Categorical features,
        type int represents index,
        type str represents feature names (need to specify feature_name as well)
        If 'auto' and data is pandas DataFrame, use pandas categorical columns
    params : dict, optional
        Other parameters
    free_raw_data : Bool
        True if need to free raw data after construct inner dataset


####create_valid(data, label=None, weight=None, group=None, silent=False, params=None)

    Create validation data align with current dataset.

    Parameters
    ----------
    data : str/numpy array/scipy.sparse
        Data source of _InnerDataset.
        When data type is string, it represents the path of txt file
    label : list or numpy 1-D array, optional
        Label of the training data.
    weight : list or numpy 1-D array, optional
        Weight for each instance.
    group : list or numpy 1-D array, optional
        Group/query size for dataset
    silent : boolean, optional
        Whether print messages during construction
    params : dict, optional
        Other parameters


####get_group()

    Get the initial score of the Dataset.

    Returns
    -------
    init_score : array


####get_init_score()

    Get the initial score of the Dataset.

    Returns
    -------
    init_score : array


####get_label()

    Get the label of the Dataset.

    Returns
    -------
    label : array


####get_weight()

    Get the weight of the Dataset.

    Returns
    -------
    weight : array


####num_data()

    Get the number of rows in the Dataset.

    Returns
    -------
    number of rows : int


####num_feature()

    Get the number of columns (features) in the Dataset.

    Returns
    -------
    number of columns : int


####save_binary(filename)

    Save Dataset to binary file.

    Parameters
    ----------
    filename : str
        Name of the output file.


####set_categorical_feature(categorical_feature)

    Set categorical features.

    Parameters
    ----------
    categorical_feature : list of str or list of int
        Name (str) or index (int) of categorical features



####set_feature_name(feature_name)

    Set feature name.

    Parameters
    ----------
    feature_name : list of str
        Feature names


####set_group(group)

    Set group size of Dataset (used for ranking).

    Parameters
    ----------
    group : numpy array or list or None
        Group size of each group


####set_init_score(init_score)

    Set init score of booster to start from.

    Parameters
    ----------
    init_score : numpy array or list or None
        Init score for booster


####set_label(label)

    Set label of Dataset.

    Parameters
    ----------
    label : numpy array or list or None
        The label information to be set into Dataset


####set_reference(reference)

    Set reference dataset.

    Parameters
    ----------
    reference : Dataset
        Will use reference as template to consturct current dataset


####set_weight(weight)

    Set weight of each instance.

    Parameters
    ----------
    weight : numpy array or list or None
        Weight for each data point


####subset(used_indices, params=None)

    Get subset of current dataset.

    Parameters
    ----------
    used_indices : list of int
        Used indices of this subset
    params : dict
        Other parameters


###Booster

####__init__(params=None, train_set=None, model_file=None, silent=False)

    Initialize the Booster.

    Parameters
    ----------
    params : dict
        Parameters for boosters.
    train_set : Dataset
        Training dataset
    model_file : str
        Path to the model file.
    silent : boolean, optional
        Whether print messages during construction


####add_valid(data, name)

    Add an validation data.

    Parameters
    ----------
    data : Dataset
        Validation data
    name : str
        Name of validation data


####attr(key)

    Get attribute string from the Booster.

    Parameters
    ----------
    key : str
        The key to get attribute from.

    Returns
    -------
    value : str
        The attribute value of the key, returns None if attribute do not exist.


####current_iteration()

    Get current number of iterations.

    Returns
    -------
    result : int
        Current number of iterations

####dump_model()

    Dump model to json format.

    Returns
    -------
    result : dict or list
        Json format of model


####eval(data, name, feval=None)

    Evaluate for data.

    Parameters
    ----------
    data : _InnerDataset object
    name :
        Name of data
    feval : function
        Custom evaluation function.
    Returns
    -------
    result : list
        Evaluation result list.


####eval_train(feval=None)

    Evaluate for training data.

    Parameters
    ----------
    feval : function
        Custom evaluation function.

    Returns
    -------
    result: str
        Evaluation result list.


####eval_valid(feval=None)

    Evaluate for validation data.

    Parameters
    ----------
    feval : function
        Custom evaluation function.

    Returns
    -------
    result : str
        Evaluation result list.


####feature_importance(importance_type="split")

    Feature importances.

    Parameters
    ----------
    importance_type : str, default "split"
    How the importance is calculated: "split" or "gain"
    "split" is the number of times a feature is used in a model
    "gain" is the total gain of splits which use the feature

    Returns
    -------
    result : array
        Array of feature importances


####predict(data, num_iteration=-1, raw_score=False, pred_leaf=False, data_has_header=False, is_reshape=True)

    Predict logic.

    Parameters
    ----------
    data : str/numpy array/scipy.sparse
        Data source for prediction
        When data type is string, it represents the path of txt file
    num_iteration : int
        Used iteration for prediction
    raw_score : bool
        True for predict raw score
    pred_leaf : bool
        True for predict leaf index
    data_has_header : bool
        Used for txt data
    is_reshape : bool
        Reshape to (nrow, ncol) if true

    Returns
    -------
    Prediction result


####reset_parameter(params)

    Reset parameters for booster.

    Parameters
    ----------
    params : dict
        New parameters for boosters
    silent : boolean, optional
        Whether print messages during construction


####rollback_one_iter()

    Rollback one iteration.


####save_model(filename, num_iteration=-1)

    Save model of booster to file.

    Parameters
    ----------
    filename : str
        Filename to save
    num_iteration : int
        Number of iteration that want to save. < 0 means save all


####set_attr(**kwargs)

    Set the attribute of the Booster.

    Parameters
    ----------
    **kwargs
        The attributes to set. Setting a value to None deletes an attribute.


####set_train_data_name(name)

    Set training data name.

    Parameters
    ----------
    name : str
        Name of training data.

####update(train_set=None, fobj=None)

    Update for one iteration.
    Note: for multi-class task, the score is group by class_id first, then group by row_id
          if you want to get i-th row score in j-th class, the access way is score[j*num_data+i]
          and you should group grad and hess in this way as well.

    Parameters
    ----------
    train_set :
        Training data, None means use last training data
    fobj : function
        Customized objective function.

    Returns
    -------
    is_finished, bool


##Training API

####train(params, train_set, num_boost_round=100, valid_sets=None, valid_names=None, fobj=None, feval=None, init_model=None, feature_name='auto', categorical_feature='auto', early_stopping_rounds=None, evals_result=None, verbose_eval=True, learning_rates=None, callbacks=None)

    Train with given parameters.

    Parameters
    ----------
    params : dict
        Parameters for training.
    train_set : Dataset
        Data to be trained.
    num_boost_round: int
        Number of boosting iterations.
    valid_sets: list of Datasets
        List of data to be evaluated during training
    valid_names: list of str
        Names of valid_sets
    fobj : function
        Customized objective function.
    feval : function
        Customized evaluation function.
        Note: should return (eval_name, eval_result, is_higher_better) of list of this
    init_model : file name of lightgbm model or 'Booster' instance
        model used for continued train
    feature_name : list of str, or 'auto'
        Feature names
        If 'auto' and data is pandas DataFrame, use data columns name
    categorical_feature : list of str or int, or 'auto'
        Categorical features,
        type int represents index,
        type str represents feature names (need to specify feature_name as well)
        If 'auto' and data is pandas DataFrame, use pandas categorical columns
    early_stopping_rounds: int
        Activates early stopping.
        Requires at least one validation data and one metric
        If there's more than one, will check all of them
        Returns the model with (best_iter + early_stopping_rounds)
        If early stopping occurs, the model will add 'best_iteration' field
    evals_result: dict or None
        This dictionary used to store all evaluation results of all the items in valid_sets.
        Example: with a valid_sets containing [valid_set, train_set]
                 and valid_names containing ['eval', 'train']
                 and a paramater containing ('metric':'logloss')
        Returns: {'train': {'logloss': ['0.48253', '0.35953', ...]},
                  'eval': {'logloss': ['0.480385', '0.357756', ...]}}
        passed with None means no using this function
    verbose_eval : bool or int
        Requires at least one item in evals.
        If `verbose_eval` is True,
            the eval metric on the valid set is printed at each boosting stage.
        If `verbose_eval` is int,
            the eval metric on the valid set is printed at every `verbose_eval` boosting stage.
        The last boosting stage
            or the boosting stage found by using `early_stopping_rounds` is also printed.
        Example: with verbose_eval=4 and at least one item in evals,
            an evaluation metric is printed every 4 (instead of 1) boosting stages.
    learning_rates : list or function
        List of learning rate for each boosting round
        or a customized function that calculates learning_rate
        in terms of current number of round (e.g. yields learning rate decay)
        - list l: learning_rate = l[current_round]
        - function f: learning_rate = f(current_round)
    callbacks : list of callback functions
        List of callback functions that are applied at each iteration.
        See Callbacks in Python-API.md for more information.

    Returns
    -------
    booster : a trained booster model


####cv(params, train_set, num_boost_round=10, nfold=5, stratified=False, shuffle=True, metrics=None, fobj=None, feval=None, init_model=None, feature_name='auto', categorical_feature='auto', early_stopping_rounds=None, fpreproc=None, verbose_eval=None, show_stdv=True, seed=0, callbacks=None)

    Cross-validation with given paramaters.

    Parameters
    ----------
    params : dict
        Booster params.
    train_set : Dataset
        Data to be trained.
    num_boost_round : int
        Number of boosting iterations.
    nfold : int
        Number of folds in CV.
    stratified : bool
        Perform stratified sampling.
    shuffle: bool
        Whether shuffle before split data.
    folds : a KFold or StratifiedKFold instance
        Sklearn KFolds or StratifiedKFolds.
    metrics : str or list of str
        Evaluation metrics to be watched in CV.
    fobj : function
        Custom objective function.
    feval : function
        Custom evaluation function.
    init_model : file name of lightgbm model or 'Booster' instance
        model used for continued train
    feature_name : list of str, or 'auto'
        Feature names
        If 'auto' and data is pandas DataFrame, use data columns name
    categorical_feature : list of str or int, or 'auto'
        Categorical features,
        type int represents index,
        type str represents feature names (need to specify feature_name as well)
        If 'auto' and data is pandas DataFrame, use pandas categorical columns
    early_stopping_rounds: int
        Activates early stopping. CV error needs to decrease at least
        every <early_stopping_rounds> round(s) to continue.
        Last entry in evaluation history is the one from best iteration.
    fpreproc : function
        Preprocessing function that takes (dtrain, dtest, param)
        and returns transformed versions of those.
    verbose_eval : bool, int, or None, default None
        Whether to display the progress.
        If None, progress will be displayed when np.ndarray is returned.
        If True, progress will be displayed at boosting stage.
        If an integer is given,
            progress will be displayed at every given `verbose_eval` boosting stage.
    show_stdv : bool, default True
        Whether to display the standard deviation in progress.
        Results are not affected, and always contains std.
    seed : int
        Seed used to generate the folds (passed to numpy.random.seed).
    callbacks : list of callback functions
        List of callback functions that are applied at end of each iteration.

    Returns
    -------
    evaluation history : list of str


##Scikit-learn API

###Common Methods

####__init__(boosting_type="gbdt", num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=10, max_bin=255, subsample_for_bin=50000, objective="regression", min_split_gain=0, min_child_weight=5, min_child_samples=10, subsample=1, subsample_freq=1, colsample_bytree=1, reg_alpha=0, reg_lambda=0, scale_pos_weight=1, is_unbalance=False, seed=0, nthread=-1, silent=True, sigmoid=1.0, huber_delta=1.0, max_position=20, label_gain=None, drop_rate=0.1, skip_drop=0.5, max_drop=50, uniform_drop=False, xgboost_dart_mode=False)

    Implementation of the Scikit-Learn API for LightGBM.

    Parameters
    ----------
    boosting_type : str
        gbdt, traditional Gradient Boosting Decision Tree
        dart, Dropouts meet Multiple Additive Regression Trees
    num_leaves : int
        Maximum tree leaves for base learners.
    max_depth : int
        Maximum tree depth for base learners, -1 means no limit.
    learning_rate : float
        Boosting learning rate
    n_estimators : int
        Number of boosted trees to fit.
    max_bin : int
        Number of bucketed bin for feature values
    subsample_for_bin : int
        Number of samples for constructing bins.
    objective : str or callable
        Specify the learning task and the corresponding learning objective or
        a custom objective function to be used (see note below).
        default: binary for LGBMClassifier, regression for LGBMRegressor, lambdarank for LGBMRanker
    min_split_gain : float
        Minimum loss reduction required to make a further partition on a leaf node of the tree.
    min_child_weight : int
        Minimum sum of instance weight(hessian) needed in a child(leaf)
    min_child_samples : int
        Minimum number of data need in a child(leaf)
    subsample : float
        Subsample ratio of the training instance.
    subsample_freq : int
        frequence of subsample, <=0 means no enable
    colsample_bytree : float
        Subsample ratio of columns when constructing each tree.
    reg_alpha : float
        L1 regularization term on weights
    reg_lambda : float
        L2 regularization term on weights
    scale_pos_weight : float
        Balancing of positive and negative weights.
    is_unbalance : bool
        Is unbalance for binary classification
    seed : int
        Random number seed.
    nthread : int
        Number of parallel threads
    silent : boolean
        Whether to print messages while running boosting.
    sigmoid : float
        Only used in binary classification and lambdarank. Parameter for sigmoid function.
    huber_delta : float
        Only used in regression. Parameter for Huber loss function.
    gaussian_eta : float
        Only used in regression. Parameter for L1 and Huber loss function.
        It is used to control the width of Gaussian function to approximate hessian.
    fair_c : float
        Only used in regression. Parameter for Fair loss function.
    max_position : int
        Only used in lambdarank, will optimize NDCG at this position.
    label_gain : list of float
        Only used in lambdarank, relevant gain for labels.
        For example, the gain of label 2 is 3 if using default label gains.
        None (default) means use default value of CLI version: {0,1,3,7,15,31,63,...}.
    drop_rate : float
        Only used when boosting_type='dart'. Probablity to select dropping trees.
    skip_drop : float
        Only used when boosting_type='dart'. Probablity to skip dropping trees.
    max_drop : int
        Only used when boosting_type='dart'. Max number of dropped trees in one iteration.
    uniform_drop : bool
        Only used when boosting_type='dart'. If true, drop trees uniformly, else drop according to weights.
    xgboost_dart_mode : bool
        Only used when boosting_type='dart'. Whether use xgboost dart mode.

    Note
    ----
    A custom objective function can be provided for the ``objective``
    parameter. In this case, it should have the signature
    ``objective(y_true, y_pred) -> grad, hess``
        or ``objective(y_true, y_pred, group) -> grad, hess``:

        y_true: array_like of shape [n_samples]
            The target values
        y_pred: array_like of shape [n_samples] or shape[n_samples * n_class]
            The predicted values
        group: array_like
            group/query data, used for ranking task
        grad: array_like of shape [n_samples] or shape[n_samples * n_class]
            The value of the gradient for each sample point.
        hess: array_like of shape [n_samples] or shape[n_samples * n_class]
            The value of the second derivative for each sample point

    for multi-class task, the y_pred is group by class_id first, then group by row_id
        if you want to get i-th row y_pred in j-th class, the access way is y_pred[j*num_data+i]
        and you should group grad and hess in this way as well


####apply(X, num_iteration=0)

    Return the predicted leaf every tree for each sample.

    Parameters
    ----------
    X : array_like, shape=[n_samples, n_features]
        Input features matrix.

    num_iteration : int
        Limit number of iterations in the prediction; defaults to 0 (use all trees).

    Returns
    -------
    X_leaves : array_like, shape=[n_samples, n_trees]


####fit(X, y, sample_weight=None, init_score=None, group=None, eval_set=None, eval_sample_weight=None, eval_init_score=None, eval_group=None, eval_metric=None, early_stopping_rounds=None, verbose=True, feature_name='auto', categorical_feature='auto', callbacks=None)

    Fit the gradient boosting model.

    Parameters
    ----------
    X : array_like
        Feature matrix
    y : array_like
        Labels
    sample_weight : array_like
        weight of training data
    init_score : array_like
        init score of training data
    group : array_like
        group data of training data
    eval_set : list, optional
        A list of (X, y) tuple pairs to use as a validation set for early-stopping
    eval_sample_weight : list or dict of array
        weight of eval data; if you use dict, the index should start from 0
    eval_init_score : list or dict of array
        init score of eval data; if you use dict, the index should start from 0
    eval_group : list or dict of array
        group data of eval data; if you use dict, the index should start from 0
    eval_metric : str, list of str, callable, optional
        If a str, should be a built-in evaluation metric to use.
        If callable, a custom evaluation metric, see note for more details.
        default: binary_error for LGBMClassifier, l2 for LGBMRegressor, ndcg for LGBMRanker
    early_stopping_rounds : int
    verbose : bool
        If `verbose` and an evaluation set is used, writes the evaluation
    feature_name : list of str, or 'auto'
        Feature names
        If 'auto' and data is pandas DataFrame, use data columns name
    categorical_feature : list of str or int, or 'auto'
        Categorical features,
        type int represents index,
        type str represents feature names (need to specify feature_name as well)
        If 'auto' and data is pandas DataFrame, use pandas categorical columns
    callbacks : list of callback functions
        List of callback functions that are applied at each iteration.
        See Callbacks in Python-API.md for more information.

    Note
    ----
    Custom eval function expects a callable with following functions:
        ``func(y_true, y_pred)``, ``func(y_true, y_pred, weight)``
            or ``func(y_true, y_pred, weight, group)``.
        return (eval_name, eval_result, is_bigger_better)
            or list of (eval_name, eval_result, is_bigger_better)

        y_true: array_like of shape [n_samples]
            The target values
        y_pred: array_like of shape [n_samples] or shape[n_samples * n_class] (for multi-class)
            The predicted values
        weight: array_like of shape [n_samples]
            The weight of samples
        group: array_like
            group/query data, used for ranking task
        eval_name: str
            name of evaluation
        eval_result: float
            eval result
        is_bigger_better: bool
            is eval result bigger better, e.g. AUC is bigger_better.
    for multi-class task, the y_pred is group by class_id first, then group by row_id
      if you want to get i-th row y_pred in j-th class, the access way is y_pred[j*num_data+i]


####predict(X, raw_score=False, num_iteration=0)

    Return the predicted value for each sample.

    Parameters
    ----------
    X : array_like, shape=[n_samples, n_features]
        Input features matrix.

    num_iteration : int
        Limit number of iterations in the prediction; defaults to 0 (use all trees).

    Returns
    -------
    predicted_result : array_like, shape=[n_samples] or [n_samples, n_classes]


###Common Attributes

####booster_

    Get the underlying lightgbm Booster of this model.

####evals_result_

    Get the evaluation results.

####feature_importances_

    Get normailized feature importances.


###LGBMClassifier

####predict_proba(X, raw_score=False, num_iteration=0)

    Return the predicted probability for each class for each sample.

    Parameters
    ----------
    X : array_like, shape=[n_samples, n_features]
        Input features matrix.

    num_iteration : int
        Limit number of iterations in the prediction; defaults to 0 (use all trees).

    Returns
    -------
    predicted_probability : array_like, shape=[n_samples, n_classes]

####classes_

    Get class label array.

####n_classes_

    Get number of classes.


###LGBMRegressor

###LGBMRanker

####fit(X, y, sample_weight=None, init_score=None, group=None, eval_set=None, eval_sample_weight=None, eval_init_score=None, eval_group=None, eval_metric='ndcg', eval_at=1, early_stopping_rounds=None, verbose=True, feature_name='auto', categorical_feature='auto', callbacks=None)

    Most arguments are same as Common Methods except:

    eval_at : int or list of int, default=1
        The evaulation positions of NDCG

##Callbacks

###Before iteration

####reset_parameter(**kwargs)

    Reset parameter after first iteration

    NOTE: the initial parameter will still take in-effect on first iteration.

    Parameters
    ----------
    **kwargs: value should be list or function
        List of parameters for each boosting round
        or a customized function that calculates learning_rate in terms of
        current number of round (e.g. yields learning rate decay)
        - list l: parameter = l[current_round]
        - function f: parameter = f(current_round)
    Returns
    -------
    callback : function
        The requested callback function.

###After iteration

####print_evaluation(period=1, show_stdv=True)

    Create a callback that print evaluation result.
    (Same function as `verbose_eval` in lightgbm.train())

    Parameters
    ----------
    period : int
        The period to log the evaluation results

    show_stdv : bool, optional
        Whether show standard deviation if provided

    Returns
    -------
    callback : function
        A callback that prints evaluation every period iterations.

####record_evaluation(eval_result)

    Create a call back that records the evaluation history into eval_result.
    (Same function as `evals_result` in lightgbm.train())

    Parameters
    ----------
    eval_result : dict
       A dictionary to store the evaluation results.

    Returns
    -------
    callback : function
        The requested callback function.

####early_stopping(stopping_rounds, verbose=True)

    Create a callback that activates early stopping.
    To activates early stopping, at least one validation data and one metric is required.
    If there's more than one, all of them will be checked.
    (Same function as `early_stopping_rounds` in lightgbm.train())

    Parameters
    ----------
    stopping_rounds : int
       The stopping rounds before the trend occur.

    verbose : optional, bool
        Whether to print message about early stopping information.

    Returns
    -------
    callback : function
        The requested callback function.

##Plotting

####plot_importance(booster, ax=None, height=0.2, xlim=None, ylim=None, title='Feature importance', xlabel='Feature importance', ylabel='Features', importance_type='split', max_num_features=None, ignore_zero=True, grid=True, **kwargs):

    Plot model feature importances.

    Parameters
    ----------
    booster : Booster, LGBMModel or array
        Booster or LGBMModel instance, or array of feature importances
    ax : matplotlib Axes
        Target axes instance. If None, new figure and axes will be created.
    height : float
        Bar height, passed to ax.barh()
    xlim : tuple
        Tuple passed to axes.xlim()
    ylim : tuple
        Tuple passed to axes.ylim()
    title : str
        Axes title. Pass None to disable.
    xlabel : str
        X axis title label. Pass None to disable.
    ylabel : str
        Y axis title label. Pass None to disable.
    importance_type : str
        How the importance is calculated: "split" or "gain"
        "split" is the number of times a feature is used in a model
        "gain" is the total gain of splits which use the feature
    max_num_features : int
        Max number of top features displayed on plot.
        If None or smaller than 1, all features will be displayed.
    ignore_zero : bool
        Ignore features with zero importance
    grid : bool
        Whether add grid for axes
    **kwargs :
        Other keywords passed to ax.barh()

    Returns
    -------
    ax : matplotlib Axes
