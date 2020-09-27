from multivariate_analysis import *
import pandas as pd
import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Import models
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVR, LinearSVR
import xgboost as xgb

basedir = "C:/Users/ManuelCosta/Documents/GitHub/MLSKG/house-prices-advanced-regression-techniques/"
command = "kaggle competitions submit -c house-prices-advanced-regression-techniques -f submission.csv -m "


def lasso_alpha_tuning(X, y, plot=True):
    """
    Lasso Model's hyperparameter alpha tuning. Uses K-Cross validation (10-fold).
    :param X: X of the current dataset
    :param y: target of the current dataset
    :param plot: either true or false. Controls if plots are shown while training models.
    """
    print("Tuning Lasso Model Alpha")
    alphas = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
    k = 10
    kf = KFold(n_splits=k)
    e = list()
    for alpha in alphas:
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index].values, X.iloc[test_index].values
            y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values
            model = Lasso(alpha=alpha)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            e.append(rmse)
    splits = np.split(np.array(e), len(alphas))
    errors = np.column_stack(splits)
    best_idx = np.argpartition(np.mean(errors, axis=0), -5)[-5:]
    for i in best_idx:
        print(alphas[i])
        # 0.0006
        # 5e-05
        # 0.0002
        # 0.0007
        # 0.0008
        # 0.0001
        # 5e-05
    if plot:
        means = np.mean(errors, axis=0)
        mins = np.min(errors, axis=0)
        f, ax = plt.subplots(figsize=(5, 5))
        ax.xaxis.set_ticks(np.arange(0, 0.1, 0.01))
        d = pd.Series(means, index=alphas)
        fig = sns.lineplot(data=d)
        fig.set_title("Lasso CV scores for different alphas\n (k={})".format(k))
        fig.set_xlabel("alpha")
        fig.set_ylabel("score")
        plt.show()


def train_lasso(X, y, plot=False, alpha=0.01):
    """
    Trains a Lasso Model.
    :param X: X of the current dataset
    :param y: target of the current dataset
    :param plot: either true or false. Controls if plots are shown while training this model.
    :return: trained Lasso model
    """
    estimator = Lasso(alpha=alpha)
    estimated_test_error = estimate_test_error(estimator, X, y)
    print("Estimated test error for Lasso  model (alpha={}): {}".format(alpha, estimated_test_error))
    model = Lasso(alpha=alpha)
    model.fit(X, y)
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print("Training error for Lasso model (alpha={}): {}".format(alpha, rmse))
    if plot:
        plot_residuals(y_pred, y, "Lasso (alpha={})".format(alpha))
    return model


def ridge_alpha_tuning(X, y, plot=False):
    """
    Ridge Model's hyperparameter alpha tuning. Uses K-Cross validation (10-fold).
    :param X: X of the current dataset
    :param y: target of the current dataset
    :param plot: either true or false. Controls if plots are shown while training models.
    """
    print("Tuning Ridge Model Alpha")
    alphas = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
    k = 10
    kf = KFold(n_splits=k)
    e = list()
    for alpha in alphas:
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index].values, X.iloc[test_index].values
            y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            e.append(rmse)
    splits = np.split(np.array(e), len(alphas))
    errors = np.column_stack(splits)
    if plot:
        means = np.mean(errors, axis=0)
        mins = np.min(errors, axis=0)
        f, ax = plt.subplots(figsize=(8, 6))
        ax.xaxis.set_ticks(np.arange(0, 75, 5))
        d = pd.Series(means, index=alphas)
        fig = sns.lineplot(data=d)
        fig.set_title("Ridge CV scores for different alphas\n (k={})".format(k))
        fig.set_xlabel("alpha")
        fig.set_ylabel("score")
        plt.show()


def train_ridge(X, y, plot=False, alpha=1):
    """
    Trains a Ridge Model.
    :param X: X of the current dataset
    :param y: target of the current dataset
    :param plot: either true or false. Controls if plots are shown while training this model.
    :return: trained Ridge model
    """
    estimator = Ridge(alpha=alpha)
    estimated_test_error = estimate_test_error(estimator, X, y)
    print("Estimated test error for Ridge  model (alpha={}): {}".format(alpha, estimated_test_error))
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print("Training error for Ridge model (alpha={}): {}".format(alpha, rmse))
    if plot:
        plot_residuals(y_pred, y, "Ridge (alpha={})".format(alpha))
    return model


def train_linear(X, y, plot=False):
    """
    Trains a Linear Regression Model.
    :param X: X of the current dataset
    :param y: target of the current dataset
    :param plot: either true or false. Controls if plots are shown while training this model.
    :return: trained Linear Regression model
    """
    print("Training Linear Model")
    estimator = LinearRegression()
    estimated_test_error = estimate_test_error(estimator, X, y)
    print("Estimated test error for Linear  model : {}".format(estimated_test_error))
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print("Training error for Linear model : {}".format(rmse))
    if plot:
        plot_residuals(y_pred, y, "Linear")
    return model


def random_forest_hp_tuning(estimator, X, y, nj):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
        random_state=42, scoring='neg_mean_squared_error', n_jobs=nj)

    # Fit the random search model
    rf_random.fit(X, y)
    # Best model from Random Grid Search CV
    print(rf_random.best_params_)
    # {'n_estimators': 400,
    # 'min_samples_split': 2,
    # 'min_samples_leaf': 1,
    # 'max_features': 'sqrt',
    # 'max_depth': None,
    # 'bootstrap': False}
    # print(rf_random.best_score_)
    # 0.018950937597094954


def xgb_hp_tuning(X, y):
    """
    Extreme boosted tree's hyper parameters tuning. Uses GridSearchCV from the Sci-kit learning library. The parameters
    that are tested were chosen based on the following information from the XGBoost documentation:

    https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html

    There are in general two ways that you can control overfitting in XGBoost:

    The first way is to directly control model complexity.

        This includes max_depth, min_child_weight and gamma.

    The second way is to add randomness to make training robust to noise.

        This includes subsample and colsample_bytree.

        You can also reduce stepsize eta. Remember to increase num_round when you do so.

    :param X: X of the current dataset
    :param y: target of the current dataset
    """
    max_depth = [int(x) for x in np.arange(start=1, stop=10, step=2)]
    min_child_weight = [int(x) for x in np.arange(start=1, stop=10, step=2)]
    gamma = [x for x in np.arange(start=0, stop=0.5, step=0.1)]
    n_estimators = [int(x) for x in np.arange(start=100, stop=3000, step=100)]
    learning_rate = [x for x in np.arange(start=0.01, stop=0.1, step=0.01)]
    grid = {'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'n_estimators': n_estimators,
            'learning_rate': learning_rate}
    xgb_model = xgb.XGBRegressor(n_jobs=-1)
    grid_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=grid, n_iter=100, cv=5, verbose=2,
                                     random_state=42, scoring='neg_mean_squared_error', n_jobs=1)
    # grid_search = GridSearchCV(estimator=xgb_model, param_grid=grid, cv=3, verbose=2, scoring='neg_mean_squared_error',
    #                           n_jobs=1)
    grid_search.fit(X, y)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    scores = grid_search.cv_results_['mean_test_score']
    params_combinations = grid_search.cv_results_['params']
    i = 0
    my_d = list()
    for combination in params_combinations:
        tmp = list(combination.values())
        tmp.append(scores[i])
        my_d.append(tmp)
        i += 1
    scores = np.array(my_d)
    i = 1
    for k in list(grid.keys()):
        p_index = get_param_index(k, params_combinations)
        df = get_df_parameter(p_index, grid[k], scores)
        x = df[:, 0]
        y = df[:, 1]
        plt.subplot(4, 2, i)
        plt.plot(x, y)
        plt.title(k)
        plt.ylabel("score")
        i += 1
    plt.show()
    # {'n_estimators': 860, 'min_child_weight': 3, 'max_depth': 10, 'gamma': 0}, 0.016546847765073678
    # {'n_estimators': 690, 'min_child_weight': 2, 'max_depth': 10, 'gamma': 0}
    #
    # {'n_estimators': 120, 'min_child_weight': 3, 'max_depth': 10, 'gamma': 0}, -0.01650642034698381
    # {'gamma': 0, 'max_depth': 10, 'min_child_weight': 3, 'n_estimators': 120}, -0.01650642034698381
    # {'gamma': 0, 'max_depth': 10, 'min_child_weight': 3, 'n_estimators': 130}, -0.01649072406422703
    # {'gamma': 0, 'max_depth': 10, 'min_child_weight': 3, 'n_estimators': 110}, -0.016452422168526753
    # {'gamma': 0, 'max_depth': 10, 'min_child_weight': 3, 'n_estimators': 100}, -0.01639837965635243
    # Using new data:
    # {'n_estimators': 180, 'min_child_weight': 2, 'max_depth': 10, 'gamma': 0}, -0.014972665473360368
    # {'n_estimators': 1900, 'min_child_weight': 7, 'max_depth': 1, 'learning_rate': 0.05, 'gamma': 0.0}, -0.013471785428194584
    # {'n_estimators': 2700, 'min_child_weight': 7, 'max_depth': 3, 'learning_rate': 0.04, 'gamma': 0.0}, -0.01402197351780234
    # {'n_estimators': 1600, 'min_child_weight': 3, 'max_depth': 3, 'learning_rate': 0.01, 'gamma': 0.0}, -0.013662255625430351
    # {'n_estimators': 1900, 'min_child_weight': 5, 'max_depth': 1, 'learning_rate': 0.06999999999999999, 'gamma': 0.0}, -0.013505285149724003
    # {'n_estimators': 500, 'min_child_weight': 7, 'max_depth': 3, 'learning_rate': 0.08, 'gamma': 0.0}, -0.013848513602727196
    # {'n_estimators': 1900, 'min_child_weight': 7, 'max_depth': 1, 'learning_rate': 0.05, 'gamma': 0.0}, -0.013471785428194584

    # {'n_estimators': 2700, 'min_child_weight': 7, 'max_depth': 3, 'learning_rate': 0.04, 'gamma': 0.0},-0.01402197351780234
    # {'n_estimators': 1600, 'min_child_weight': 3, 'max_depth': 3, 'learning_rate': 0.01, 'gamma': 0.0},-0.013662255625430351
    # {'n_estimators': 1900, 'min_child_weight': 5, 'max_depth': 1, 'learning_rate': 0.06999999999999999, 'gamma': 0.0},-0.013505285149724003
    # {'n_estimators': 500, 'min_child_weight': 7, 'max_depth': 3, 'learning_rate': 0.08, 'gamma': 0.0},-0.013848513602727196
    # {'n_estimators': 1900, 'min_child_weight': 7, 'max_depth': 1, 'learning_rate': 0.05, 'gamma': 0.0},-0.013471785428194584


def get_df_parameter(index, combinations_param, scores):
    result = list()
    for ne in combinations_param:
        s = scores[scores[:, index] == ne, 4]
        result.append([ne, np.mean(s)])
    df = np.array(result)
    return df


def get_param_index(param, combinations_param):
    index = list(combinations_param[0].keys()).index(param)
    return index


def train_random_forest(X, y, plot=False, ET=False, nj=None):
    """
    Trains a Random Forest Model.
    :param X: X of the current dataset
    :param y: target of the current dataset
    :param plot: either true or false. Controls if plots are shown while training this model.
    :return: trained Random Forest model
    """
    print("Training Random Forest")
    if(ET):
        estimator = ExtraTreesRegressor(n_jobs=nj)
        model = ExtraTreesRegressor(n_jobs=nj)
    else:
        estimator = RandomForestRegressor(n_estimators=400, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=None, bootstrap=False, n_jobs=nj)
        model = RandomForestRegressor(n_estimators=400, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=None, bootstrap=False, n_jobs=nj)

    estimated_test_error = estimate_test_error(estimator, X, y)
    model_name = type(estimator).__name__
    print("Estimated test error for {} model : {}".format(model_name, estimated_test_error))
    # Train RF Regressor with the whole data set
    model.fit(X, y)
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print("Training error for {} model : {}".format(model_name, rmse))
    if plot:
        plot_residuals(y_pred, y,  model_name)
    return model


def train_svr(X, y, plot=False, linear=False):
    """
    Trains a SVR Model. If the parameter linear is given, trains a Linear SVR.
    :param X: X of the current dataset
    :param y: target of the current dataset
    :param plot: either true or false. Controls if plots are shown while training this model.
    :param linear: either true or false. Controls if the trained model will be a Linear SVR or  Epsilon-SVR.
    :return: trained SVR model
    """
    print("Training SVR Model")
    if linear:
        estimator = LinearSVR()
        model = LinearSVR()
    else:
        estimator = SVR()
        model = SVR()
    model_name = type(estimator).__name__
    estimated_test_error = estimate_test_error(estimator, X, y)
    print("Estimated test error for {} model : {}".format(model_name, estimated_test_error))
    model.fit(X, y)
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print("Training error for {} model : {}".format(model_name, rmse))
    if plot:
        plot_residuals(y_pred, y, model_name)
    return model


def train_xgb(X, y, plot=False, param_dist=None):
    """
    Trains a boosted tree regressor using the XGBoost library (https://xgboost.readthedocs.io/en/).
    :param X: X of the current dataset
    :param y: target of the current dataset
    :param plot: either true or false. Controls if plots are shown while training this model.
    :return: trained extreme boosted tree model
    """
    if param_dist is None:
        param_dist = {'gamma': 0,
                      'max_depth': 1,
                      'min_child_weight': 2,
                      'n_estimators': 180,
                      'n_jobs': -1,
                      'subsample': 0.6,
                      'colsample_bytree': 0.6,
                      'learning_rate': 0.05}

    estimator = xgb.XGBRegressor(**param_dist)
    model_name = type(estimator).__name__
    estimated_test_error = estimate_test_error(estimator, X, y)
    print("Estimated test error for {} model : {}".format(model_name, estimated_test_error))
    model = xgb.XGBRegressor(**param_dist)
    model.fit(X, y)
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print("Training error for {} model : {}".format(model_name, rmse))
    if plot:
        plot_residuals(y_pred, y, model_name)
    return model


def train_meta_model(estimators, weights):
    """
    TODO check for stacked cv etc
    :param estimators:
    :param weights:
    :return:
    """
    pass


def wrap_and_submit(predictions, test_id, message):
    """
    Perform a submission to the kaggle competition with the given predictions and message
    :param predictions: predicted target values to be evaluated
    :param message: message for the submission
    :return:
    """
    submission = pd.DataFrame(pd.concat([test_id, predictions], axis=1), columns=["Id", "SalePrice"])
    submission.to_csv(basedir + "submission.csv", sep=",", header=True, index=False)
    os.system(command + "\"" + message + "\"")


def plot_residuals(y_pred, y, model):
    """
    Plots the residual values of a model.
    :param y_pred: predicted target values
    :param y: real target values
    :param model: model name to be used in the title of the plot
    :return:
    """
    preds = pd.DataFrame({"preds": y_pred, "true": y})
    preds["residuals"] = preds["true"] - preds["preds"]
    preds.plot(x="preds", y="residuals", kind="scatter", title="{} model residuals".format(model))
    sdv = np.std(preds.residuals.values)
    var = np.var(preds.residuals.values)
    ax = plt.gca()
    ax.axhline(color='lightgray')
    plt.text(0.90, 1, "stdev: " + sdv.__str__(), transform=ax.transAxes)
    plt.text(0.90, 0.95, "var:" + var.__str__(), transform=ax.transAxes)
    plt.show()


def estimate_test_error(estimator, X, y, k=5):
    """
    Estimates the test error using a given estimator. Perform K-Cross validation to obtain a better estimation of the error.
    :param estimator: estimator to compute the test error.
    :param X: data's examples/instances
    :param y: data's target value
    :param k: number of folds. Must be at least 2 due to scikit-learn restrictions.
    :return: estimated mean squared error for a given estimator on given test data.
    """
    kf = KFold(n_splits=k)
    errors = list()
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index].values, X.iloc[test_index].values
        y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        errors.append(np.sqrt(mse))
    mean_error = np.mean(np.array(errors))
    return mean_error


def load_data_set(filename):
    """
    Loads a dataset from a .csv file into a pandas DataFrame structure.
    :param filename: file's containing the desired dataset
    :return: DataFrame containing the loaded data
    """
    dataset = pd.read_csv(basedir + filename)
    return dataset


if __name__ == '__main__':
    """
    Load data sets
    """
    # Train data set
    df_train = load_data_set("train_houses.csv")
    df_test = load_data_set("test.csv")
    # Transform data set based on kaggle forums
    test_id = df_test['Id']
    df_test.drop(axis=1, labels=['Id'], inplace=True)
    df_train, df_test  = my_analysis(df_train, df_test)
    # df_train, transformations = transform_train_data(df_train)
    X = df_train.drop('SalePrice', axis=1)
    y = df_train['SalePrice']
    # Test data set
    # Transform data set, inversely on what was done previously
    # df_test = transform_test_data(df_test, transformations[0], df_train.columns)
    """
    Train models.
    Note: When making the predictions, np.exp() is used on the predictions given by models.
    This is due to performing np.log() on the target while transforming the training and test data.
    """
    # Lasso
    # Lasso hyper parameter tuning
    # lasso_alpha_tuning(X, y, False)
    # Training, prediction and submission (optional, uncomment to use)
    alphas = [0.01]
    for alpha in alphas:
        lasso = train_lasso(X, y, plot=False, alpha=alpha)
        # Use expm1 on submission's SalePrice because log1p was applied before
        lasso_predictions = np.expm1(lasso.predict(X=df_test))
        lasso_predictions = pd.DataFrame(lasso_predictions, columns=["SalePrice"])
        # wrap_and_submit(lasso_predictions, test_id, "lasso regression with alpha: " + str(lasso.alpha))
    # XGB
    # XGB hyper parameter tuning
    # xgb_hp_tuning(X, y)
    # training, prediction and submission
    # TODO check over fitting, random state
    params = [
        # {'n_estimators': 2700, 'min_child_weight': 7, 'max_depth': 3, 'learning_rate': 0.01, 'gamma': 0.0,
        # 'colsample_by_tree': 0.6, 'subsample': 0.6, 'random_state': 42},
        # {'n_estimators': 1600, 'min_child_weight': 3, 'max_depth': 3, 'learning_rate': 0.01, 'gamma': 0.0},
        # {'n_estimators': 1900, 'min_child_weight': 5, 'max_depth': 1, 'learning_rate': 0.06999999999999999, 'gamma': 0.0},
        # {'n_estimators': 500, 'min_child_weight': 7, 'max_depth': 3, 'learning_rate': 0.08, 'gamma': 0.0},
        {'n_estimators': 1900, 'min_child_weight': 7, 'max_depth': 1, 'learning_rate': 0.05, 'gamma': 0.0,
         'colsample_by_tree': 0.6, 'subsample': 0.6, 'random_state': 42},
    ]
    for param_dist in params:
        xgb_model = train_xgb(X, y, False, param_dist)
        xgb_predictions = np.expm1(xgb_model.predict(data=df_test[X.columns]))
        xgb_predictions = pd.DataFrame(xgb_predictions, columns=["SalePrice"])
#        wrap_and_submit(xgb_predictions, test_id, "{}: {}".format(type(xgb_model).__name__, param_dist))
    tmp1 = xgb_predictions.apply(lambda x: x * 0.70)
    tmp2 = lasso_predictions.apply(lambda x: x * 0.30)
    final_predictions = tmp1.add(tmp2, fill_value=0)
    wrap_and_submit(final_predictions, test_id, "Meta model using best lasso and xgb scores".format())
    sys.exit(0)

    # Ridge
    # Ridge hyper parameter tuning
    # ridge_alpha_tuning(X, y, False)
    # training, prediction and submission (optional, uncomment to use)
    alphas = [15.1, 14.5, 15.4, 14.8, 14.9]
    for alpha in alphas:
        ridge = train_ridge(X, y, plot=False, alpha=alpha)
        ridge_predictions = np.expm1(ridge.predict(X=df_test))
        ridge_predictions = pd.DataFrame(ridge_predictions, columns=["SalePrice"])
        # wrap_and_submit(ridge_predictions, "ridge regression with alpha: " + str(ridge.alpha))

    # Linear
    # Training, prediction and submission (optional, uncomment to use)
    linear = train_linear(X, y, False)
    linear_predictions = np.exp(linear.predict(X=df_test))
    linear_predictions = pd.DataFrame(linear_predictions, columns=["SalePrice"])
    # wrap_and_submit(linear_predictions, "linear submission")

    # Random forests
    # Training, prediction and submission (optional, uncomment to use)
    random_forest = train_random_forest(X, y, False)
    random_forest_predictions = np.exp(random_forest.predict(X=df_test))
    random_forest_predictions = pd.DataFrame(random_forest_predictions, columns=["SalePrice"])
    # wrap_and_submit(random_forest_predictions, type(random_forest).__name__)

    # Extra Trees
    # Training, prediction and submission (optional, uncomment to use)
    random_forest = train_random_forest(X, y, False, ET=True)
    random_forest_predictions = np.exp(random_forest.predict(X=df_test))
    random_forest_predictions = pd.DataFrame(random_forest_predictions, columns=["SalePrice"])
    # wrap_and_submit(random_forest_predictions, type(random_forest).__name__)

    # Support Vector Regressor
    # Training, prediction and submission (optional, uncomment to use)
    svr = train_svr(X, y, False)
    svr_predictions = np.exp(svr.predict(X=df_test))
    svr_predictions = pd.DataFrame(svr_predictions, columns=["SalePrice"])
    # wrap_and_submit(svr_predictions, type(svr).__name__)

    # SVR Linear Regression
    svr = train_svr(X, y, False, linear=True)
    svr_predictions = np.exp(svr.predict(X=df_test))
    svr_predictions = pd.DataFrame(svr_predictions, columns=["SalePrice"])
    # wrap_and_submit(svr_predictions, type(svr).__name__)
