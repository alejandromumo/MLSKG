from multivariate_analysis import *
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Import models
from sklearn.linear_model import Ridge, RidgeCV, LassoCV, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

basedir = "C:/Users/ManuelCosta/Documents/GitHub/MLSKG/house-prices-advanced-regression-techniques/"
command = "kaggle competitions submit -c house-prices-advanced-regression-techniques -f submission.csv -m "


# TODO - tested without scaling SalePrice and it works in the same way

def train_lasso(X, y, plot=False):
    # Train Lasso model
    alphas = [1, 0.1, 0.001, 0.0005]
    lasso = LassoCV(alphas=alphas, cv=10)
    lasso.fit(X=X, y=y)
    scores = lasso.mse_path_
    # CV scores:
    # mean  : 0.020324380305446494
    if plot:
        f, ax = plt.subplots(figsize=(8, 10))
        coef = pd.Series(lasso.coef_, index = X.columns)

        imp_coef = pd.concat([coef.sort_values().head(10),
                              coef.sort_values().tail(10)])

        imp_coef.plot(kind="barh")
        plt.title("Coefficients in the Lasso Model")
        plt.show()
        f, ax = plt.subplots(figsize=(6, 6))

        preds = pd.DataFrame({"preds": lasso.predict(X), "true": y})
        preds["residuals"] = preds["true"] - preds["preds"]
        preds.plot(x="preds", y="residuals", kind="scatter")
        plt.show()
        # Alpha = 0.0005 seems a good value
    lasso = Lasso(alpha=0.0005)
    lasso.fit(X=X, y=y)
    # Train error (using mean squared error):
    # 221 coefficients. It managed to get rid of 51 features.
    return lasso


def train_ridge(X, y, plot=False):
    # Train Ridge model
    alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
    ridge_cv = RidgeCV(alphas=alphas, cv=None, store_cv_values=True)
    ridge_cv.fit(X=X, y=y)
    scores = ridge_cv.cv_values_
    # CV scores
    # mean  : 0.012599418861990593
    # min : 6.971718617122844e-11
    if plot:
        means = np.mean(scores, axis=0)
        mins = np.min(scores, axis=0)
        f, ax = plt.subplots(figsize=(8, 6))
        ax.xaxis.set_ticks(np.arange(0, 75, 5))
        fig = sns.lineplot(data=pd.Series(means, index=alphas))
        plt.show()
    # Alpha = between 5 and 15 seems a good value
    # Train with alpha = 5
    ridge = Ridge(alpha=5)
    ridge.fit(X=X, y=y)
    # Train error (using mean squared error):
    # 221 coefficients. It managed to get rid of 51 features.
    return ridge


def train_linear(X, y, plot=False):
    # Train Linear model
    linear = LinearRegression()
    linear.fit(X=X, y=y)
    # Train score:
    # score = 0.9455712442934993
    # 221 coefficients. It managed to get rid of 51 features.
    return linear


def train_random_forest(X, y, plot=False):
    # Number of trees in random forest
    # n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # # Number of features to consider at every split
    # max_features = ['auto', 'sqrt']
    # # Maximum number of levels in tree
    # max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    # max_depth.append(None)
    # # Minimum number of samples required to split a node
    # min_samples_split = [2, 5, 10]
    # # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 4]
    # # Method of selecting samples for training each tree
    # bootstrap = [True, False]
    # # Create the random grid
    # random_grid = {'n_estimators': n_estimators,
    #                'max_features': max_features,
    #                'max_depth': max_depth,
    #                'min_samples_split': min_samples_split,
    #                'min_samples_leaf': min_samples_leaf,
    #                'bootstrap': bootstrap}
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor(n_estimators=400, min_samples_split=2,min_samples_leaf=1, max_features='sqrt',
                               max_depth=None, bootstrap=False)
    rf.fit(X, y)
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    # rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
    #                               random_state=42, scoring='neg_mean_squared_error')

    # Fit the random search model
    # rf_random.fit(X, y)

    # Best model from Random Grid Search CV
    # print(rf_random.best_params_)
    # {'n_estimators': 400,
    # 'min_samples_split': 2,
    # 'min_samples_leaf': 1,
    # 'max_features': 'sqrt',
    # 'max_depth': None,
    # 'bootstrap': False}
    # print(rf_random.best_score_)
    # 0.018950937597094954
    return rf


def wrap_and_submit(predictions, message):
    submission = pd.DataFrame(pd.concat([df_test["Id"], predictions], axis=1), columns=["Id", "SalePrice"])
    submission.to_csv(basedir + "submission.csv", sep=",", header=True, index=False)
    os.system(command + "\"" + message + "\"")


# Load datasets
# Train
df_train = pd.read_csv("C:/Users/ManuelCosta/Documents/GitHub/"
                       "MLSKG/house-prices-advanced-regression-techniques/train_houses.csv")
# 81 features
df_train, transformations = transform_train_data(df_train)
scaler = transformations[1]
# 221 features + target
X = df_train.drop('SalePrice', axis=1)
y = df_train['SalePrice']

# Test
df_test = pd.read_csv("C:/Users/ManuelCosta/Documents/GitHub/"
                      "MLSKG/house-prices-advanced-regression-techniques/test.csv")
df_test = transform_test_data(df_test, transformations[0], df_train.columns)

# Train models
# Ridge training and prediction
ridge = train_ridge(X, y)
ridge_predictions = np.exp(ridge.predict(X=df_test))
ridge_predictions = pd.DataFrame(ridge_predictions, columns=["SalePrice"])
# wrap_and_submit(ridge_predictions, "ridge regression with alpha: " + str(ridge.alpha))

# Lasso training and prediction
lasso = train_lasso(X, y)
lasso_predictions = np.exp(lasso.predict(X=df_test))
lasso_predictions = pd.DataFrame(lasso_predictions, columns=["SalePrice"])
# wrap_and_submit(lasso_predictions, "lasso regression with alpha: " + str(lasso.alpha))

# Linear training and prediction
linear = train_linear(X, y)
linear_predictions = np.exp(linear.predict(X=df_test))
linear_predictions = pd.DataFrame(linear_predictions, columns=["SalePrice"])
# wrap_and_submit(linear_predictions, "linear submission")

# Random forests training and prediction
random_forest = train_random_forest(X, y)
random_forest_predictions = np.exp(random_forest.predict(X=df_test))
random_forest_predictions = pd.DataFrame(random_forest_predictions, columns=["SalePrice"])
# wrap_and_submit(random_forest_predictions, "RF regression" )

