# MLSKG
Participation in Kaggle's Machine Learning competitions in the context of Machine Learning course in Aristotle University of Thessaloniki, Greece.

## Kaggle Competitions

- [Regression techniques for house's price prediction ](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/ "House Prices Advanced Regression Techniques")

- [Classification of bank transactions for fraud detection](https://www.kaggle.com/c/santander-customer-transaction-prediction/ "Santander Customer Transaction Prediction")

This repository contains both competitions source code. Each competition has its own sub-directory and independent files. Datasets can be found in the competitions page (section "Data")

***In order to test the code, the dataset files must be located in the same directory as the main source files.***
## Regression

The competition's challenge is to predict houses prices using regression techniques. 
Approximately 1900 houses from Ames, Iowa are available for training, validation and testing. Each house has *"79 explanatory variables describing (almost) every aspect of residential homes"* (from the competition's land page, section *Competition Description*)

### Work description
Data pre-processing is made using [Pedro Marcelino's ](http://pmarcelino.com/) approach found in [Comprehensive data exploration with Python](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python#Out-liars!). 
 The data, after being pre-processed, is used to train several regression models.

#### Models
- Linear Models
  * Ordinary Least Squares Linear Regressor
  * Lasso Regressor
  * Ridge Regressor
- Random Forests
  * Random Forest Regressor
  * Extra Trees
- Support Vector Machines
  * Support Vector Regressor (SVR)
  * Linear SVR
- Boosting
  * in progress
### Results
| Model  | Estimated test error | Training rmse | Real score | #  | Notes                                                                                                          |
|--------|----------------------|---------------|------------|----|----------------------------------------------------------------------------------------------------------------|
| Linear | 0.120908904          | 0.093253489   | 0.59279    | 1  | Default                                                                                                        |
| RF     | 0.145722203          | 0.06105498    | 0.18264    | 2  | Default                                                                                                        |
| RF     | 0.134576161          | 0.061981003   | 0.18385    | 3  | n_estimators=400, min_samples_split=2, min_samples_leaf=1, max_features='sqrt',max_depth=None, bootstrap=False |
| Ridge  | 0.116345852          | 0.094236802   | 0.43935    | 4  | alpha=1 (Default)                                                                                              |
| Ridge  | 0.113571525          | 0.096893286   | 0.32987    | 5  | alpha=5                                                                                                        |
| Ridge  | 0.113392101          | 0.098962901   | 0.29065    | 6  | alpha=10                                                                                                       |
| Ridge  | 0.11358832           | 0.100400247   | 0.26921    | 7  | alpha=15                                                                                                       |
| Lasso  | 0.116345852          | 0.167788121   | 0.18687    | 8  | alpha=1(Default)                                                                                               |
| Lasso  | 0.120819004          | 0.128345118   | 0.13698    | 9  | alpha=0.01                                                                                                     |
| ET     | 0.142216183          | 1.20E-05      | 0.20535    | 10 | Default                                                                                                        |
| ET     | 0.131307263          | 1.35E-05      | 0.1949     | 11 | n_estimators=400                                                                                               |
| SVR    | 0.399700654          | 0.100100785   | 0.41927    | 12 | Default                                                                                                        |
| LSVR   | 0.20549275           | 0.187193702   |            | 13 | Default                                                                                                        |

### Future Work
- Some models were briefly analysed such as Support Vector Machines and Extra trees. Their hyper-parameters combinations were not deeply explored and there is work to be done in that aspect.
- Data is "dummy" encoded (a.k.a. one-hot encoded) by default using panda's method ```get_dummies()``` [(doc)](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html). Further analysis allows data to be encoded manually. The file *data_description.txt* already contains some data analysis that separates categorical data into ordinal and nominal but it is not implemented yet.
## Classification

