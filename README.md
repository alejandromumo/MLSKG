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
  * Extreme Gradient Boosted Trees
### Results
| Model  | Estimated test error | Training rmse | Real score | #  | Notes                                                                                                                      |
|--------|----------------------|---------------|------------|----|----------------------------------------------------------------------------------------------------------------------------|
| Linear | 0.120908904          | 0.093253489   | 0.59279    | 1  | Default                                                                                                                    |
| RF     | 0.145722203          | 0.06105498    | 0.18264    | 2  | Default                                                                                                                    |
| RF     | 0.134576161          | 0.061981003   | 0.18385    | 3  | n_estimators=400, min_samples_split=2, min_samples_leaf=1, max_features='sqrt',max_depth=None, bootstrap=False             |
| Ridge  | 0.116345852          | 0.094236802   | 0.43935    | 4  | alpha=1 (Default)                                                                                                          |
| Ridge  | 0.113571525          | 0.096893286   | 0.32987    | 5  | alpha=5                                                                                                                    |
| Ridge  | 0.113392101          | 0.098962901   | 0.29065    | 6  | alpha=10                                                                                                                   |
| Ridge  | 0.11358832           | 0.100400247   | 0.26921    | 7  | alpha=15                                                                                                                   |
| Lasso  | 0.116345852          | 0.167788121   | 0.18687    | 8  | alpha=1(Default)                                                                                                           |
| Lasso  | 0.120819004          | 0.128345118   | 0.13698    | 9  | alpha=0.01                                                                                                                 |
| ET     | 0.142216183          | 1.20E-05      | 0.20535    | 10 | Default                                                                                                                    |
| ET     | 0.131307263          | 1.35E-05      | 0.1949     | 11 | n_estimators=400                                                                                                           |
| SVR    | 0.399700654          | 0.100100785   | 0.41927    | 12 | Default                                                                                                                    |
| LSVR   | 0.20549275           | 0.187193702   |            | 13 | Default                                                                                                                    |
| XGB    | 0.119230251          | 0.053223098   | 0.13992    | 14 | n_estimators=400                                                                                                           |
| XGB    | 0.122424538          | 0.022773584   | 0.13458    | 15 | gamma': 0, 'max_depth': 10, 'min_child_weight': 3, 'n_estimators': 100                                                     |
| XGB    | 0.122356405          | 0.019058915   | 0.13455    | 16 | gamma': 0, 'max_depth': 10, 'min_child_weight': 3, 'n_estimators': 110                                                     |
| XGB    | 0.122323862          | 0.016552849   | 0.13454    | 17 | gamma': 0, 'max_depth': 10, 'min_child_weight': 3, 'n_estimators': 120                                                     |
| XGB    | 0.122356414          | 0.014266287   | 0.1344     | 18 | gamma': 0, 'max_depth': 10, 'min_child_weight': 3, 'n_estimators': 130                                                     |
| XGB    | 0.121933305          | 0.034796934   | 0.13143    | 19 | gamma': 0, 'max_depth': 10, 'min_child_weight': 3, 'n_estimators': 100, subsample:0.6, colsample_bytree:0.6                |
| XGB    | 0.148958117          | 0.107965568   | 0.14948    |    | gamma': 0, 'max_depth': 10, 'min_child_weight': 3, 'n_estimators': 100, subsample:0.6, colsample_bytree:0.6, reg_lamba:100 |
| XGB    | 0.229409825          | 0.192657999   | 0.21862    |    | gamma': 0, 'max_depth': 10, 'min_child_weight': 3, 'n_estimators': 100, subsample:0.6, colsample_bytree:0.6, reg_lamba:500 |
| XGB    | 0.136947548          | 0.088152734   | 0.14723    |    | gamma': 0, 'max_depth': 10, 'min_child_weight': 3, 'n_estimators': 100, subsample:0.6, colsample_bytree:0.6, reg_lamba:50  |

### Future Work
- Some models were briefly analysed such as Support Vector Machines and Extra trees. Their hyper-parameters combinations were not deeply explored and there is work to be done in that aspect.
- Data is "dummy" encoded (a.k.a. one-hot encoded) by default using panda's method ```get_dummies()``` [(doc)](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html). Further analysis allows data to be encoded manually. The file *data_description.txt* already contains some data analysis that separates categorical data into ordinal and nominal but it is not implemented yet.
## Classification
The competition's challenge is to classify bank transactions to detect fraudulent ones. It is a binary classification problem where models are expected to classify bank transactions in one of the two existing classes: fraud and legitimate. The data set contains more than 100k transactions where each one is characterized by 400 encoded, numerical columns.

The main focuses of this challenge are: 

- Data exploration on big data sets. 

- Train classification models using hyper-parameter tuning. 

### Work Description 
One of the main challenges from this assignment is its dataset's size. Algorithms that have higher complexity (like SVM with rbf kernel) are slow on this data especially when we the tuning of its hyper-parameters is performed with cross-validation. Sub-sampling is an alternative approach implemented to tackle this. 

Moreover, the train data set is unbalanced (8.95:1). Both the classification algorithms and the score metrics take the unbalance into account (i.e. error scores are weighted).


### Models
- LDA (Linear Discriminant Analysis): 

    * Linear with default settings 

    * Quadratic with default settings 
- Logistic Regression:
    
    * Logistic Regressor with regularization factor of 1.0000000000000005e-09. 
Hyper-parameters were determined using cross-validation (10-fold). 

- Support Vector Machines
Hyper-parameters were determined by cross-validation using sub-sampling due to the data set's size. 
Three models were tested using rbf kernel.

* Model #2 (1.5% of the data was used to find the best hyper-parameters, 3-fold) 

* Model #4 (2.5% of the data was used to find the best hyper-parameters, 10-fold) 

* Model #1 (0.25% of the data was used to find the best hyper-parameters, 10-fold) 
  

### Results

|   Model   |    AUC   | #  |                                Notes                                 |
|:---------:|:--------:|:--:|:--------------------------------------------------------------------:|
| SVM       | 0.77907  | 1  | rbf kernel, C=512 gamma=0.0001220703125.  Sub-sampling (0.025)       |
| SVM       | 0.77873  | 2  | rbf kernel, C=1024 gamma=0.0001220703125. Sub sampling data (0.015)  |
| Logistic  | 0.5241   | 3  | penalty="l2", max_iter=500, C= 1.0000000000000005e-09                |
| SVM       | 0.77916  | 4  | C=8192, gamma=3.0517578125e-05, Sub-sampling (0.0025)                |
| LDA       | 0.63557  | 5  | Default                                                              |
| QDA       | 0.64633  | 6  | Default                                                              |
