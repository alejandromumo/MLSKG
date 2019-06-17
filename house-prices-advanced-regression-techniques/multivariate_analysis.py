# invite people for the Kaggle party
# From: https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python#Out-liars!
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def transform_train_data(df_train):
    returned_objects = []
    # dealing with missing data
    total = df_train.isnull().sum().sort_values(ascending=False)
    percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    dropped_indexes = (missing_data[missing_data['Total'] > 1]).index
    print(dropped_indexes)
    returned_objects.append(dropped_indexes)
    df_train = df_train.drop(dropped_indexes, 1)
    df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
    df_train.isnull().sum().max()  # just checking that there's no missing data missing...
    # standardizing data
    # TODO Try without scaling and log the target SalePrice
    scaler = StandardScaler()
    # saleprice_scaled = scaler.fit_transform(df_train['SalePrice'][:, np.newaxis])
    returned_objects.append(scaler)
    # low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
    # high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
    #print('outer range (low) of the distribution:')
    #print(low_range)
    #print('\nouter range (high) of the distribution:')
    #print(high_range)
    # deleting points
    df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
    df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
    # applying log transformation to achieve normal distribution
    # TODO do I need to apply inverse log in test to obtain thousands of dollars?
    df_train['SalePrice'] = np.log(df_train['SalePrice'])
    # data transformation
    # TODO use inverse (exp) in testing
    df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
    # create column for new variable (one is enough because it's a binary categorical feature)
    # if area>0 it gets 1, for area==0 it gets 0
    # TODO add this
    df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
    df_train['HasBsmt'] = 0
    # TODO apply this on test
    # Every house that has basement SF bigger than 0, set hasBsmt to 1
    df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
    # transform data
    # TODO apply this on test. Use inverse (exp) in testing
    # Every house with basement, transform the squared feet to log
    df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
    # convert categorical variable into dummy
    # TODO apply this on test
    # TODO try customized encoding
    new_df = pd.get_dummies(df_train)
    print(new_df.columns)
    return new_df, returned_objects


def transform_test_data(df_test, dropped_features, train_features):
    print(dropped_features)
    # First: drop features and add
    # Second: apply inverse log (exp) to:
    #   - GrLivArea
    #   - TotalBsmtSF
    df_test = df_test.drop(dropped_features, 1)
    df_test = df_test.drop(df_test.loc[df_test['Electrical'].isnull()].index)
    df_test.isnull().sum().max()  # just checking that there's no missing data missing...
    df_test['HasBsmt'] = pd.Series(len(df_test['TotalBsmtSF']), index=df_test.index)
    df_test['HasBsmt'] = 0
    df_test.loc[df_test['TotalBsmtSF']>0,'HasBsmt'] = 1
    df_test.loc[df_test['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_test['TotalBsmtSF'])
    df_test['GrLivArea'] = np.log(df_test['GrLivArea'])
    df_test = pd.get_dummies(df_test)
    test_features = df_test.columns
    features_to_add = np.setdiff1d(train_features.values, test_features.values)
    for feature in features_to_add:
        df_test.insert(len(df_test.columns), feature, 0, True)
    df_test = df_test.fillna(0)
    df_test = df_test.drop("SalePrice", axis=1)
    return df_test


def to_thousands_of_dollars(scaler):
    pass





def get_plots_and_analysis(df_train):
    var = 'OverallQual'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000)
    # Year built
    var = 'YearBuilt'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    f, ax = plt.subplots(figsize=(16, 8))
    fig = sns.boxplot(x=var, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000)
    plt.xticks(rotation=90)
    # correlation matrix
    corrmat = df_train.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    # saleprice correlation matrix
    k = 10 # number of variables for heatmap
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(df_train[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
    # scatterplot
    sns.set()
    # Strongly correlated variables
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    sns.pairplot(df_train[cols], size = 2.5)
    plt.show()
    # check missing data
    total = df_train.isnull().sum().sort_values(ascending=False)
    percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    missing_data.head(20)
    # dealing with missing data
    df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index, 1)
    df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
    df_train.isnull().sum().max() # just checking that there's no missing data missing...
    # standardizing data
    saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])
    low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
    high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
    print('outer range (low) of the distribution:')
    print(low_range)
    print('\nouter range (high) of the distribution:')
    print(high_range)
    # bivariate analysis saleprice/grlivarea
    var = 'GrLivArea'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
    # deleting points
    df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
    df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
    #bivariate analysis saleprice/grlivarea
    var = 'TotalBsmtSF'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
    #histogram and normal probability plot
    sns.distplot(df_train['SalePrice'], fit=norm)
    fig = plt.figure()
    res = stats.probplot(df_train['SalePrice'], plot=plt)
    # applying log transformation to achieve normal distribution
    df_train['SalePrice'] = np.log(df_train['SalePrice'])
    # transformed histogram and normal probability plot
    sns.distplot(df_train['SalePrice'], fit=norm)
    fig = plt.figure()
    res = stats.probplot(df_train['SalePrice'], plot=plt)
    # data transformation
    df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
    # transformed histogram and normal probability plot
    sns.distplot(df_train['GrLivArea'], fit=norm);
    fig = plt.figure()
    res = stats.probplot(df_train['GrLivArea'], plot=plt)
    # histogram and normal probability plot
    sns.distplot(df_train['TotalBsmtSF'], fit=norm);
    fig = plt.figure()
    res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
    # create column for new variable (one is enough because it's a binary categorical feature)
    # if area>0 it gets 1, for area==0 it gets 0
    df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
    df_train['HasBsmt'] = 0
    df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
    # transform data
    df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
    # histogram and normal probability plot
    sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm)
    fig = plt.figure()
    res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
    # scatter plot
    plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])
    # scatter plot
    plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice'])
    # convert categorical variable into dummy
    df_train = pd.get_dummies(df_train)
