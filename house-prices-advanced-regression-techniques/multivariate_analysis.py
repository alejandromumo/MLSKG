import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats, special
import warnings
warnings.filterwarnings('ignore')


def transform_train_data(df_train):
    """
    Pre-process training data using multi-variate statistical analysis.
    Retrieved from: https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python#Out-liars!
    :param df_train: DataFrame containing the training data.
    :return: tuple containing the new data frame and transformations used (data scaler).
    """
    returned_objects = []
    # dealing with missing data
    total = df_train.isnull().sum().sort_values(ascending=False)
    percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    dropped_indexes = (missing_data[missing_data['Total'] > 1]).index
    returned_objects.append(dropped_indexes)
    df_train = df_train.drop(dropped_indexes, 1)
    df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
    df_train.isnull().sum().max()  # just checking that there's no missing data missing...
    # standardizing data
    scaler = StandardScaler()
    # saleprice_scaled = scaler.fit_transform(df_train['SalePrice'][:, np.newaxis])
    returned_objects.append(scaler)
    # low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
    # high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
    # deleting points
    df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
    df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
    # applying log transformation to achieve normal distribution
    df_train['SalePrice'] = np.log(df_train['SalePrice'])
    # data transformation
    df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
    # create column for new variable (one is enough because it's a binary categorical feature)
    # if area>0 it gets 1, for area==0 it gets 0
    df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
    df_train['HasBsmt'] = 0
    # Every house that has basement SF bigger than 0, set hasBsmt to 1
    df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
    # transform data
    # Every house with basement, transform the squared feet to log
    df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
    # convert categorical variable into dummy
    new_df = pd.get_dummies(df_train)
    return new_df, returned_objects


def transform_test_data(df_test, dropped_features, train_features):
    """
    Pre-process test data using the inverse operations performed on previously analysed training data.
    - Drops features that were dropped from the training dataset
    - Adds features that were added in the training dataset.
    - Applies inverse log (exp) to features that were scaled using log operation in the training dataset (GrLivArea, TotalBsmtSF)
    :param df_test: DataFrame containing the test data.
    :param dropped_features: list of features that were dropped during training
    :param train_features: features that currently exist in the training dataset. 
    :return: DataFrame containing the test data, compatible with the training data, eady to be used.
    """
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


def get_plots_and_analysis(df_train):
    """
    Shows plots related with data analysis. Does not perform any writing operations on training data. 
    Used for analysis only.
    :param df_train: DataFrame containing the training data.
    """
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
    sns.pairplot(df_train[cols], size=2.5)
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
    sns.distplot(df_train['GrLivArea'], fit=norm)
    fig = plt.figure()
    res = stats.probplot(df_train['GrLivArea'], plot=plt)
    # histogram and normal probability plot
    sns.distplot(df_train['TotalBsmtSF'], fit=norm)
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


def my_analysis(df_train, df_test):
    """
    TODO
    :param df_train:
    :return:
    """
    sns.set()
    fig = plt.figure()
    # sns.distplot(df_train["SalePrice"], fit=stats.johnsonsb, kde=False, fit_kws={'color': 'green'})
    # sns.distplot(df_train["SalePrice"], fit=stats.lognorm, kde=False, fit_kws={'color': 'blue'},
    #              axlabel="Sale Price Distribution (log fit)")
    # plt.show(block=False)
    # print(df_train['SalePrice'].describe())
    # fig = plt.figure()
    # df_train["SalePrice"] = np.log(df_train["SalePrice"])
    # sns.distplot(df_train["SalePrice"], fit=stats.lognorm, kde=False, fit_kws={'color': 'blue'},
    #              axlabel="Log Sale Price Distribution")
    # plt.show()
    # input()
    # cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'Fireplaces']
    # Distributions of selected features (fit=log)
    # for c in cols:
    #     fig = plt.figure()
    #     sns.distplot(df_train[c], fit=stats.lognorm, kde=False, fit_kws={'color': 'blue'},
    #                  axlabel="{} Distribution (log fit)".format(c))
    #     plt.show(block=False)
    # plt.show()
    # input()
    # # Distribution of selected features (fit=norm)
    # for c in cols:
    #     fig = plt.figure()
    #     sns.distplot(df_train[c], fit=stats.norm, kde=False, fit_kws={'color': 'blue'},
    #                  axlabel="{} Distribution (norm fit)".format(c))
    #     plt.show(block=False)
    # # Scatter plots of selected features
    # plt.show()
    # input()
    # cols_scatter = ['GrLivArea', 'TotalBsmtSF', 'YearBuilt']
    # for c in cols_scatter:
    #     fig = plt.figure()
    #     sns.scatterplot(x=c, y='SalePrice', data=pd.concat([df_train[c], df_train['SalePrice']], axis=1))
    #     plt.show(block=False)
    # # Boxplot of selected features
    # cols_box_plot = ['OverallQual', 'GarageCars', 'FullBath', 'Fireplaces']
    # for c in cols_box_plot:
    #     fig = plt.figure()
    #     sns.boxplot(x=c, y='SalePrice', data=pd.concat([df_train[c], df_train['SalePrice']], axis=1))
    #     plt.show(block=False)
    remove_outliers(df_train)
    df_train.drop(axis=1, labels=['Id'], inplace=True)
    deal_missing_data(df_train, df_test)
    categorical_to_ordinal_features(df_train, df_test)
    transform_numerical_features(df_train, df_test)
    new_df_train, new_df_test = pd.get_dummies(df_train), pd.get_dummies(df_test)
    if new_df_train.shape[1] != new_df_test.shape[1]:
        train_features = new_df_train.columns
        test_features = new_df_test.columns
        features_to_add = np.setdiff1d(train_features.values, test_features.values)
        features_to_add = np.delete(features_to_add, np.where(features_to_add == 'SalePrice'))
        for feature in features_to_add:
            new_df_test[feature] = 0
        features_to_add = np.setdiff1d(test_features.values, train_features.values)
        for feature in features_to_add:
            new_df_train[feature] = 0
    assert (new_df_train.isna().any().sum() == 0)
    assert (new_df_test.isna().any().sum() == 0)
    assert (new_df_train.shape[1] == new_df_test.shape[1]+1)
    return new_df_train, new_df_test


def remove_outliers(df_train):
    """
    Removes outliers, in place, from a given data frame.
    After analysing the scatter plot of the features with the SalePrice, the following cases were detected:
    - GrLivArea : i=523 and i=1298 are obvious outliers
    - TotalBsmtSF: i=1298 is an outlier
                   i=440, 523, 496, 332 might be outliers
    :param df_train: Train data frame
    """
    indexes = [523, 1298]
    df_train.drop(axis=0, labels=indexes, inplace=True)


def deal_missing_data(df_train, df_test):
    """
    Deals with missing data (NaN). Changes are made in place, does not return a new data frame.
    For categorical features:
        - Some features have NaN as a possible value. Thus, they are not missing values.
        In such cases, values are changed to 'None'
        - Features that have in fact missing values are filled with the mode.
    For numerical features:
        - Missing values are filled the feature's median value.
    :param df_train: Train data frame
    """
    total = df_train.isnull().sum().sort_values(ascending=False)
    percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    missing_data = missing_data[missing_data['Total'] > 0]
    # Fill rows that have missing data with the median value of the feature
    numerical_features = df_train.select_dtypes(exclude=object).columns
    for c in numerical_features:
        df_train[c].fillna(np.nanmedian(df_train[c]), inplace=True)
        if c != 'SalePrice':
            df_test[c].fillna(np.nanmedian(df_test[c]), inplace=True)
    # Fill rows that have missing data with specific values
    df_train['MasVnrType'].fillna('None', inplace=True)
    df_train['Electrical'].fillna('SBrkr', inplace=True)
    # Fill rows that have missing data with None or the median (numerical features)
    categorical_features = df_train.select_dtypes(include=object).columns
    for c in categorical_features:
        df_train[c].fillna('None', inplace=True)
        df_test[c].fillna('None', inplace=True)
    # Drop PoolQC because almost 100% has no information regarding it.
    #   The feature PoolArea represents most of the feature PoolQC.
    df_train.drop(['PoolQC'], axis=1, inplace=True)
    df_test.drop(['PoolQC'], axis=1, inplace=True)
    assert(df_train.isna().any().sum() == 0)
    assert (df_test.isna().any().sum() == 0)


def combine_features(df_train):
    # TODO
    pass


def categorical_to_ordinal_features(df_train, df_test):
    """
    Transforms categorical features into ordinal. Selected features are based on manual analysis of data.
    The transformation is made in place (i.e. does not return a new data frame).
    :param df_train: Train data frame
    """
    mapper = {'LotShape': {'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0},
              'LandContour': {'Lvl': 1, 'Bnk': 0, 'HLS': 3, 'Low': 2},
              'LandSlope': {'Mod': 2, 'Gtl': 1, 'Sev': 0},
              'ExterQual': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
              'ExterCond': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
              'BsmtQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
              'BsmtCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
              'BsmtExposure': {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0},
              'BsmtFinType1': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0},
              'BsmtFinType2': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0},
              'HeatingQC': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
              'CentralAir': {'Y': 1, 'N': 0},
              'KitchenQual': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
              'FireplaceQu': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
              'GarageFinish': {'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0},
              'GarageQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
              'GarageCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
              'Fence': {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'None': 0}
              }
    for k in list(mapper.keys()):
        df_train[k].replace(mapper.get(k), inplace=True)
        df_test[k].replace(mapper.get(k), inplace=True)
    assert(df_train.isna().any().all() == False)
    assert(df_test.isna().any().all() == False)


def transform_numerical_features(df_train, df_test):
    """
    TODO currently deals with positive skewed features, not negative. Analyse this.
    :param df_train:
    :param df_test:
    :return:
    """
    # apply log, scaling features
    # SalePrice. Check log against log(1+x), log1p
    df_train['SalePrice'] = np.log1p(df_train['SalePrice'])
    # Check for skewed features
    features_skewness = []
    for k in df_train.columns:
        if k == 'SalePrice':
            pass
        elif df_train.dtypes[k] != object:
            features_skewness.append([k, np.float(stats.skew(df_train[k]))])
    features_skewness_df = pd.DataFrame(features_skewness, columns=['F', 'S']).sort_values(by='S')
    left_skewed_features = features_skewness_df[features_skewness_df['S'] > 0.5]['F']
    right_skewed_features = features_skewness_df[features_skewness_df['S'] < -0.5]['F']
    # Apply log for right-skewed features (skewness<-0.5)
    # Apply boxcox1p for left-skewed features (skewness>0.5)
    #     boxcox1p(x,lmbda):
    #       y = log(1+x) if lmbda==0
    #       y = ((1+x)**lmbda - 1) / lmbda  if lmbda != 0
    # the Box-Cox Power transformation only works if all the data is positive and greater than 0
    for f in left_skewed_features:
        df_train[f] = special.boxcox1p(df_train[f], stats.boxcox_normmax(df_train[f] + 1))
        df_test[f] = special.boxcox1p(df_test[f], stats.boxcox_normmax(df_test[f] + 1))

'''
Notes about data:
---
Based on distributions analysis:
- SalePrice follows a lognorm or johnsons SB distribution.
- SalePrice and GrLivArea have a high positive correlation. 
    Two points may be outliers (GrLivArea > 4000, SalePrice < 12.5)
- SalePrice and TotalBsmtSF have a high positive correlation. 
    One data point seems like an outlier (TaoalBsmtSF > 6000, might be information). 
    4 points may be outliers (TotalBsmtSF > 3000)
- OverallQual, GarageCars maybe log
- OverallQual, GarageCars maybe norm
- GrLivArea follows a norm distribution
- TotalBsmtSF follows a norm distribution, slightly left skewed
- YearBuilt follows a norm distribution, slightly right skewed

---
Dealing with missing data:
The following features have missing data:
    'PoolQC'            - NA means no pool 
    'MiscFeature'       - NA means no misc feature
    'Alley'             - NA means no alley access
    'Fence'             - NA means no fence
    'FireplaceQu'       - NA means no fireplace (same as Fireplaces=0)
    'GarageCond'        - NA means no garage
    'GarageType'        - NA means no garage
    'GarageYrBlt'       - NA means no garage
    'GarageFinish'      - NA means no garage
    'GarageQual'        - NA means no garage
    'BsmtExposure'      - NA means no basement (there is one extra basement NA in Exposure and Fin Type 2)
    'BsmtFinType2'      - NA means no basement
    'BsmtFinType1'      - NA means no basement
    'BsmtCond'          - NA means no basement
    'BsmtQual'          - NA means no basement 
    'MasVnrArea'        - NA is missing data (X)
    'MasVnrType'        - NA is missing data (X)
    'Electrical'        - NA is missing data (X)
    'LotFrontage'       - NA is missing data (X)
---
Dealing with outliers:
- GrLivArea : i=523 and i=1298 are outliers
- TotalBsmtSF: i=1298 is an outlier
               i=440, 523, 496, 332 might be outliers
'''


