# Main imports
import pandas as pd
import numpy as np
from numpy.random import shuffle
import matplotlib.pyplot as plt
# Utilities imports
import os
import psutil
import platform
# Sklearn imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.feature_selection import RFE, RFECV
# Other imports
from numpy import linalg as LA
from joblib import dump

# ----                         ----#
# ---- Parameters are          ----#
# ---- initialized here,       ----#
# ---- change only this        ----#
# ---- portion !               ----#
# ----                         ----#

# Select mode
mode = 'evaluation'
# Select which model to train
model = 'svm'
# K-cross validation parameters
k = 10
# Perform feature selection
feature_selection = False
# Perform sub sampling
sub_sampling = False
sample_size = 0.0025
# Logistic regression classifier specific parameters
solver = 'lbfgs'
# SVM classifier specific parameters
cache_size = 1000
kernel = "rbf"
nu = 0.2  # used for NuSVC
# ----                         ----#
# ---- Don't edit after this ! ----#
# ----                         ----#
if mode == 'evaluation':
    C_values = [512]
    gamma_values = [0.000030517578125]
elif mode == 'tuning':
    C_values = [2 ** x for x in range(-5, 15)]  # 11 values or 4   -5 to 15
    gamma_values = [2 ** x for x in range(-15, 3)]  # 11 values or 4   -15 to 3
else:
    raise ValueError("Wrong argument mode!")
hyper_parameters = {'model': 'svm',
                    'kernel': kernel,
                    'cache_size': cache_size,
                    'C_values': C_values,
                    'gamma_values': gamma_values
                    }


def weighted_mean_absolute_error(y_true, y_pred, weights):
    if len(y_true) != len(y_pred):
        raise Exception("Y and predictions must have the same size! Given: y_true :{} \t y_pred : {}".format
                        (len(y_true), len(y_pred)))
    size = len(y_true)
    J = 0
    for i in range(0, size):
        y_t = y_true[i]
        y_p = y_pred[i]
        s = np.abs(y_t - y_p)
        w = weights.get(y_t)
        J += w * s
    J = J / size
    return J


def scorer(estimator, X, y):
    class_weights = estimator.get_params().get('class_weight')
    y_pred = estimator.predict(X)
    size = len(y)
    J = 0
    for i in range(0, size):
        y_t = y[i]
        y_p = y_pred[i]
        s = np.abs(y_t - y_p)
        w = class_weights.get(y_t)
        J += w * s
    J = J / size
    return J


def print_mem_usage(tab=""):
    memory_usage = process.memory_info().rss / (1024 * 1024)
    print(tab + "Memory usage : {} MB".format(memory_usage))



def load_data(file_name="train.csv"):
    # Load data set and analyze it
    # Force data types to optimize memory usage
    d = {'ID_code': 'object', 'target': 'int32', 'var_0': 'float32',
         'var_1': 'float32', 'var_2': 'float32', 'var_3': 'float32', 'var_4': 'float32', 'var_5': 'float32',
         'var_6': 'float32', 'var_7': 'float32', 'var_8': 'float32', 'var_9': 'float32', 'var_10': 'float32',
         'var_11': 'float32', 'var_12': 'float32', 'var_13': 'float32', 'var_14': 'float32', 'var_15': 'float32',
         'var_16': 'float32', 'var_17': 'float32', 'var_18': 'float32', 'var_19': 'float32', 'var_20': 'float32',
         'var_21': 'float32', 'var_22': 'float32', 'var_23': 'float32', 'var_24': 'float32', 'var_25': 'float32',
         'var_26': 'float32', 'var_27': 'float32', 'var_28': 'float32', 'var_29': 'float32', 'var_30': 'float32',
         'var_31': 'float32', 'var_32': 'float32', 'var_33': 'float32', 'var_34': 'float32', 'var_35': 'float32',
         'var_36': 'float32', 'var_37': 'float32', 'var_38': 'float32', 'var_39': 'float32', 'var_40': 'float32',
         'var_41': 'float32', 'var_42': 'float32', 'var_43': 'float32', 'var_44': 'float32', 'var_45': 'float32',
         'var_46': 'float32', 'var_47': 'float32', 'var_48': 'float32', 'var_49': 'float32', 'var_50': 'float32',
         'var_51': 'float32', 'var_52': 'float32', 'var_53': 'float32', 'var_54': 'float32', 'var_55': 'float32',
         'var_56': 'float32', 'var_57': 'float32', 'var_58': 'float32', 'var_59': 'float32', 'var_60': 'float32',
         'var_61': 'float32', 'var_62': 'float32', 'var_63': 'float32', 'var_64': 'float32', 'var_65': 'float32',
         'var_66': 'float32', 'var_67': 'float32', 'var_68': 'float32', 'var_69': 'float32', 'var_70': 'float32',
         'var_71': 'float32', 'var_72': 'float32', 'var_73': 'float32', 'var_74': 'float32', 'var_75': 'float32',
         'var_76': 'float32', 'var_77': 'float32', 'var_78': 'float32', 'var_79': 'float32', 'var_80': 'float32',
         'var_81': 'float32', 'var_82': 'float32', 'var_83': 'float32', 'var_84': 'float32', 'var_85': 'float32',
         'var_86': 'float32', 'var_87': 'float32', 'var_88': 'float32', 'var_89': 'float32', 'var_90': 'float32',
         'var_91': 'float32', 'var_92': 'float32', 'var_93': 'float32', 'var_94': 'float32', 'var_95': 'float32',
         'var_96': 'float32', 'var_97': 'float32', 'var_98': 'float32', 'var_99': 'float32', 'var_100': 'float32',
         'var_101': 'float32', 'var_102': 'float32', 'var_103': 'float32', 'var_104': 'float32', 'var_105': 'float32',
         'var_106': 'float32', 'var_107': 'float32', 'var_108': 'float32', 'var_109': 'float32', 'var_110': 'float32',
         'var_111': 'float32', 'var_112': 'float32', 'var_113': 'float32', 'var_114': 'float32', 'var_115': 'float32',
         'var_116': 'float32', 'var_117': 'float32', 'var_118': 'float32', 'var_119': 'float32', 'var_120': 'float32',
         'var_121': 'float32', 'var_122': 'float32', 'var_123': 'float32', 'var_124': 'float32', 'var_125': 'float32',
         'var_126': 'float32', 'var_127': 'float32', 'var_128': 'float32', 'var_129': 'float32', 'var_130': 'float32',
         'var_131': 'float32', 'var_132': 'float32', 'var_133': 'float32', 'var_134': 'float32', 'var_135': 'float32',
         'var_136': 'float32', 'var_137': 'float32', 'var_138': 'float32', 'var_139': 'float32', 'var_140': 'float32',
         'var_141': 'float32', 'var_142': 'float32', 'var_143': 'float32', 'var_144': 'float32', 'var_145': 'float32',
         'var_146': 'float32', 'var_147': 'float32', 'var_148': 'float32', 'var_149': 'float32', 'var_150': 'float32',
         'var_151': 'float32', 'var_152': 'float32', 'var_153': 'float32', 'var_154': 'float32', 'var_155': 'float32',
         'var_156': 'float32', 'var_157': 'float32', 'var_158': 'float32', 'var_159': 'float32', 'var_160': 'float32',
         'var_161': 'float32', 'var_162': 'float32', 'var_163': 'float32', 'var_164': 'float32', 'var_165': 'float32',
         'var_166': 'float32', 'var_167': 'float32', 'var_168': 'float32', 'var_169': 'float32', 'var_170': 'float32',
         'var_171': 'float32', 'var_172': 'float32', 'var_173': 'float32', 'var_174': 'float32', 'var_175': 'float32',
         'var_176': 'float32', 'var_177': 'float32', 'var_178': 'float32', 'var_179': 'float32', 'var_180': 'float32',
         'var_181': 'float32', 'var_182': 'float32', 'var_183': 'float32', 'var_184': 'float32', 'var_185': 'float32',
         'var_186': 'float32', 'var_187': 'float32', 'var_188': 'float32', 'var_189': 'float32', 'var_190': 'float32',
         'var_191': 'float32', 'var_192': 'float32', 'var_193': 'float32', 'var_194': 'float32', 'var_195': 'float32',
         'var_196': 'float32', 'var_197': 'float32', 'var_198': 'float32', 'var_199': 'float32'}
    train_data = pd.read_csv("C:/Users/ManuelCosta/Documents/GitHub/MLSKG/santander-customer-transaction-prediction/"
                             + file_name,
                             dtype=d, lineterminator='\n')
    train_data = train_data.iloc[:, 1:].values
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    dump(scaler, "train_scaler.joblib")
    return train_data


def sub_sample(sample_size, original_data):
    target_count_0 = int((sample_size * target_count[0]))
    target_count_1 = int((sample_size * target_count[1]))
    data_0 = original_data[original_data[:, 0] == 0, 0:]
    data_1 = original_data[original_data[:, 0] == 1, 0:]
    idx_0 = np.arange(1, target_count_0)
    shuffle(idx_0)
    idx_1 = np.arange(1, target_count_1)
    shuffle(idx_1)
    sample_0 = data_0[idx_0, :]
    sample_1 = data_1[idx_1, :]
    new_data = np.vstack((sample_0, sample_1))
    shuffle(new_data)
    X = new_data[:, 1:]
    y = new_data[:, 0]
    return X,y


def perform_k_cross(X, y, class_weight, k=10, model_properties=None, output_file=None):
    """
    Perform k-cross validation with given arguments. Writes output to given file in file path.
    :param X:
    :param y:
    :param k:
    :param model_properties: dict containing current model properties.
    For svm: kernel, class_weight, cache_size
    For linear: solver
    For logistic: class_weight, solver
    For quadratic: solver
    Properties must be in conformity with sk-learn allowed arguments.
    :param output_file: path to file to store the results for the training session.
    :return:
    """
    score_matrix = []
    C_values = model_properties.get('C_values')
    gamma_values = model_properties.get('gamma_values')
    model = model_properties.get('model')
    kernel = model_properties.get('kernel')
    cache_size = model_properties.get('cache_size')
    kf = KFold(n_splits=k, shuffle=False)
    print("Performing k-fold validation with k={}".format(k))
    for C in C_values:
        # Formatted print
        tab = 1
        print("\n\n")
        print(tab*"\t" + "Trying C = {}".format(C))
        # Lists initialization
        accuracies = []
        errors = []
        current_model = None
        for gamma in gamma_values:
            # Formatted print
            tab = 2
            print("\n")
            print(tab*"\t" + "Trying Gamma = {}".format(gamma))
            for train_index, test_index in kf.split(X):
                print_mem_usage(tab*"\t")
                X_train, X_test = X[train_index,:], X[test_index,:]
                y_train, y_test = y[train_index], y[test_index]
                print_mem_usage(tab*"\t")
                # Instantiate Linear Discriminant Analysis classifier
                if model == 'linear':
                    print(tab*"\t" + "Training a linear model")
                    current_model = LinearDiscriminantAnalysis(solver='lsqr')
                # Instantiate Logistic Regression Classifier with regularization parameter C.
                elif model == 'logistic':
                    print(tab*"\t" + "Training a logistic model with solver = {}".format(solver))
                    current_model = LogisticRegression(class_weight=class_weight, solver=solver,
                                                       penalty="l2", max_iter=500, C=C)
                # Instantiate Quadratic Discriminant Analaysis classifier
                elif model == 'quadratic':
                    print(tab*"\t" + "Training a quadratic model")
                    current_model = LinearDiscriminantAnalysis(solver='lsqr')
                # Instantiate Support Vector Machine classifier
                elif model == 'svm':
                    print(tab*"\t" + "Training a SVM model")
                    # current_model = LinearSVC(C=C, class_weight=class_weight, dual=False)
                    current_model = SVC(C=C, kernel=kernel, gamma=gamma, class_weight=class_weight, cache_size=cache_size)
                    # current_model = NuSVC(gamma=gamma, kernel=kernel, verbose=True,
                    #  class_weight=class_weight, nu=nu, cache_size=cache_size)
                else:
                    raise Exception("Current model is None!")
                print(tab*"\t" + "Model created. Fitting data")
                print_mem_usage(tab*"\t")
                # Train the model without feature selection and compute the score for the current split
                current_model.fit(X_train, y_train)
                print(tab*"\t" + "Data fit. Making predictions")
                print_mem_usage(tab*"\t")
                predictions = current_model.predict(X_test)
                error = weighted_mean_absolute_error(y_true=y_test, y_pred=predictions, weights=class_weight)
                if model != 'svm' or kernel != "rbf":
                    accuracy = accuracy_score(y_true=y_test, y_pred=predictions)
                    coefficients = current_model.coef_
                    J = error + (C * LA.norm(coefficients, 2))
                else:
                    accuracy = balanced_accuracy_score(y_true=y_test, y_pred=predictions)
                    J = 1/accuracy
                accuracies.append(accuracy)
                errors.append(J)
            # Compute the mean scores for the current hyper parameters
            mean_accuracy = sum(accuracies) / len(accuracies)
            mean_error = sum(errors) / len(errors)
            # Update the score matrix
            score_matrix.append([C, gamma, mean_error, mean_accuracy])
    # Print score matrix
    out = open(output_file, 'w')
    print("\n\nScore Matrix:\n"
          "C, gamma, score, accuracy")
    for l in score_matrix:
        print(l)
        out.write(l)
    out.close()


def train_model_to_evaluate(X, Y, class_weight, model_properties=None):
    # Train one single model and save it
    print("Training one single model")
    C_values = model_properties.get('C_values')
    gamma_values = model_properties.get('gamma_values')
    model = model_properties.get('model')
    kernel = model_properties.get('kernel')
    cache_size = model_properties.get('cache_size')
    current_model = SVC(C=C_values[0], kernel=kernel, gamma=gamma_values[0],
                        class_weight=class_weight, cache_size=cache_size)
    current_model.fit(X=X, y=Y)
    # Dump best model to test it with test data set
    if model=='logistic':
        dump(current_model, "logistic_model.joblib")
    if model=='linear':
        dump(current_model, "lda_model.joblib")
    if model=='quadratic':
        dump(current_model, "qda_model.joblib")
    if model=='svm':
        dump(current_model, "svm_" + kernel + "_model.joblib")


def feature_selection(X, y):
    """
    Not used
    :return:
    """
    f_model = LinearSVC(class_weight=class_weight, dual=False)
    # Use custom scorer. Change to return 1/J to have an increasing function
    selector = RFECV(estimator=f_model, step=10, cv=KFold(n_splits=5), scoring=scorer, verbose=1)
    selector.fit(X=X, y=y)
    print("Optimal number of features : %d" % selector.n_features_)
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
    plt.show()
    new_train_data = train_data.loc[:, selector.support_]
    X = new_train_data.iloc[:, 2:]
    y = new_train_data["target"]

# Global parameters
process = psutil.Process(os.getpid())
architecture = platform.architecture()
print(architecture)
if architecture[0] != '64bit':
    raise Warning("It is recommended to use the 64-bit version of python's interpreter due to memory constraints")

# Class 0: 179902  (Negative)
# Class 1: 20098   (Positive)
# Proportion: 8.95 : 1

if __name__ == '__main__':
    # Load data
    train_data = load_data()
    # Compute data set balance
    target_count = list()
    target_count.append(train_data[train_data[:, 0] == 0, :].shape[0])
    target_count.append(train_data[train_data[:, 0] == 1, :].shape[0])
    class_weight = {0: max(0.05, target_count[1] / (target_count[0] + target_count[1])),
                    1: min(0.95, target_count[0] / (target_count[0] + target_count[1]))
                    }
    # Perform sub-sampling
    if sub_sampling:
        X,y = sub_sample(sample_size=sample_size, original_data=train_data)
    else:
        X = train_data[:, 1:]
        y = train_data[:, 0]
    # Train models and find hyper parameters
    if mode == 'tuning':
        perform_k_cross(X=X, y=y, k=k, class_weight=class_weight,
                        model_properties=hyper_parameters, output_file="")
    # Train just one model to be evaluated
    elif mode == 'evaluation':
        train_model_to_evaluate(X=X, Y=y, class_weight=class_weight,
                                model_properties=hyper_parameters)
    else:
        raise ValueError("Wrong mode provided!")

