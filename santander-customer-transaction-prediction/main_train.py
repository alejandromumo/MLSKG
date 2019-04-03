import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from numpy import linalg as LA
import numpy as np
from joblib import dump
import os
import psutil
import platform


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


def print_mem_usage(tab=""):
    memory_usage = process.memory_info().rss / (1024 * 1024)
    print(tab + "Memory usage : {} MB".format(memory_usage))


# Global parameters
process = psutil.Process(os.getpid())
print(platform.architecture())

# Select which model to train
linear = False
quadratic = False
logistic = False
svm = True

# Models generic metrics
best_model = None
best_J = np.inf
best_accuracy = -np.inf

# C = 0.0010000000000000002 # score : 0.041255680820771426,
# C = 0.1                   # score : 0.8362250265325759
# C = 1.0                   # use default

gamma_values = [1]
if logistic:       # Try three C values if training a logistic regression classifier model. Gamma is irrelevant
    C_values = [0.0010000000000000002, 0.1, 1]
elif svm:          # Try C and gamma values power of 2
    C_values = [2**x for x in range(-10, 10)]
#    gamma_values = [2**x for x in range(-10, 5)]
else:              # Otherwise C and gamma are irrelevant
    C_values = [1]

# Matrix containing the mean score and accuracy for each combination of C and gamma (both ignored when not used)
score_matrix = []

# Logistic regression classifier specific parameters
solver = 'lbfgs'

# SVM classifier specific parameters
kernel = "linear"

# Load data set and analyze it
# Force data types to optimize memory usage
d = {'ID_code' : 'object', 'target' : 'int32', 'var_0' : 'float32',
     'var_1' : 'float32', 'var_2' : 'float32', 'var_3' : 'float32', 'var_4' : 'float32', 'var_5' : 'float32',
     'var_6' : 'float32', 'var_7' : 'float32', 'var_8' : 'float32', 'var_9' : 'float32', 'var_10' : 'float32',
     'var_11' : 'float32', 'var_12' : 'float32', 'var_13' : 'float32', 'var_14' : 'float32', 'var_15' : 'float32',
     'var_16' : 'float32', 'var_17' : 'float32', 'var_18' : 'float32', 'var_19' : 'float32', 'var_20' : 'float32',
     'var_21' : 'float32', 'var_22' : 'float32', 'var_23' : 'float32', 'var_24' : 'float32', 'var_25' : 'float32',
     'var_26' : 'float32', 'var_27' : 'float32', 'var_28' : 'float32', 'var_29' : 'float32', 'var_30' : 'float32',
     'var_31' : 'float32', 'var_32' : 'float32', 'var_33' : 'float32', 'var_34' : 'float32', 'var_35' : 'float32',
     'var_36' : 'float32', 'var_37' : 'float32', 'var_38' : 'float32', 'var_39' : 'float32', 'var_40' : 'float32',
     'var_41' : 'float32', 'var_42' : 'float32', 'var_43' : 'float32', 'var_44' : 'float32', 'var_45' : 'float32',
     'var_46' : 'float32', 'var_47' : 'float32', 'var_48' : 'float32', 'var_49' : 'float32', 'var_50' : 'float32',
     'var_51' : 'float32', 'var_52' : 'float32', 'var_53' : 'float32', 'var_54' : 'float32', 'var_55' : 'float32',
     'var_56' : 'float32', 'var_57' : 'float32', 'var_58' : 'float32', 'var_59' : 'float32', 'var_60' : 'float32',
     'var_61' : 'float32', 'var_62' : 'float32', 'var_63' : 'float32', 'var_64' : 'float32', 'var_65' : 'float32',
     'var_66' : 'float32', 'var_67' : 'float32', 'var_68' : 'float32', 'var_69' : 'float32', 'var_70' : 'float32',
     'var_71' : 'float32', 'var_72' : 'float32', 'var_73' : 'float32', 'var_74' : 'float32', 'var_75' : 'float32',
     'var_76' : 'float32', 'var_77' : 'float32', 'var_78' : 'float32', 'var_79' : 'float32', 'var_80' : 'float32',
     'var_81' : 'float32', 'var_82' : 'float32', 'var_83' : 'float32', 'var_84' : 'float32', 'var_85' : 'float32',
     'var_86' : 'float32', 'var_87' : 'float32', 'var_88' : 'float32', 'var_89' : 'float32', 'var_90' : 'float32',
     'var_91' : 'float32', 'var_92' : 'float32', 'var_93' : 'float32', 'var_94' : 'float32', 'var_95' : 'float32',
     'var_96' : 'float32', 'var_97' : 'float32', 'var_98' : 'float32', 'var_99' : 'float32', 'var_100' : 'float32',
     'var_101' : 'float32', 'var_102' : 'float32', 'var_103' : 'float32', 'var_104' : 'float32', 'var_105' : 'float32',
     'var_106' : 'float32', 'var_107' : 'float32', 'var_108' : 'float32', 'var_109' : 'float32', 'var_110' : 'float32',
     'var_111' : 'float32', 'var_112' : 'float32', 'var_113' : 'float32', 'var_114' : 'float32', 'var_115' : 'float32',
     'var_116' : 'float32', 'var_117' : 'float32', 'var_118' : 'float32', 'var_119' : 'float32', 'var_120' : 'float32',
     'var_121' : 'float32', 'var_122' : 'float32', 'var_123' : 'float32', 'var_124' : 'float32', 'var_125' : 'float32',
     'var_126' : 'float32', 'var_127' : 'float32', 'var_128' : 'float32', 'var_129' : 'float32', 'var_130' : 'float32',
     'var_131' : 'float32', 'var_132' : 'float32', 'var_133' : 'float32', 'var_134' : 'float32', 'var_135' : 'float32',
     'var_136' : 'float32', 'var_137' : 'float32', 'var_138' : 'float32', 'var_139' : 'float32', 'var_140' : 'float32',
     'var_141' : 'float32', 'var_142' : 'float32', 'var_143' : 'float32', 'var_144' : 'float32', 'var_145' : 'float32',
     'var_146' : 'float32', 'var_147' : 'float32', 'var_148' : 'float32', 'var_149' : 'float32', 'var_150' : 'float32',
     'var_151' : 'float32', 'var_152' : 'float32', 'var_153' : 'float32', 'var_154' : 'float32', 'var_155' : 'float32',
     'var_156' : 'float32', 'var_157' : 'float32', 'var_158' : 'float32', 'var_159' : 'float32', 'var_160' : 'float32',
     'var_161' : 'float32', 'var_162' : 'float32', 'var_163' : 'float32', 'var_164' : 'float32', 'var_165' : 'float32',
     'var_166' : 'float32', 'var_167' : 'float32', 'var_168' : 'float32', 'var_169' : 'float32', 'var_170' : 'float32',
     'var_171' : 'float32', 'var_172' : 'float32', 'var_173' : 'float32', 'var_174' : 'float32', 'var_175' : 'float32',
     'var_176' : 'float32', 'var_177' : 'float32', 'var_178' : 'float32', 'var_179' : 'float32', 'var_180' : 'float32',
     'var_181' : 'float32', 'var_182' : 'float32', 'var_183' : 'float32', 'var_184' : 'float32', 'var_185' : 'float32',
     'var_186' : 'float32', 'var_187' : 'float32', 'var_188' : 'float32', 'var_189' : 'float32', 'var_190' : 'float32',
     'var_191' : 'float32', 'var_192' : 'float32', 'var_193' : 'float32', 'var_194' : 'float32', 'var_195' : 'float32',
     'var_196' : 'float32', 'var_197' : 'float32', 'var_198' : 'float32', 'var_199' : 'float32'}
file_name = "scaled_data.csv"
train_data = pd.read_csv(file_name, dtype=d)
train_data.info()
columns = train_data.keys()

# Perform K-cross validation
k = 3
print("Performing k-fold validation with k={}".format(k))
kf = KFold(n_splits=k, shuffle=False)
X = train_data.iloc[:, 2:]
y = train_data["target"]
class_weight = {0: 0.10049, 1: 0.89951}

i = 1
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
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            print_mem_usage(tab*"\t")
            # Instantiate Linear Discriminant Analysis classifier
            if linear:
                current_model = LinearDiscriminantAnalysis(solver='lsqr')
            # Instantiate Logistic Regression Classifier with regularization parameter C.
            elif logistic:
                current_model = LogisticRegression(class_weight=class_weight, solver=solver,
                                                               penalty="l2", max_iter=500, C=C)
            # Instantiate Quadratic Discriminant Analaysis classifier
            elif quadratic:
                current_model = LinearDiscriminantAnalysis(solver='lsqr')
            # Instantiate Support Vector Machine classifier
            elif svm:
                current_model = LinearSVC(C=C, class_weight=class_weight, dual=False)
                # current_model = SVC(C=C, kernel=kernel, gamma=gamma, class_weight=class_weight)
            if current_model is None:
                raise Exception("Current model is None!")
            # Train the model and compute the score for the current split
            print(tab*"\t" + "Model created. Fitting data")
            print_mem_usage(tab*"\t")
            current_model.fit(X_train, y_train)
            print(tab*"\t" + "Data fit. Making predictions")
            print_mem_usage(tab*"\t")
            predictions = current_model.predict(X_test)
            error = weighted_mean_absolute_error(y_true=y_test.values, y_pred=predictions, weights=class_weight)
            accuracy = accuracy_score(y_true=y_test, y_pred=predictions)
            accuracies.append(accuracy)
            coefficients = current_model.coef_
            J = error + (C * LA.norm(coefficients, 2))
            errors.append(J)
            # Keep track of the best score (and model) for future use
            if J < best_J:
                print(tab*"\t" + "Best score found in iteration : {}".format(i))
                best_J = J
                best_model = current_model
            if accuracy > best_accuracy:
                print(tab*"\t" + "Best accuracy found in iteration : {}".format(i))
                best_accuracy = accuracy

            i += 1

        # Compute the mean scores for the current hyper parameters
        mean_accuracy = sum(accuracies) / len(accuracies)
        mean_error = sum(errors) / len(errors)
        # Update the score matrix
        score_matrix.append([C, gamma, mean_error, mean_accuracy])

# Print score matrix
print("\n\nScore Matrix:\n"
      "C, gamma, score, accuracy")
for l in score_matrix:
    print(l)

# Dump best model to test it with test data set
if logistic:
    dump(best_model, "logistic_model.joblib")
if linear:
    dump(best_model, "lda_model.joblib")
if quadratic:
    dump(best_model, "qda_model.joblib")
if svm:
    dump(best_model, "svm_" + kernel + "_model.joblib")


# #### Results ####
#
# SVM - Linear kernel
# Note: Gamma is not used since the model uses a linear kernel
# Score Matrix (C, gamma, score, accuracy):
#  C,                  G,   score,                  accuracy
# [0.0009765625,       1,   0.04203960924753503,    0.7824050116391784]
# [0.001953125,        1,   0.04490776483508835,    0.7824800094641713]
# [0.00390625,         1,   0.051520208299241234,   0.7824550083891647]
# [0.0078125,          1,   0.06559387343915657,    0.7822600065141455]
# [0.015625,           1,   0.09458662906908659,    0.782050006514135]
# [0.03125,            1,   0.1531425348055708,     0.7820150051141264]
# [0.0625,             1,   0.270735380139757,      0.7819400033141136]
# [0.125,              1,   0.5061336323514872,     0.7818850042641156]
# [0.25,               1,   0.9771906459317347,     0.7817850043141109]
# [0.5,                1,   1.9194458576050242,     0.7817400045391096]
# [1,                  1,   3.804069968654497,      0.7817400046141101]
# [2,                  1,   7.573377585855803,      0.7818000043141117]
# [4,                  1,   15.112035442830669,     0.7817450043641091]
# [8,                  1,   30.189372352715008,     0.7817650044141103]
# [16,                 1,   60.34404209285143,      0.7817450044391094]
# [32,                 1,   120.65339018685245,     0.7817450045141099]
# [64,                 1,   241.27209201508128,     0.7817400045391096]
# [128,                1,   482.5095013925805,      0.7817450045141099]
# [256,                1,   964.9843141263276,      0.7817400045391096]
# [512,                1,   1929.9339441608456,     0.7817400045391096]
