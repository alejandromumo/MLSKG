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
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.feature_selection import RFE, RFECV
# Other imports
from numpy import linalg as LA
from joblib import dump


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


# Global parameters
process = psutil.Process(os.getpid())
architecture = platform.architecture()
print(architecture)
if architecture[0] != '64bit':
    raise Warning("It is recommended to use the 64-bit version of python's interpreter due to memory constraints")

# Perform cross validation
cv = False

# Select which model to train
linear = False
quadratic = False
logistic = False
svm = True

# Perform feature selection
feature_selection = False

# Perform sub sampling
sub_sampling = False

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
elif svm:
    if cv:          # Try C and gamma values power of 2
        C_values = [2**x for x in range(-5, 15)]      # 11 values or 4   -5 to 15
        gamma_values = [2**x for x in range(-15, 3)]  # 11 values or 4   -15 to 3
    else:           # Use best known values
        C_values = [0.03125]
        gamma_values = [0.0001220703125]
else:              # Otherwise C and gamma are irrelevant
    C_values = [1]

# Matrix containing the mean score and accuracy for each combination of C and gamma (both ignored when not used)
score_matrix = []

# Logistic regression classifier specific parameters
solver = 'lbfgs'

# SVM classifier specific parameters
kernel = "rbf"  # implies O(Nf * Ns**2) complexity, Nf = Number of Features, Ns = Number of Samples
#                   therefore giving : 200 * 200k**2 =  200 * 40 000K = 8 000 000 000K iterations.
#                   Given a CPU (i5 4200M) with 2.50GHz clock (0.4 ns),
#                   Worst case scenario, this takes around 0.4 * 8 000 000 000k = 3 200 000 000 000 ns
#                   = 3200 s = 53 minutes per model. With CV K=3, 11 c_values and gamma_values
#                   this means 19239 minutes (320 hours).
#                   With 4 gamma and c values, K=3 and sample of 10%: 5 minutes per model means 240 minutes (4 hours)
#                   With 4 gamma and c values, K=3 and sample of 5% : 2.5 minutes per model means 120 minutes (2 hours)
cache_size = 1000
nu = 0.2  # used for NuSVC

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
file_name = "scaled_data.csv" # TODO save scaler from train to use in test. Linear > std
train_data = pd.read_csv("C:/Users/ManuelCosta/Documents/GitHub/MLSKG/santander-customer-transaction-prediction/scaled_data.csv",
                         dtype=d, lineterminator='\n')
train_data = train_data.iloc[:, 1:].values
# train_data = np.loadtxt(file_name, delimiter=',')

# Compute data set balance
target_count = list()
target_count.append(train_data[train_data[:,0]==0, :].shape[0])
target_count.append(train_data[train_data[:,0]==1, :].shape[0])
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
class_weight = {0: target_count[1], 1: target_count[0]} # TODO
# Class 0: 179902  (Negative)
# Class 1: 20098   (Positive)
# Proportion: 8.95 : 1

# Setup K-cross validation
k = 3
print("Performing k-fold validation with k={}".format(k))
kf = KFold(n_splits=k, shuffle=False)
# Perform sub-sampling
if sub_sampling:
    if cv:
        sample_size = 0.015  # TODO increase
    else:
        sample_size = 0.5
    target_count_0 = int((sample_size * target_count[0]))
    target_count_1 = int((sample_size * target_count[1]))
    data_0 = train_data[train_data[:, 0] == 0, 0:]
    data_1 = train_data[train_data[:, 0] == 1, 0:]
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
else:
    X = train_data[:, 1:]
    y = train_data[:, 0]

print_mem_usage()

# Perform feature selection using a LinearSVM model before training a final model
if feature_selection:
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
    print_mem_usage()

# Train models
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
            X_train, X_test = X[train_index,:], X[test_index,:]
            y_train, y_test = y[train_index], y[test_index]
            print_mem_usage(tab*"\t")
            # Instantiate Linear Discriminant Analysis classifier
            if linear:
                print(tab*"\t" + "Training a linear model")
                current_model = LinearDiscriminantAnalysis(solver='lsqr')
            # Instantiate Logistic Regression Classifier with regularization parameter C.
            elif logistic:
                print(tab*"\t" + "Training a logistic model with solver = {}".format(solver))
                current_model = LogisticRegression(class_weight=class_weight, solver=solver,
                                                   penalty="l2", max_iter=500, C=C)
            # Instantiate Quadratic Discriminant Analaysis classifier
            elif quadratic:
                print(tab*"\t" + "Training a quadratic model")
                current_model = LinearDiscriminantAnalysis(solver='lsqr')
            # Instantiate Support Vector Machine classifier
            elif svm:
                print(tab*"\t" + "Training a SVM model")
                # current_model = LinearSVC(C=C, class_weight=class_weight, dual=False)
                current_model = SVC(C=C, kernel=kernel, gamma=gamma, class_weight=class_weight, cache_size=cache_size)
                # current_model = NuSVC(gamma=gamma, kernel=kernel, verbose=True,
                #  class_weight=class_weight, nu=nu, cache_size=cache_size)
            if current_model is None:
                raise Exception("Current model is None!")
            print(tab*"\t" + "Model created. Fitting data")
            print_mem_usage(tab*"\t")
            # Train the model without feature selection and compute the score for the current split
            current_model.fit(X_train, y_train)
            print(tab*"\t" + "Data fit. Making predictions")
            print_mem_usage(tab*"\t")
            predictions = current_model.predict(X_test)
            error = weighted_mean_absolute_error(y_true=y_test, y_pred=predictions, weights=class_weight)
            if not svm or kernel != "rbf":
                accuracy = accuracy_score(y_true=y_test, y_pred=predictions)
                coefficients = current_model.coef_
                J = error + (C * LA.norm(coefficients, 2))
            else:
                accuracy = balanced_accuracy_score(y_true=y_test, y_pred=predictions)
                J = 1/accuracy
            accuracies.append(accuracy)
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

# Score Matrix: (feature selection using LinearSVM)
# C, gamma, score, accuracy
# [0.0009765625,        1,          0.05688709738789968,            0.6966750121848947]
# [0.001953125,         1,          0.05922842091341488,            0.69699000610988]
# [0.00390625,          1,          0.06435836744396237,            0.6973700051098941]
# [0.0078125,           1,          0.0750543526793274,             0.6975600046849014]
# [0.015625,            1,          0.09697642311425696,            0.6976300031348973]
# [0.03125,             1,          0.14105116231382198,            0.6978150026599041]
# [0.0625,              1,          0.22942528213342608,            0.6976900032849009]
# [0.125,               1,          0.4063305208492434,             0.6976200031098965]
# [0.25,                1,          0.76007741763514,               0.6976700027098971]
# [0.5,                 1,          1.4676113848622625,             0.6977300024848989]
# [1,                   1,          2.882727096242484,              0.6977450021098978]

# Optimal number of features obtained with CV, 5 splits using LinearSVM : 190
# Score Matrix: (feature selection using LinearSVM and CV)
# C, gamma, score, accuracy
# [0.0009765625, 1, 0.04293735099850637, 0.7776600138389522]
# [0.001953125, 1, 0.04572953495568621, 0.7778850108389485]
# [0.00390625, 1, 0.052293495968186464, 0.7776400082389232]
# [0.0078125, 1, 0.06617359831964358, 0.7778600072889295]
# [0.015625, 1, 0.09480323165481463, 0.7778200065889239]
# [0.03125, 1, 0.15268937003120117, 0.7776550072639191]
# [0.0625, 1, 0.268735136370499, 0.7780650065639362]
# [0.125, 1, 0.5012714764572018, 0.7780350061139324]
# [0.25, 1, 0.9666517545595849, 0.7778450072139284]
# [0.5, 1, 1.897507632706983, 0.7779700079389382]
# [1, 1, 3.759412643016182, 0.7779500078139364]


# Score Matrix: (Sub sampling, 10%, SVM with radial kernel)
# Class 0: 17931
# Class 1: 2069
# Proportion: 8.67 : 1
# C,                        gamma,                      score,                          accuracy
# [0.0009765625,            0.0009765625,               1.2946340207694857,             0.7724321660663679]
# [0.0009765625,            0.0078125,                  1.285941775818993,              0.7776990297872955]
# [0.0009765625,            0.0625,                     1.3672546014705658,             0.736353682248219]
# [0.0009765625,            0.5,                        1.5112685445220968,             0.6809176949630351]
# [0.0078125,               0.0009765625,               1.2866190743230976,             0.7772357466880454]
# [0.0078125,               0.0078125,                  1.3318948652788947,             0.7517210264740841]
# [0.0078125,               0.0625,                     1.3983095370905951,             0.7188574134285932]
# [0.0078125,               0.5,                        1.5345597462371188,             0.6677954933483159]
# [0.0625,                  0.0009765625,               1.277030492155242,              0.7830920388628085]
# [0.0625,                  0.0078125,                  1.3998456717205343,             0.7199373139895336]
# [0.0625,                  0.0625,                     1.443610074718355,              0.6976682717722263]
# [0.0625,                  0.5,                        1.5685351494579385,             0.6519036371060407]
# [0.5,                     0.0009765625,               1.3793140852766694,             0.7250769092119761]
# [0.5,                     0.0078125,                  1.4508074646700606,             0.6910090601829157]
# [0.5,                     0.0625,                     1.4775846033513724,             0.678382769234481]
# [0.5,                     0.5,                        1.5940160459327017,             0.6374395102027317]


# Score Matrix: (Sub sampling, 14%, SVM with radial kernel)
# Class 0: 25226
# Class 1: 2774# C, gamma, score, accuracy
# Proportion: 9.09 : 1
# [0.0009765625, 0.0009765625, 1.2947498080770152, 0.772362680656161]
# [0.0009765625, 0.0078125, 1.2830658882045374, 0.7795053699749644]
# [0.0009765625, 0.0625, 1.3744954346559455, 0.7337669636866015]
# [0.0009765625, 0.5, 1.5086737423613445, 0.6811444915747548]
# [0.0078125, 0.0009765625, 1.2861411193319638, 0.7775653065806264]
# [0.0078125, 0.0078125, 1.3289079722169763, 0.7533124088814502]
# [0.0078125, 0.0625, 1.4049441463362056, 0.7163506599778374]
# [0.0078125, 0.5, 1.5315102761215396, 0.6680822637931816]
# [0.0625, 0.0009765625, 1.2719297204236353, 0.7863006259084658]
# [0.0625, 0.0078125, 1.411267079592056, 0.715811115368956]
# [0.0625, 0.0625, 1.459850217919592, 0.6913497976361747]
# [0.0625, 0.5, 1.5726898298090795, 0.6493316170369345]
# [0.5, 0.0009765625, 1.3739760562545724, 0.7278417695868228]
# [0.5, 0.0078125, 1.4605379493804829, 0.6872800347345903]
# [0.5, 0.0625, 1.49269746444521, 0.6723290772132642]
# [0.5, 0.5, 1.597325264703293, 0.6350660767197517]


# Score Matrix: (sub sampling, 1.5% SVM with radial kernel
# C, gamma, score, accuracy
# [0.03125, 3.0517578125e-05, 1.4246093448854493,                               0.702098830043314]
# [0.03125, 6.103515625e-05, 1.4319628020443187,                                0.6986555434420367]
# [0.03125, 0.0001220703125, 1.432171076318634,                                 0.6986716180557065]
# [0.03125, 0.000244140625, 1.4412466208285382,                                 0.6943708329798115]
# [0.03125, 0.00048828125, 1.4388958332948774,                                  0.6954568095760902]
# [0.03125, 0.0009765625, 1.443261183650623,                                    0.6934238280402576]
# [0.03125, 0.001953125, 1.4453160751966836,                                    0.6924157449920492]
# [0.03125, 0.00390625, 1.4519816097526588,                                     0.6893211276686725]
# [0.03125, 0.0078125, 1.4653199321136516,                                      0.6834377496232759]
# [0.03125, 0.015625, 1.476663363352254,                                        0.6784657995413728]
# [0.03125, 0.03125, 1.4872002774856767,                                        0.6738987779221316]
# [0.03125, 0.0625, 1.4983502871927554,                                         0.6691950833758757]
# [0.03125, 0.125, 1.5134184907421766,                                          0.6631223492685755]
# [0.03125, 0.25, 1.541038782157398,                                            0.6533489201304462]
# [0.03125, 0.5, 1.5716361966802384,                                            0.6431256587884164]
# [0.03125, 1, 1.5984089343877235,                                              0.6341803051141405]
# [0.03125, 2, 1.622031938247269,                                               0.6262873459897792]
# [0.03125, 4, 1.6430301639001985,                                              0.6192713823236805]
# [0.0625, 3.0517578125e-05, 1.4397106268801767,                                0.695028298047529]
# [0.0625, 6.103515625e-05, 1.438809219495641,                                  0.695597743673563]
# [0.0625, 0.0001220703125, 1.4504641901408726,                                 0.6900444648614703]
# [0.0625, 0.000244140625, 1.4456085242530279,                                  0.6922975157316421]
# [0.0625, 0.00048828125, 1.4483148740297895,                                   0.6910368628066412]
# [0.0625, 0.0009765625, 1.4513173424857295,                                    0.6895671835921369]
# [0.0625, 0.001953125, 1.4511697170514855,                                     0.6896530165782511]
# [0.0625, 0.00390625, 1.4614129864299918,                                      0.6850067609234888]
# [0.0625, 0.0078125, 1.4736533290753813,                                       0.6796234249503085]
# [0.0625, 0.015625, 1.4841634206178105,                                        0.6750329073357021]
# [0.0625, 0.03125, 1.4940185113634554,                                         0.6707779668260674]
# [0.0625, 0.0625, 1.5046003349140527,                                          0.6663343398711502]
# [0.0625, 0.125, 1.5191877655618355,                                           0.6604816629565212]
# [0.0625, 0.25, 1.5463959659185098,                                            0.6508968542692529]
# [0.0625, 0.5, 1.5766362348572758,                                             0.640837063984636]
# [0.0625, 1, 1.603096470178696,                                                0.6320347474855963]
# [0.0625, 2, 1.6264437366387727,                                               0.6242679976335024]
# [0.0625, 4, 1.6471968623810633,                                               0.6173642199871968]
# [0.125, 3.0517578125e-05, 1.4341115821291106,                                 0.6979696867392345]
# [0.125, 6.103515625e-05, 1.4555780585880116,                                  0.6877428289068508]
# [0.125, 0.0001220703125, 1.4492767200240793,                                  0.6906038022116275]
# [0.125, 0.000244140625, 1.451164145983456,                                    0.6897462815491154]
# [0.125, 0.00048828125, 1.4556241593823676,                                    0.6876335223124237]
# [0.125, 0.0009765625, 1.4546021898307733,                                     0.6880448220597803]
# [0.125, 0.001953125, 1.4611283253826437,                                      0.6850264755124487]
# [0.125, 0.00390625, 1.475331302039552,                                        0.6788052024826429]
# [0.125, 0.0078125, 1.486025165172768,                                         0.6741109285584455]
# [0.125, 0.015625, 1.4952980731054586,                                         0.6700716605830254]
# [0.125, 0.03125, 1.5041409227158629,                                          0.6662677425054522]
# [0.125, 0.0625, 1.5138792119870927,                                           0.6621999675772529]
# [0.125, 0.125, 1.5277528828600264,                                            0.656665319300616]
# [0.125, 0.25, 1.5543492891239727,                                             0.6473531065887695]
# [0.125, 0.5, 1.5840593365157078,                                              0.6375295661495183]
# [0.125, 1, 1.6100556279834761,                                                0.6289339682651733]
# [0.125, 2, 1.6329935322197422,                                                0.6213496171907513]
# [0.125, 4, 1.6533827804297567,                                                0.6146079717912651]
# [0.25, 3.0517578125e-05, 1.4661403801227575,                                  0.6825165448284197]
# [0.25, 6.103515625e-05, 1.4527396901581826,                                   0.6887718465178695]
# [0.25, 0.0001220703125, 1.4558579111943701,                                   0.6873976671390235]
# [0.25, 0.000244140625, 1.4598923518346087,                                    0.6854511198078499]
# [0.25, 0.00048828125, 1.4621605729748117,                                     0.6843721531377244]
# [0.25, 0.0009765625, 1.4595117277090734,                                      0.6856371579984051]
# [0.25, 0.001953125, 1.4703400558239585,                                       0.6807685368850438]
# [0.25, 0.00390625, 1.483334966564147,                                         0.6751027576717589]
# [0.25, 0.0078125, 1.493139533639075,                                          0.6708198665043262]
# [0.25, 0.015625, 1.5017010047251347,                                          0.6671097047343182]
# [0.25, 0.03125, 1.5099617696428411,                                           0.6635750553702638]
# [0.25, 0.0625, 1.519214988336823,                                             0.6597316710366635]
# [0.25, 0.125, 1.5326782148751619,                                             0.6543868917246873]
# [0.25, 0.25, 1.5589228117094558,                                              0.645237423839693]
# [0.25, 0.5, 1.588327957595492,                                                0.6355549289170467]
# [0.25, 1, 1.6140574602457738,                                                 0.6270827458597313]
# [0.25, 2, 1.6367599625842577,                                                 0.6196072902209235]
# [0.25, 4, 1.65693996466291,                                                   0.6129624407642056]
# [0.5, 3.0517578125e-05, 1.43857937372391,                                     0.6953993798513736]
# [0.5, 6.103515625e-05, 1.44780863601481,                                      0.6910129158143491]
# [0.5, 0.0001220703125, 1.4566095919861957,                                    0.6868447760362151]
# [0.5, 0.000244140625, 1.461399960705216,                                      0.6846685099287834]
# [0.5, 0.00048828125, 1.4597738178317508,                                      0.6854311088246675]
# [0.5, 0.0009765625, 1.466606161245952,                                        0.682315040417438]
# [0.5, 0.001953125, 1.4821932705968233,                                        0.675539802313167]
# [0.5, 0.00390625, 1.493706529490404,                                          0.6705276149213667]
# [0.5, 0.0078125, 1.5023587006846366,                                          0.6667530729484221]
# [0.5, 0.015625, 1.5099982550661402,                                           0.6634495905340044]
# [0.5, 0.03125, 1.5175047244983007,                                            0.6602476788245241]
# [0.5, 0.0625, 1.526129363620994,                                              0.6566815758697354]
# [0.5, 0.125, 1.5390607151374738,                                              0.6515714192629076]
# [0.5, 0.25, 1.5648494190958884,                                               0.6426230565537546]
# [0.5, 0.5, 1.5938594578228291,                                                0.6331148527835043]
# [0.5, 1, 1.6192432417089024,                                                  0.6247951744845353]
# [0.5, 2, 1.6416406980789668,                                                  0.6174542818677979]
# [0.5, 4, 1.661549548185691,                                                   0.6109290439862536]
# [1, 3.0517578125e-05, 1.4553429766566108,                                     0.6875404461093996]
# [1, 6.103515625e-05, 1.4623662945205866,                                      0.6841075671028715]
# [1, 0.0001220703125, 1.4726902468842673,                                      0.6794984308252661]
# [1, 0.000244140625, 1.4707396098521022,                                       0.680350328027707]
# [1, 0.00048828125, 1.4673859217256153,                                        0.6819172818121823]
# [1, 0.0009765625, 1.4787783034489228,                                         0.6768262012268376]
# [1, 0.001953125, 1.492492946576426,                                           0.6708879359637943]
# [1, 0.00390625, 1.5027187459725564,                                           0.6664572318656655]
# [1, 0.0078125, 1.5103695597798832,                                            0.6631349546766878]
# [1, 0.015625, 1.517208028251862,                                          0.6601932840894436]
# [1, 0.03125, 1.524059063758048,                                           0.6572874002385597]
# [1, 0.0625, 1.532137507942429,                                            0.6539679871659347]
# [1, 0.125, 1.544606694511106,                                             0.6490665681517069]
# [1, 0.25, 1.5699992570856895,                                             0.6402971233790682]
# [1, 0.5, 1.598665973279977,                                           0.6309439818204636]
# [1, 1, 1.6237493499499784,                                            0.6227599829566847]
# [1, 2, 1.6458817411293916,                                            0.6155388074886444]
# [1, 4, 1.665554977733314,                                             0.6091199848503863]
# [2, 3.0517578125e-05, 1.4768264380575402,                                             0.6772800092984387]
# [2, 6.103515625e-05, 1.4777665985786783,                                          0.6769050856718767]
# [2, 0.0001220703125, 1.477194438950834,                                           0.6773136330263717]
# [2, 0.000244140625, 1.4744654073466574,                                           0.6784923878720407]
# [2, 0.00048828125, 1.4805647877199766,                                            0.6757458076907548]
# [2, 0.0009765625, 1.4965794398885697,                                             0.6688745735738872]
# [2, 0.001953125, 1.507751063524695,                                           0.6640722551184083]
# [2, 0.00390625, 1.5160695983022914,                                           0.6604935111259528]
# [2, 0.0078125, 1.522236984072981,                                             0.6578338695747209]
# [2, 0.015625, 1.5278887101156502,                                             0.6554223074976734]
# [2, 0.03125, 1.5337687745433097,                                          0.6529501487914958]
# [2, 0.0625, 1.5410380761622524,                                           0.6499921733394596]
# [2, 0.125, 1.552822603637097,                                             0.6453965861580375]
# [2, 0.25, 1.5776283155598239,                                             0.6368892829563753]
# [2, 0.5, 1.6057864278558356,                                          0.6277633307592836]
# [2, 1, 1.6304247761148458,                                            0.6197781225868283]
# [2, 2, 1.6521644951669137,                                            0.6127323506699561]
# [2, 4, 1.6714886898798629,                                            0.6064694422994029]
# [4, 3.0517578125e-05, 1.4824782550015054,                                             0.6748437709126164]
# [4, 6.103515625e-05, 1.4782077564251148,                                          0.6769529511729716]
# [4, 0.0001220703125, 1.474709343676074,                                           0.678631621949347]
# [4, 0.000244140625, 1.4691854044595132,                                           0.6812808553197452]
# [4, 0.00048828125, 1.4817681206214657,                                            0.6756193151694257]
# [4, 0.0009765625, 1.4974260883015633,                                             0.668830824922445]
# [4, 0.001953125, 1.5084767621644035,                                          0.664034756274315]
# [4, 0.00390625, 1.5167045846120366,                                           0.6604606996373711]
# [4, 0.0078125, 1.52280141634831,                                          0.6578047038070928]
# [4, 0.015625, 1.5283966991634463,                                             0.655396058306808]
# [4, 0.03125, 1.534230582768579,                                           0.6529262858907092]
# [4, 0.0625, 1.541461400368749,                                            0.6499702990137384]
# [4, 0.125, 1.5532133644430939,                                            0.6453763944727565]
# [4, 0.25, 1.5779911648796783,                                             0.6368705335343285]
# [4, 0.5, 1.606125087221033,                                           0.6277458312987066]
# [4, 1, 1.6307422692697184,                                            0.6197617168425374]
# [4, 2, 1.6524633122538528,                                            0.612716909969447]
# [4, 4, 1.6717709060175276,                                            0.6064548594155889]
# [8, 3.0517578125e-05, 1.4746173606341504,                                             0.6784872286189799]
# [8, 6.103515625e-05, 1.4679823216983399,                                          0.6817051451763351]
# [8, 0.0001220703125, 1.4638943258540198,                                          0.6836442555142113]
# [8, 0.000244140625, 1.4724767715068505,                                           0.679686239084692]
# [8, 0.00048828125, 1.4917569805642672,                                            0.6712998794896959]
# [8, 0.0009765625, 1.5057501382538978,                                             0.6652312951893369]
# [8, 0.001953125, 1.5156116621235476,                                          0.6609494450745079]
# [8, 0.00390625, 1.5229476220762876,                                           0.65776105233754]
# [8, 0.0078125, 1.5283507829832,                                           0.655405017318354]
# [8, 0.015625, 1.5333911291348472,                                             0.6532363404669431]
# [8, 0.03125, 1.5387709736516706,                                          0.6509629060362865]
# [8, 0.0625, 1.5456234253449166,                                           0.6481705341471843]
# [8, 0.125, 1.5570552336518637,                                            0.6437150730574758]
# [8, 0.25, 1.5815586148592504,                                             0.635327877934425]
# [8, 0.5, 1.609454707201967,                                           0.6263060194054634]
# [8, 1, 1.633863788001844,                                             0.6184118931926219]
# [8, 2, 1.6554012122370296,                                            0.611446487710703]
# [8, 4, 1.6745455893349726,                                            0.6052550161712195]
# [16, 3.0517578125e-05, 1.4785296984804661,                                            0.6771723242440837]
# [16, 6.103515625e-05, 1.474806765970954,                                          0.6790080018790844]
# [16, 0.0001220703125, 1.4648984145532213,                                             0.68345669684811]
# [16, 0.000244140625, 1.4840312282879953,                                          0.6748510610495172]
# [16, 0.00048828125, 1.503413941228552,                                            0.6664632673125018]
# [16, 0.0009765625, 1.515464272140802,                                             0.6612007850416751]
# [16, 0.001953125, 1.523938062598037,                                          0.6574947220907978]
# [16, 0.00390625, 1.5302332224914659,                                          0.6547381697267937]
# [16, 0.0078125, 1.534826872241136,                                            0.6527180105532461]
# [16, 0.015625, 1.5392196094669897,                                            0.6508180343783461]
# [16, 0.03125, 1.5440695921354366,                                             0.6487644459557437]
# [16, 0.0625, 1.5504804922883686,                                          0.6461552790733535]
# [16, 0.125, 1.561538680061204,                                            0.6418548376047088]
# [16, 0.25, 1.585721815096495,                                             0.63360051644257]
# [16, 0.5, 1.6133403607567285,                                             0.6246938153463987]
# [16, 1, 1.637506588209433,                                            0.6169004518872487]
# [16, 2, 1.6588297300794663,                                           0.6100239547174106]
# [16, 4, 1.6777836339639405,                                           0.6039115127886655]
# [32, 3.0517578125e-05, 1.468917362741634,                                             0.681510268541652]
# [32, 6.103515625e-05, 1.4813376687362039,                                             0.6763093525318572]
# [32, 0.0001220703125, 1.4851333289034454,                                             0.6742184823356869]
# [32, 0.000244140625, 1.5032309858950896,                                          0.6662665562393204]
# [32, 0.00048828125, 1.5187737473142278,                                           0.6595956634643444]
# [32, 0.0009765625, 1.5282641105455317,                                            0.6554777818348773]
# [32, 0.001953125, 1.5349093526592337,                                             0.6525892907706854]
# [32, 0.00390625, 1.539833101295013,                                           0.6504459173216953]
# [32, 0.0078125, 1.5433600978442892,                                           0.6489026750820476]
# [32, 0.015625, 1.5468995125098275,                                            0.6473842324542673]
# [32, 0.03125, 1.55105132217438,                                           0.6456428078429449]
# [32, 0.0625, 1.5568804114907335,                                          0.6432937774699545]
# [32, 0.125, 1.567446297786464,                                            0.6392134515092637]
# [32, 0.25, 1.5912074601270934,                                            0.6311478007825138]
# [32, 0.5, 1.6184602961186205,                                             0.6224046140636795]
# [32, 1, 1.6423065276112068,                                           0.6147543256846996]
# [32, 2, 1.6633473201046651,                                           0.6080040712326584]
# [32, 4, 1.6820502467655172,                                           0.6020038450530663]
# [64, 3.0517578125e-05, 1.4737105078661417,                                            0.6788149287492423]
# [64, 6.103515625e-05, 1.4918064357643195,                                             0.6710084433053054]
# [64, 0.0001220703125, 1.500291567817404,                                          0.6671569345599987]
# [64, 0.000244140625, 1.5145996650805584,                                          0.6609703954075542]
# [64, 0.00048828125, 1.5278686906626031,                                           0.6553587347989315]
# [64, 0.0009765625, 1.535843230002511,                                             0.6519470079470332]
# [64, 0.001953125, 1.5414057407652162,                                             0.6495629131525333]
# [64, 0.00390625, 1.5455174408877477,                                          0.6477978369058123]
# [64, 0.0078125, 1.548412844148942,                                            0.6465488258234848]
# [64, 0.015625, 1.551446984184015,                                             0.6452657681215609]
# [64, 0.03125, 1.5551853873327324,                                             0.6437169311768481]
# [64, 0.0625, 1.5606699712192231,                                          0.6415283905260325]
# [64, 0.125, 1.5709443529204545,                                           0.6375838635610279]
# [64, 0.25, 1.5944556541800845,                                            0.6296346119734377]
# [64, 0.5, 1.6214919439014122,                                             0.6209923045085419]
# [64, 1, 1.645148697407574,                                            0.613430285476758]
# [64, 2, 1.6660223034424224,                                           0.6067579157428311]
# [64, 4, 1.6845766199178434,                                           0.6008269204237848]
# [128, 3.0517578125e-05, 1.5038408602436395,                                           0.6652721221703355]
# [128, 6.103515625e-05, 1.5034310908501076,                                            0.6654633696826]
# [128, 0.0001220703125, 1.5127955639914008,                                            0.6614245645323009]
# [128, 0.000244140625, 1.5239776622110561,                                             0.6566711178867809]
# [128, 0.00048828125, 1.5353710883670009,                                          0.6519193127823129]
# [128, 0.0009765625, 1.5420952280895093,                                           0.6490808229331843]
# [128, 0.001953125, 1.5467645962683574,                                            0.6471061831406628]
# [128, 0.00390625, 1.5502064394529962,                                             0.6456481981454255]
# [128, 0.0078125, 1.5525808428736074,                                          0.6446380358142522]
# [128, 0.015625, 1.555198183036214,                                            0.6435460571132515]
# [128, 0.03125, 1.5585955681074588,                                            0.6421535575329305]
# [128, 0.0625, 1.5637959702627222,                                             0.640095298019108]
# [128, 0.125, 1.573829890499069,                                           0.6362610089392515]
# [128, 0.25, 1.5971350819316552,                                           0.6284062469675025]
# [128, 0.5, 1.6239927431362113,                                            0.6198458305030023]
# [128, 1, 1.6474931966901982,                                          0.6123554660965647]
# [128, 2, 1.6682288910025396,                                          0.6057463210320608]
# [128, 4, 1.6866606192801763,                                          0.5998715254191685]
# [256, 3.0517578125e-05, 1.5242580219839743,                                           0.6566170537273611]
# [256, 6.103515625e-05, 1.51832837728078,                                          0.6591738732241438]
# [256, 0.0001220703125, 1.526557527731605,                                             0.6556616836916622]
# [256, 0.000244140625, 1.5342991350162094,                                             0.6523489572563018]
# [256, 0.00048828125, 1.5436282666111234,                                          0.6484615842779297]
# [256, 0.0009765625, 1.5489762099596114,                                           0.6461993825128649]
# [256, 0.001953125, 1.552662580728445,                                             0.6446363770661034]
# [256, 0.00390625, 1.555367175855573,                                          0.6434871178301861]
# [256, 0.0078125, 1.5571681641203423,                                          0.6427170755340393]
# [256, 0.015625, 1.5593267721582753,                                           0.64181719286106]
# [256, 0.03125, 1.5623488309456963,                                            0.6405818627582109]
# [256, 0.0625, 1.5672364611977734,                                             0.6386545778089484]
# [256, 0.125, 1.57700572828527,                                            0.6349311133606427]
# [256, 0.25, 1.600084074161699,                                            0.6271713439302228]
# [256, 0.5, 1.6267451358842524,                                            0.6186932543348747]
# [256, 1, 1.6500735648914866,                                          0.611274925938945]
# [256, 2, 1.6706574728390462,                                          0.6047293420601835]
# [256, 4, 1.6889542799035437,                                          0.5989110452790621]
# [512, 3.0517578125e-05, 1.5073740986283657,                                           0.6636181929613576]
# [512, 6.103515625e-05, 1.5149907970132561,                                            0.6608623706312028]
# [512, 0.0001220703125, 1.5315117713968887,                                            0.6537216369933255]
# [512, 0.000244140625, 1.5380148177651718,                                             0.6508939222325494]
# [512, 0.00048828125, 1.5466008128102937,                                          0.6472975562589276]
# [512, 0.0009765625, 1.5514533317922534,                                           0.6452293591636966]
# [512, 0.001953125, 1.5547858280135667,                                            0.643804928481102]
# [512, 0.00390625, 1.5572250172300544,                                             0.6427596003183098]
# [512, 0.0078125, 1.5588195786754369,                                          0.6420703933012605]
# [512, 0.015625, 1.5608130452578604,                                           0.641235178851559]
# [512, 0.03125, 1.5636999883089555,                                            0.64005275911321]
# [512, 0.0625, 1.5684750221140942,                                             0.6381695661343643]
# [512, 0.125, 1.5781490152849509,                                          0.6344834102764111]
# [512, 0.25, 1.6011456978042597,                                           0.6267556196377221]
# [512, 0.5, 1.627735984617309,                                             0.6183052449952073]
# [512, 1, 1.6510024855787273,                                          0.6109111671830069]
# [512, 2, 1.6715317511329197,                                          0.604386980878124]
# [512, 4, 1.6897799871810908,                                          0.5985877041626727]
# [1024, 3.0517578125e-05, 1.4946944818853527,                                          0.669626565616416]
# [1024, 6.103515625e-05, 1.5186219263070686,                                           0.6594784701941344]
# [1024, 0.0001220703125, 1.5337860870426498,                                           0.6528610406702006]
# [1024, 0.000244140625, 1.5397205544994927,                                            0.6502484749902057]
# [1024, 0.00048828125, 1.5479654021977505,                                             0.6467811984650527]
# [1024, 0.0009765625, 1.552590489615134,                                           0.6447990610021341]
# [1024, 0.001953125, 1.555760534718893,                                            0.643436101485477]
# [1024, 0.00390625, 1.5580778855972148,                                            0.6424368766971379]
# [1024, 0.0078125, 1.5595776838906905,                                             0.6417835278602189]
# [1024, 0.015625, 1.5614953399515887,                                          0.6409769999546214]
# [1024, 0.03125, 1.5643202562123448,                                           0.639818051025085]
# [1024, 0.0625, 1.5690436010255346,                                            0.6379544170535829]
# [1024, 0.125, 1.5786738573570496,                                             0.6342848111249207]
# [1024, 0.25, 1.601633051156923,                                           0.6265712061399096]
# [1024, 0.5, 1.6281908477464615,                                           0.6181331257305823]
# [1024, 1, 1.6514289197623075,                                             0.6107498053724209]
# [1024, 2, 1.67193310095276,                                           0.604235110938749]
# [1024, 4, 1.6901590397887178,                                             0.5984442714421518]
# [2048, 3.0517578125e-05, 1.5024757794104888,                                          0.6663106535998244]
# [2048, 6.103515625e-05, 1.516335283537165,                                            0.66050453665413]
# [2048, 0.0001220703125, 1.532261658529381,                                            0.6535450849768644]
# [2048, 0.000244140625, 1.538577233114541,                                             0.6507615082202035]
# [2048, 0.00048828125, 1.5470507450897888,                                             0.6471916250490508]
# [2048, 0.0009765625, 1.5518282753584993,                                          0.6451410831554659]
# [2048, 0.001953125, 1.555107208213206,                                            0.64372926333119]
# [2048, 0.00390625, 1.5575062249047387,                                            0.6426933933121368]
# [2048, 0.0078125, 1.559069541052934,                                          0.6420115426291068]
# [2048, 0.015625, 1.561038011397608,                                           0.6411822132466206]
# [2048, 0.03125, 1.5639045029814533,                                           0.640004608563266]
# [2048, 0.0625, 1.5686624938972171,                                            0.6381254281302489]
# [2048, 0.125, 1.57832206616168,                                           0.6344426675033815]
# [2048, 0.25, 1.6013063879040794,                                          0.626717787062766]
# [2048, 0.5, 1.6278859620438075,                                           0.618269934591915]
# [2048, 1, 1.6511430894160695,                                             0.6108780636799204]
# [2048, 2, 1.6716640841563006,                                             0.604355824639925]
# [2048, 4, 1.6899049683698395,                                             0.5985582788265958]
# [4096, 3.0517578125e-05, 1.5022018048594117,                                          0.6662272729735212]
# [4096, 6.103515625e-05, 1.5142895282086952,                                           0.6612973398314358]
# [4096, 0.0001220703125, 1.5308978216437343,                                           0.6540736204284016]
# [4096, 0.000244140625, 1.5375543554503064,                                            0.6511579098088563]
# [4096, 0.00048828125, 1.5462324429584011,                                             0.6475087463199732]
# [4096, 0.0009765625, 1.551146356915676,                                           0.6454053508812346]
# [4096, 0.001953125, 1.554522706690786,                                            0.643955778524706]
# [4096, 0.00390625, 1.5569947860726214,                                            0.6428915941064633]
# [4096, 0.0078125, 1.5586149287577187,                                             0.6421877211129525]
# [4096, 0.015625, 1.560628860331914,                                           0.6413407738820818]
# [4096, 0.03125, 1.5635325474671862,                                           0.6401487545955035]
# [4096, 0.0625, 1.5683215346758057,                                            0.6382575619931332]
# [4096, 0.125, 1.5780073345726844,                                             0.6345646372229671]
# [4096, 0.25, 1.6010141371428694,                                          0.6268310446595241]
# [4096, 0.5, 1.627613194666678,                                            0.6183756416822225]
# [4096, 1, 1.6508873700000108,                                             0.6109771640770836]
# [4096, 2, 1.6714234070588336,                                             0.604449095601961]
# [4096, 4, 1.6896776622222318,                                             0.5986463680685188]
# [8192, 3.0517578125e-05, 1.4892510103542411,                                          0.6719593118742587]
# [8192, 6.103515625e-05, 1.5062255753100604,                                           0.6648815718468936]
# [8192, 0.0001220703125, 1.5255218530446444,                                           0.6564631084387067]
# [8192, 0.000244140625, 1.5335223790009886,                                            0.6529500258165852]
# [8192, 0.00048828125, 1.5430068617989472,                                             0.6489424391261562]
# [8192, 0.0009765625, 1.5484583726161312,                                          0.6466000948863871]
# [8192, 0.001953125, 1.552218720148319,                                            0.6449798448148367]
# [8192, 0.00390625, 1.5549787978479628,                                            0.6437876521103277]
# [8192, 0.0078125, 1.5568229392246888,                                             0.6429842171163875]
# [8192, 0.015625, 1.559016069752187,                                           0.6420576202851733]
# [8192, 0.03125, 1.5620663742128889,                                           0.6408004331437684]
# [8192, 0.0625, 1.5669775425260333,                                            0.6388549339957095]
# [8192, 0.125, 1.576766726434433,                                          0.6351160575330375]
# [8192, 0.25, 1.599862143871636,                                           0.6273430778045894]
# [8192, 0.5, 1.6265380009468604,                                           0.6188535392842834]
# [8192, 1, 1.6498793758876815,                                             0.6114251930790158]
# [8192, 2, 1.670474706717818,                                          0.6048707699567207]
# [8192, 4, 1.688781667455717,                                          0.5990446160702363]
# [16384, 3.0517578125e-05, 1.4987555292136563,                                             0.667829213572494]
# [16384, 6.103515625e-05, 1.5135844110426042,                                          0.6616720862325156]
# [16384, 0.0001220703125, 1.530427743533007,                                           0.6543234513624547]
# [16384, 0.000244140625, 1.5372017968672607,                                           0.6513452830093963]
# [16384, 0.00048828125, 1.5459503960919647,                                            0.6476586448804051]
# [16384, 0.0009765625, 1.5509113178603124,                                             0.6455302663482612]
# [16384, 0.001953125, 1.5543212446433317,                                          0.6440628489250145]
# [16384, 0.00390625, 1.5568185067810987,                                           0.6429852807067332]
# [16384, 0.0078125, 1.558458236054143,                                             0.6422709980909702]
# [16384, 0.015625, 1.5604878368986959,                                             0.6414157231622978]
# [16384, 0.03125, 1.5634043443460788,                                          0.6402168903047907]
# [16384, 0.0625, 1.5682040151481238,                                           0.6383200197266465]
# [16384, 0.125, 1.5778988550086706,                                            0.634622290515441]
# [16384, 0.25, 1.6009134061191423,                                             0.6268845798596784]
# [16384, 0.5, 1.627519179044533,                                           0.6184256078690331]
# [16384, 1, 1.6507992303542496,                                            0.6110240073772185]
# [16384, 2, 1.6713404520981172,                                            0.6044931834138527]
# [16384, 4, 1.689599315870444,                                             0.5986880065575276]