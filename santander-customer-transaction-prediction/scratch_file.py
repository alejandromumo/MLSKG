import pandas as pd
from numpy import linalg as LA
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import timeit
import matplotlib.pyplot as plt
import os
import psutil
import platform
from pandas.plotting import scatter_matrix


def print_mem_usage(tab=""):
    memory_usage = process.memory_info().rss / (1024 * 1024)
    print(tab+"Memory usage : {} MB".format(memory_usage))


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


def plot_corr(df):
    scatter_matrix(df, grid=True)


# Process
process = psutil.Process(os.getpid())
architecture = platform.architecture()
print(architecture)
if architecture[0] != '64bit':
    raise Warning("It is recommended to use the 64-bit version of python's interpreter due to memory constraints")

# Load train data set and analyze it
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
print("Loaded data set")
# Calculate data set balance
balance = train_data["target"].value_counts()
print("Data set balance: \n{}".format(balance/len(train_data.target)))
# Data set balance:
# 0    179902  (0.89951)
# 1     20098  (0.10049)
# Data set is unbalanced. We might need to use weights to estimate errors.

# Perform K-Fold to determine alpha parameter
k = 3
k2 = 2
print("Performing k-fold validation with k={}".format(k))
kf = KFold(n_splits=k, shuffle=False)
kf2 = KFold(n_splits=k2)
X = train_data.iloc[:,2:]
y = train_data["target"]

# Perform data analysis of the features (X)
# Covariance  : The sign of the covariance therefore shows the tendency in the linear relationship between the variables
# Correlation : Covariance normalized
data_analysis = True
covariance = False
correlation = not covariance
if data_analysis:
    print("Performing data analysis")
    if correlation:
        pass
        # corr = train_data.corr()
        # sns.heatmap(corr,
        #             xticklabels=corr.columns.values,
        #             yticklabels=corr.columns.values)
        # plt.show()
    else:
        c = train_data.cov()
    histograms = []
    n_bins = 10
    for i in range(20):
        data = X.iloc[:, i*10:i*10 + 10]
        plt.figure(i, figsize=(20, 20))
        fig, ax = plt.subplots(nrows=2, ncols=5)
        plt.hist(data, bins=n_bins)
    plt.show()
    exit(0)

class_weight = {0:0.10049, 1:0.89951}
print_mem_usage()
i = 1
score_matrix = []
scores = []

# Try different alphas
for alpha in [(1/10)**x for x in range(0,10)]:
    tab = 0
    print("\n")
    print(tab*"\t" + "Trying alpha = {}".format(alpha))
    scores = []
    for train_index, test_index in kf.split(X):
        tab = 1
        print(tab*"\t" + "Performing main split number : {}".format(i))
        i += 1
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        print_mem_usage(tab*"\t")
        sub_data = pd.concat([y_train, X_train], axis=1)
        sub_data_X = sub_data.iloc[:,2:]
        sub_data_Y = sub_data["target"]
        X_train = None  # To release some memory and avoid MemoryError !
        X_test = None   #    (only relevant with 32-bit version of python's interpreter)
        y_train = None  #
        y_test = None   #
        # K-split the main training split
        for train2_i, test2_i in kf2.split(sub_data_X):
            XX_train, XX_test = sub_data_X.iloc[train2_i], sub_data_X.iloc[test2_i]
            YY_train, YY_test = sub_data_Y.iloc[train2_i], sub_data_Y.iloc[test2_i]
            # Create the model
            logistic_regression_model = LogisticRegression(class_weight=class_weight, solver='lbfgs',
                                                           penalty="l2", max_iter=500, C=alpha)
            # Train the model
            logistic_regression_model.fit(XX_train, YY_train)
            # Make predictions and compute error score
            predictions = logistic_regression_model.predict(XX_test)
            error = weighted_mean_absolute_error(YY_test.values, predictions, class_weight)
            coefficients = logistic_regression_model.coef_
            # Score = mean weighted absolute error + (alpha * L2-norm of the model's coefficients)
            J = error + (alpha * LA.norm(coefficients,2))
            scores.append(J)
    # Keep track of each alpha's score and average
    alpha_avg_score = sum(scores) / k2
    print(tab*"\t"+"Average score : {}".format(alpha_avg_score))
    score_matrix.append([alpha, scores, alpha_avg_score])

# Print score matrix
print("\n\nScore Matrix:\n"
      "alpha, scores, average_score")
for l in score_matrix:
    print(l)


# #### Results ####
# Score Matrix:
# Note : WITHOUT shuffling the data
# Score Matrix:
# Resumed
# alpha                              average_score
# [1.0,                              30.391091603988436]
# [0.1,                              2.5238620119295536]
# [0.010000000000000002,             0.22621694979152837]
# [0.0010000000000000002,            0.1250574189906519]
# [0.00010000000000000002,           0.15533646186850628]
# [1.0000000000000003e-05,           0.22231576592261532]
# [1.0000000000000004e-06,           0.2558494997083791]
# [1.0000000000000004e-07,           0.26111694402330643]
# [1.0000000000000005e-08,           0.2615611539533349]
# [1.0000000000000005e-09,           0.2716430341353086]
#
#
# Detailed
# alpha,                        scores,                                                                                                                                 average_score
# [1.0,                         [10.418520491985241, 9.891710299562256, 10.417928951042946, 10.079976962760014, 9.892651317666248, 10.081395184960167],                 30.391091603988436]
# [0.1,                         [0.860016885875341, 0.8249538712742579, 0.8595797927108071, 0.8378463944936957, 0.8258961776666737, 0.8394309018383314],                2.5238620119295536]
# [0.010000000000000002,        [0.07702757348501514, 0.07374911774008384, 0.07640309150742548, 0.07413638842215664, 0.07509296964648435, 0.07602475878189124],         0.22621694979152837]
# [0.0010000000000000002,       [0.042746328148604004, 0.0408014865544496, 0.0419120677199067, 0.040325374406915256, 0.04201198110354408, 0.04231760004788413],         0.1250574189906519]
# [0.00010000000000000002,      [0.057172507549016725, 0.05792870054516935, 0.056768465069229494, 0.039845051501098516, 0.057195923580985394, 0.04176227549151309],     0.15533646186850628]
# [1.0000000000000003e-05,      [0.09048583850436952, 0.09122478787462943, 0.09039087592918257, 0.03997770754845952, 0.09040037197592529, 0.04215195001266425],         0.22231576592261532]
# [1.0000000000000004e-06,      [0.09048576877609713, 0.09122471958760492, 0.09039080620091018, 0.0744214471537385, 0.09040030368890078, 0.07477595400950666],          0.2558494997083791]
# [1.0000000000000004e-07,      [0.09048576807810225, 0.09122471890410565, 0.09039080550291531, 0.07966692872609757, 0.09040030300540151, 0.08006536382999056],         0.26111694402330643]
# [1.0000000000000005e-08,      [0.09048576807112402, 0.09122471889726999, 0.09039080549593707, 0.08014649131489612, 0.09040030299856586, 0.08047422112887681],         0.2615611539533349]
# [1.0000000000000005e-09,      [0.09048576807105421, 0.09122471889720167, 0.09039080549586727, 0.09029870473694229, 0.09040030299849754, 0.0904857680710542],          0.2716430341353086]
#