from joblib import load
from pandas import read_csv, DataFrame, concat

# TODO scale data

def test_model(filename, output_file_name):
    model = load(filename)
    print(model)
    predictions = DataFrame(model.predict(X_test), columns=["target"])
    balance = predictions["target"].value_counts()
    print("Predictions balance : ")
    print(balance)
    submission = DataFrame(concat([test_data["ID_code"], predictions], axis=1), columns=["ID_code", "target"])
    submission.to_csv(output_file_name, sep=",", header=True, index=False)

# Load the test data
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
test_data = read_csv("test.csv", dtype=d)
test_data.info()
X_test = test_data.iloc[:, 1:]
# Choose a model to test
logistic = False
linear = False
quadratic = False
svm = True

if logistic:
    model_file_name = "logistic_model.joblib"
    submission_output_file_name = "logistic_model_submission.csv"
elif linear:
    model_file_name = "lda_model.joblib"
    submission_output_file_name = "lda_model_submission.csv"
elif quadratic:
    model_file_name = "qda_model.joblib"
    submission_output_file_name = "qda_model_submission.csv"
elif svm:
    model_file_name = "svm_rbf_model.joblib"
    submission_output_file_name = "svm_rbf_model_submission.csv"
# Test the model
test_model(model_file_name, submission_output_file_name)


# SVC(C=0.03125, cache_size=200, class_weight={0: 20098, 1: 179902}, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma=0.0001220703125,
#     kernel='rbf', max_iter=-1, probability=False, random_state=None,
#     shrinking=True, tol=0.001, verbose=False)
# Predictions balance :
# 1.0    199804
# 0.0       196


