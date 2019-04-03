import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv("train_houses.csv")
f = open("data_encode_explained.txt", "w")
# Encode binary features
print("--- Encoding Features ---")
f.write("Encoding\n")
binary_features = ['Street', 'Utilities', 'CentralAir']
for bin_feature in binary_features:
    values = train_data[bin_feature]
    uniques = values.unique()
    dic = dict()
    count = 0
    for u in uniques:
        dic.update({u: count})
        count = 1
    new_values = values.map(dic)
    f.write("Feature {}\n{}\n".format(bin_feature, dic.__str__()))
    train_data[bin_feature] = new_values
    print(values.value_counts())
    print(new_values.value_counts())
train_data.to_csv("encoded_train_houses.csv", na_rep="NA")
f.close()
print(train_data.equals(pd.read_csv("encoded_train_houses.csv")))
