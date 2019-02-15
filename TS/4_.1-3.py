import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

### Load all files
file_activity_labels = "../UCI HAR Dataset/activity_labels.txt"
file_features = "../UCI HAR Dataset/features.txt"
file_X_train = "../UCI HAR Dataset/train/X_train.txt"
file_Y_train = "../UCI HAR Dataset/train/y_train.txt"
file_X_test = "../UCI HAR Dataset/test/X_test.txt"
file_Y_test = "../UCI HAR Dataset/test/y_test.txt"

activity_labels = pd.read_csv(file_activity_labels, delimiter=" ", header=None, names=['id', 'activity'])
features = pd.read_csv(file_features, delimiter=" ", header=None, names=['id', 'feature'])
X_train = pd.read_csv(file_X_train, delimiter=" ", header=None, skipinitialspace=True)
Y_train = pd.read_csv(file_Y_train, delimiter=" ", header=None, skipinitialspace=True)
X_test = pd.read_csv(file_X_test, delimiter=" ", header=None, skipinitialspace=True)
Y_test = pd.read_csv(file_Y_test, delimiter=" ", header=None, skipinitialspace=True)

# Add labels to the measurements
X_train['label'] = Y_train
X_test['label'] = Y_test

X = pd.concat([X_train, X_test])
Y = pd.concat([Y_train, Y_test])

### 4.1 a
print("Training set :", X_train.shape)
print("    Test set :", X_test.shape)
print()
### 4.1 b
iFeature = 480
print(" feature :", features['feature'][-1+iFeature])
print("    mean : %0.3f" % X.head(10)[-1+iFeature].mean())
print("  median : %0.3f" % X.head(10)[-1+iFeature].median())
print("  stddev : %0.3f" % X.head(10)[-1+iFeature].std())
print()

### 4.2 a
print("Training labels :", Y_train.shape)
print("    Test labels :", Y_test.shape)
print()
### 4.2 b
bar_heights = Y[0].value_counts(normalize=True).sort_index()
plt.bar(bar_heights.index.values, bar_heights, tick_label=activity_labels['activity'])
# plt.show()

### 4.3
feature = 100
plt.clf() 
X_train.groupby('label')[feature].plot.kde()
plt.title(features[features['id'] == 555]['feature'].tolist()[0])
plt.show()
