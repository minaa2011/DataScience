import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis

### Load all files
print("Loading total_acc files...")


folder = "../UCI HAR Dataset/"

file_activity_labels = folder + "activity_labels.txt"
file_features = folder + "features.txt"
file_train_x = folder + "train/Inertial Signals/total_acc_x_train.txt"
file_train_y = folder + "train/Inertial Signals/total_acc_y_train.txt"
file_train_z = folder + "train/Inertial Signals/total_acc_z_train.txt"
file_test_x  = folder + "test/Inertial Signals/total_acc_x_test.txt" 
file_test_y  = folder + "test/Inertial Signals/total_acc_y_test.txt"
file_test_z  = folder + "test/Inertial Signals/total_acc_z_test.txt"

activity_labels = pd.read_csv(file_activity_labels, delimiter=" ", header=None, names=['id', 'activity'])
features = pd.read_csv(file_features, delimiter=" ", header=None, names=['id', 'feature'])
train_x = pd.read_csv(file_train_x, delimiter=" ", header=None, skipinitialspace=True)
train_y = pd.read_csv(file_train_y, delimiter=" ", header=None, skipinitialspace=True)
train_z = pd.read_csv(file_train_z, delimiter=" ", header=None, skipinitialspace=True)
test_x  = pd.read_csv(file_test_x,  delimiter=" ", header=None, skipinitialspace=True)
test_y  = pd.read_csv(file_test_y,  delimiter=" ", header=None, skipinitialspace=True)
test_z  = pd.read_csv(file_test_z,  delimiter=" ", header=None, skipinitialspace=True)

### Concatenate train-, and test-sets
print("Concatenating...")
x = pd.concat([train_x, test_x])
y = pd.concat([train_y, test_y])
z = pd.concat([train_z, test_z])

### For each axis, calculate the variance for each row and sum the variances
print("Calculating variances...")
x_var = x.apply(lambda row : row.var(), axis='columns').sum()
y_var = y.apply(lambda row : row.var(), axis='columns').sum()
z_var = z.apply(lambda row : row.var(), axis='columns').sum()
variances = [x_var, y_var, z_var]

print("  Variance x : %0.2f" % x_var)
print("  Variance y : %0.2f" % y_var)
print("  Variance z : %0.2f" % z_var)
print("Greatest variance : %s" % ['x', 'y', 'z'][np.argmax(variances)])



### Load body_acc files for axis with greatest variance
print()
print("Loading body_acc files...")
file_data = ['x', 'y', 'z'][np.argmax(variances)]
file_data_train = folder + "train/Inertial Signals/body_acc_" + file_data + "_train.txt"
file_data_test  = folder + "test/Inertial Signals/body_acc_" + file_data + "_test.txt"
file_labels_train = folder + "train/y_train.txt"
file_labels_test  = folder + "test/y_test.txt"

traindata = pd.read_csv(file_data_train, delimiter=" ", header=None, skipinitialspace=True)
testdata  = pd.read_csv(file_data_test,  delimiter=" ", header=None, skipinitialspace=True)
trainlabels = pd.read_csv(file_labels_train, delimiter=" ", header=None, skipinitialspace=True)
testlabels  = pd.read_csv(file_labels_test,  delimiter=" ", header=None, skipinitialspace=True)

trainlabels['label'] = trainlabels[0].transform(lambda c : activity_labels['activity'][c-1])
testlabels['label']  = testlabels[0].transform(lambda c : activity_labels['activity'][c-1])

### Concatenate train-, and test-set
dataset  = pd.concat([traindata, testdata], ignore_index=True)
labelset = pd.concat([trainlabels, testlabels], ignore_index=True)

### Drop the last half of the columns to solve the overlap problem, allowing retrieval of the original signal
dataset = dataset.loc[:, :63]
### Convert the dataframe to a numpy array
raw_signal = dataset.values
### Flatten the 2D array to a 1D array by concatenating all the rows, effectively retrieving the original signal
raw_signal = raw_signal.flatten()

print()
print("Datapoints expected : 64 * %d = %d" % (len(dataset.index), 64 * len(dataset.index)))
print("Datapoints in raw signal         = %d" % raw_signal.size)

### Add the labels to the dataset. Not sure why at this point, but the exercise says so
dataset['label'] = labelset['label']



############# 4.5 #############
### a
print()
print("Time domain features")
print("     Range: [%0.4f, %0.4f]" % (raw_signal.min(), raw_signal.max()))
print("      Mean: %0.4f" % raw_signal.mean())
print("    stddev: %0.4f" % raw_signal.std())
print("  kurtosis: %0.4f" % kurtosis(raw_signal))

### b
def toSignal(df): return df.iloc[:, :64].values.flatten() # Function to strip label from dataframe, and concatenate rows into raw signal

byLabel = dataset.groupby('label') # Group samples by class
groupActivities = byLabel.groups.keys() # All keys of the groups
activityToSignal = { activity : toSignal(byLabel.get_group(activity)) for activity in groupActivities } # For each group, get the raw signal

### For each activity, get the raw signal and calculate time domain features
rangePerActivity    = { activity : [signal.min(), signal.max()] for activity, signal in activityToSignal.items()}
meanPerActivity     = { activity : signal.mean()                for activity, signal in activityToSignal.items()}
stddevPerActivity   = { activity : signal.std()                 for activity, signal in activityToSignal.items()}
kurtosisPerActivity = { activity : kurtosis(signal)             for activity, signal in activityToSignal.items()}

print()
print("        ", " ".join([activity.ljust(20) for activity in groupActivities]) )
print("   range", " ".join([("[%0.4f, %0.4f]" % (val[0], val[1])).ljust(20) for _, val in rangePerActivity.items()]))
print("    mean", " ".join([("%0.4f" % val).ljust(20)                       for _, val in meanPerActivity.items()]))
print("  stddev", " ".join([("%0.4f" % val).ljust(20)                       for _, val in stddevPerActivity.items()]))
print("kurtosis", " ".join([("%0.4f" % val).ljust(20)                       for _, val in kurtosisPerActivity.items()]))
