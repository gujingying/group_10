import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def train_test_split(x, y, test_ratio, seed):
    if seed:
        np.random.seed(seed)
        # Disordering the indexes of data
    shuffle_indexes = np.random.permutation(len(x))
    # proportional split
    test_size = int(test_ratio * len(x))
    # indexes of train dataset
    test_indexes = shuffle_indexes[:test_size]
    # indexes of test dataset
    train_indexes = shuffle_indexes[test_size:]

    # fetching data
    x_train = x[train_indexes]
    x_test = x[test_indexes]
    y_train = y[train_indexes]
    y_test = y[test_indexes]

    return x_train, x_test, y_train, y_test


def data_preparation(data_location):

    try:
        data = pd.read_csv(data_location)
    except Exception as e:
        print(e)
        return

    # Counts of each cover type
    data.groupby('Cover_Type').count()

    # Randomly select the same amount of data from 7 classes
    subdata = data.groupby('Cover_Type').sample(2000, replace=False, random_state=1)
    subdata.to_csv('subdata.csv')

    pd.set_option('display.max_columns', None)  # See all columns
    subdata.describe()

    # Removing Soil_Type8 and Soil_type 15
    subdata = subdata.drop(['Soil_Type8', 'Soil_Type15'], axis=1)

    # Get feature data and target data
    X = np.array(subdata.iloc[:, :subdata.shape[1] - 1])
    Y = np.array(subdata.iloc[:, -1])

    # Define a train_test_split function to get train data and test data

    x_train, x_test, y_train, y_test = train_test_split(X, Y, 0.3, 10)

    # Merging x_train and y_train into train data for cross_validation
    train = pd.DataFrame(x_train)
    train['Cover_Type'] = y_train

    # Correlation of the features with continuous data
    con_size = 10  # the number of continuous features
    con_feature = pd.DataFrame(x_train).iloc[:, :con_size]
    con_feature.columns = subdata.iloc[:, :con_size].columns

    data_corr = con_feature.corr()

    # Threshold ( only highly correlated ones matter)
    threshold = 0.5
    corr_list = []

    # Sorting out the highly correlated values
    for i in range(0, 10):
        for j in range(i + 1, 10):
            if threshold <= data_corr.iloc[i, j] < 1 \
                    or data_corr.iloc[i, j] < 0 and data_corr.iloc[i, j] <= -threshold:
                corr_list.append([data_corr.iloc[i, j], i, j])

    # Sorting the values
    s_corr_list = sorted(corr_list, key=lambda x: -abs(x[0]))

    # print the higher values
    cols = con_feature.columns  # Get name of the columns
    for v, i, j in s_corr_list:
        print("%s and %s = %.2f" % (cols[i], cols[j], v))

    return x_train, x_test, y_train, y_test
