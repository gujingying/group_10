import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score

from sklearn.svm import SVC


def cross_validation_svm(n, k, data, c):
    """
    n : # iterations
    k : k-fold size
    data: training data
    c: penalty C in SVM
    """
    f1_scores = []
    acu_train_scores = []
    acu_test_scores = []
    for _ in range(0, n):
        data.sample(frac=1)
        fold = int(data.shape[0] / k)
        for j in range(k):
            test = data[j * fold:j * fold + fold]
            train = data[~data.index.isin(test.index)]
            X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
            X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

            svm = SVC(kernel="rbf", C=c, class_weight='balanced')
            svm.fit(X_train, y_train)
            acu_train = svm.score(X_train, y_train)
            y_pred = svm.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='macro')
            f1_scores.append(f1)
            acu_train_scores.append(acu_train)
    return np.array(f1_scores).mean(), np.array(acu_train_scores).mean()

class SVM:
    def train(self, x_train, x_test, y_train, y_test):
        con_size = 10
        # Taking only non-categorical values
        from sklearn.preprocessing import StandardScaler
        df_xtrain_svm = pd.DataFrame(x_train)
        df_xtest_svm = pd.DataFrame(x_test)
        xtrain_svm_temp = df_xtrain_svm.iloc[:, :con_size]
        xtest_svm_temp = df_xtest_svm.iloc[:, :con_size]

        # Normalization
        xtrain_svm_temp = xtrain_svm_temp.apply(lambda x: (x - np.mean(x)) / np.std(x, ddof=1))
        xtest_svm_temp = xtest_svm_temp.apply(lambda x: (x - np.mean(x)) / np.std(x, ddof=1))

        c = df_xtrain_svm.shape[1]
        x_train_svm = np.concatenate((xtrain_svm_temp, df_xtrain_svm.iloc[:, con_size:c]), axis=1)  # to be used for SVM
        x_test_svm = np.concatenate((xtest_svm_temp, df_xtest_svm.iloc[:, con_size:c]), axis=1)

        # Normalization of train data for cross_validation
        train_svm = pd.DataFrame(x_train_svm)
        train_svm['Cover_Type'] = y_train

        c_values = [1, 10, 100, 150, 200]
        cross_validation_fold = 10
        f1_scores = []
        acu_train_scores = []

        for c in c_values:
            # run cross-validation with given neighbor size k
            f1, acu_train = cross_validation_svm(1, cross_validation_fold, train_svm, c)
            f1_scores.append(f1)
            acu_train_scores.append(acu_train)
            print(f"The f1-score of C = {c} is {f1}.")

        fig = plt.figure()
        plt.plot(c_values, f1_scores, label='f1_score')
        plt.plot(c_values, acu_train_scores, label='acu_train')
        plt.xlabel('C in SVM')
        plt.ylabel('scores')
        plt.legend()
        fig.suptitle('SVM hyperparameter (C) tuning', fontsize=20)
        figure_folder = Path("Figures/")
        plt.savefig(figure_folder / 'SVM_hyperparameter_tuning.png')

        # Parameters optimized using the code in above cell
        C_opt = 100  # reasonable option
        svm = SVC(C=C_opt, kernel='rbf', class_weight='balanced')
        svm.fit(x_train_svm, y_train)

        y_pred = svm.predict(x_test_svm)
        f1_svm = f1_score(y_test, y_pred, average='macro')
        print(f"F1-score of SVM method is {f1_svm}.")

        cm = confusion_matrix(y_test, y_pred, labels=svm.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm.classes_)
        disp.plot()
        # plt.show()
        plt.savefig(figure_folder / 'SVM_confusion_matrix.png')
