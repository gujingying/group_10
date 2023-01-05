import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier


def rfc_cross_validation(n, k, data, n_estimators, criterion):
    """
    n : # iterations
    k : k-fold size
    data: training data
    n_estimators: the number of trees in the forest.
    criterion: criterion in random forest: {“gini”, “entropy”, “log_loss”}
    """
    scores = []
    for i in range(n):
        data.sample(frac=1)
        fold = int(data.shape[0] / k)
        for j in range(k):
            test = data[j * fold:j * fold + fold]
            train = data[~data.index.isin(test.index)]
            X_train, y_train = train.drop(
                'Cover_Type', axis=1), train['Cover_Type']
            X_test, y_test = test.drop(
                'Cover_Type', axis=1), test['Cover_Type']

            rfc = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion)
            rfc.fit(X_train, y_train)
            predictions = rfc.predict(X_test)
            true_values = y_test.to_numpy()
            score = f1_score(true_values, predictions, average='macro')
            print("The", j, "th cross-validation result of the round", i, "with n_estimators =", n_estimators,
                  "and criterion =", criterion, " is", score)
            scores.append(score)
    return np.array(scores).mean()


class RFC:
    def train(self, x_train, x_test, y_train, y_test):
        # Merging x_train and y_train into train data for cross_validation
        train = pd.DataFrame(x_train)
        train['Cover_Type'] = y_train

        n_estimators = range(100, 210, 10)
        cross_validation_fold = 10
        scores_1 = []
        scores_2 = []

        for n in n_estimators:
            # run cross-validation with given n_estimators n and criterion gini
            score_1 = rfc_cross_validation(1, cross_validation_fold, train, n_estimators=n, criterion="gini")
            scores_1.append(score_1)
            # run cross-validation with given n_estimators n and criterion entropy
            score_2 = rfc_cross_validation(1, cross_validation_fold, train, n_estimators=n, criterion='entropy')
            scores_2.append(score_2)

        fig = plt.figure()
        plt.plot(n_estimators, scores_1, label='gini')
        plt.plot(n_estimators, scores_2, label='entropy')
        plt.xlabel('n_estimators in random forest')
        plt.ylabel('scores')
        plt.legend()
        figure_folder = Path("Figures/")
        fig.suptitle('Random Forest Hyperparameter tuning', fontsize=15)
        plt.savefig(figure_folder / 'Random_Forest_Hyperparameter_tuning.png')

        clf = RandomForestClassifier(n_estimators=160, criterion="entropy")
        fit = clf.fit(x_train, y_train)
        pred = fit.predict(x_test)
        print('F1-score of RandomForestClassifier method is ' + str(f1_score(y_test, pred, average='macro')))

        cm = confusion_matrix(y_test, pred, labels=clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot()
        # plt.show()
        plt.savefig(figure_folder / 'RFC_confusion_matrix.png')
