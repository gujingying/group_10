import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score

from sklearn.neighbors import KNeighborsClassifier


def cross_validation_KNN(n, k, data, n_neighbors, metric):
    """
    n : # iterations
    k : k-fold size
    data: training data
    n_neighbors: k in knn
    """
    scores = []
    for _ in range(0, n):
        data.sample(frac=1)
        fold = int(data.shape[0] / k)
        for j in range(k):
            test = data[j * fold:j * fold + fold]
            train = data[~data.index.isin(test.index)]
            X_train, y_train = train.drop(
                'Cover_Type', axis=1), train['Cover_Type']
            X_test, y_test = test.drop(
                'Cover_Type', axis=1), test['Cover_Type']

            knn = KNeighborsClassifier(metric=metric, n_neighbors=n_neighbors)
            knn.fit(X_train, y_train)
            predictions = knn.predict(X_test)
            true_values = y_test.to_numpy()
            score = f1_score(true_values, predictions, average='macro')

            scores.append(score)
    return np.array(scores).mean()


class KNN:
    def train(self, x_train, x_test, y_train, y_test):
        figure_folder = Path("Figures/")

        train = pd.DataFrame(x_train)
        train['Cover_Type'] = y_train

        neigh = KNeighborsClassifier()
        neigh.fit(x_train, y_train)
        neigh_val_pred = neigh.predict(x_test)
        f1_score(y_test, neigh_val_pred, average='macro')

        k_values = np.arange(1, 6)
        cross_validation_fold = 10
        scores_1 = []
        scores_2 = []

        for k in k_values:
            # run cross-validation with ecuclidean distance
            score_1 = cross_validation_KNN(1, cross_validation_fold, train, k, 'euclidean')
            scores_1.append(score_1)
            # run cross-validation with manhattan distance
            score_2 = cross_validation_KNN(1, cross_validation_fold, train, k, 'manhattan')
            scores_2.append(score_2)
        print(scores_1)
        print(scores_2)

        fig = plt.figure()
        plt.plot(k_values, scores_1, label='euclidean')
        plt.plot(k_values, scores_2, label='manhattan')
        plt.xlabel('k in kNN')
        plt.ylabel('scores')
        plt.legend()
        fig.suptitle('kNN hyperparameter (k) tuning', fontsize=20)
        plt.savefig(figure_folder /'kNN_hyperparameter_tuning.png')

        # n_neighbors = 1 is a reasonable option
        neigh = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
        neigh.fit(x_train, y_train)
        neigh_val_pred = neigh.predict(x_test)
        f1_knn = f1_score(y_test, neigh_val_pred, average='macro')
        print(f"F1-score of KNN method is {f1_knn}.")

        cm = confusion_matrix(y_test, neigh_val_pred, labels=neigh.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=neigh.classes_)
        disp.plot()
        # plt.show()
        plt.savefig(figure_folder /'kNN_confusion_matrix.png')
