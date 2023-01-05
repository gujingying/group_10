import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# The purpose of this part of the code is to
# find the appropriate hyperparameter: n_component
# to apply to the LDA model
# by calculating the explained variance of the eigenvectors of the train dataset.


def get_class_feature_means(input):
    class_feature_means = pd.DataFrame(columns=list([1, 2, 3, 4, 5, 6, 7]))
    for c, rows in input.groupby('class'):
        class_feature_means[c] = rows.mean()
    class_feature_means = class_feature_means.drop(index='class')
    return class_feature_means


def get_within_class_scatter_matrix(input, class_feature_means, rows_numbers):
    within_class_scatter_matrix = np.zeros((rows_numbers, rows_numbers))
    for c, rows in input.groupby('class'):
        rows = rows.drop(['class'], axis=1)
        s = np.zeros((rows_numbers, rows_numbers))

        for index, row in rows.iterrows():
            x, mc = row.values.reshape(rows_numbers, 1), class_feature_means[c].values.reshape(rows_numbers, 1)

            s += (x - mc).dot((x - mc).T)

        within_class_scatter_matrix += s
    return within_class_scatter_matrix


def get_between_class_scatter_matrix(input, class_feature_means, rows_numbers):
    feature_means = input.drop(columns='class').mean()
    between_class_scatter_matrix = np.zeros((rows_numbers, rows_numbers))
    for c in class_feature_means:
        n = len(input.loc[input['class'] == c].index)

        mc, m = class_feature_means[c].values.reshape(rows_numbers, 1), feature_means.values.reshape(rows_numbers, 1)

        between_class_scatter_matrix += n * (mc - m).dot((mc - m).T)

    return between_class_scatter_matrix


def get_explained_variance(within_class_scatter_matrix, between_class_scatter_matrix):
    eigen_values, eigen_vectors = np.linalg.eig(
        np.linalg.pinv(within_class_scatter_matrix).dot(between_class_scatter_matrix))

    pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]
    pairs = sorted(pairs, key=lambda x: x[0], reverse=True)

    eigen_value_sums = sum(eigen_values)
    print('Explained Variance')
    for i, pair in enumerate(pairs):
        print('Eigenvector {0:}: {1:.2%}'.format(i, (pair[0] / eigen_value_sums).real))

    return pairs, pair[0] / eigen_value_sums


class LDA:
    def train(self, x_train, y_train, x_test, y_test):
        X = pd.DataFrame(x_train)
        df = X.join(pd.Series(y_train, name='class'))
        print(df.shape)
        class_feature_means = get_class_feature_means(df)
        rows_numbers = class_feature_means.shape[0]

        within_class_scatter_matrix = get_within_class_scatter_matrix(df, class_feature_means, rows_numbers)
        between_class_scatter_matrix = get_between_class_scatter_matrix(df, class_feature_means, rows_numbers)
        pairs = get_explained_variance(within_class_scatter_matrix, between_class_scatter_matrix)

        # we can see that the first 2 eigenvector gives the most explanatory power.

        lda = LinearDiscriminantAnalysis(n_components=2)
        lda.fit(x_train, y_train)
        X_new = lda.transform(x_train)

        fig, ax = plt.subplots()
        scatter = ax.scatter(X_new[:, 0], X_new[:, 1], marker='.', c=y_train)
        legend1 = ax.legend(*scatter.legend_elements(), loc="lower right", title="Classes")
        plt.title("LDA dimensional reduction image")
        plt.xlabel('LD1')
        plt.ylabel('LD2')
        #plt.show()
        figure_folder = Path("Figures/")
        plt.savefig(figure_folder /'LDA_dimensional_reduction.png')

        yhat = lda.predict(x_test)
        f1 = f1_score(y_test, yhat, average='macro')
        print(f"F1-score of LDA method is {f1}.")

        cm = confusion_matrix(y_test, yhat, labels=lda.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lda.classes_)
        disp.plot()
        #plt.show()
        plt.savefig(figure_folder /'LDA_confusion_matrix.png')

