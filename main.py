import argparse
import data_preparation
from linear_discriminant_analysis import LDA
from random_forest_classifier import RFC
from support_vector_machine import SVM
from k_nearest_neighbours import KNN

parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--model', '-m', help='Model selection', default="all")
parser.add_argument('--location', '-l', help='Path of the data file', default="covtype.csv")
args = parser.parse_args()

if __name__ == '__main__':
    try:
        x_train, x_test, y_train, y_test = data_preparation.data_preparation(args.location)
        if args.model == "all":
            linear_discriminant = LDA()
            linear_discriminant.train(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
            random_forest_classifier = RFC()
            random_forest_classifier.train(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
            support_vector_machine = SVM()
            support_vector_machine.train(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
            k_nearest_neighbours = KNN()
            k_nearest_neighbours.train(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        elif args.model == "knn":
            k_nearest_neighbours = KNN()
            k_nearest_neighbours.train(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        elif args.model == "rfc":
            random_forest_classifier = RFC()
            random_forest_classifier.train(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        elif args.model == "svm":
            support_vector_machine = SVM()
            support_vector_machine.train(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        elif args.model == "lda":
            x_train, x_test, y_train, y_test = data_preparation.data_preparation(args.location)
            linear_discriminant = LDA()
            linear_discriminant.train(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        else:
            print("args error")
    except Exception as e:
        print(e)
