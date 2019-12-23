import numpy as np
from sklearn.linear_model import LogisticRegression as logreg_clf
from sklearn.tree import DecisionTreeClassifier as tree_clf
from sklearn.ensemble import RandomForestClassifier as forest_clf
from sklearn.metrics import accuracy_score


from data import *


class DataClassifier:
    '''Module for classifying data using various supervised learning methods.'''

    def __init__(self, data_file):
        self.data = DataReader(data_file)


    def get_data_reader(self):
        '''
        Returns DataReader object instance.
        '''
        return self.data


    def fit(self, feature_indices=np.arange(0,8), method="logreg", verbose=False):
        '''
        Fit/predict subsets of data using supervised learning.
        Supported methods: logistic regression, decision tree, random forest.

        Args:
            feature_indices: numpy array of data features to use for predictions
                             (there are eight features in dataset, represented by indices 0-7)
                     method: supervised learning method to use ("logreg", "tree" or "forest")
                    verbose: print training and validation accuracy if toggled
        Returns relevant subsets of data, fitted classifier, and prediction accuracies.
        '''
        assert (method=="logreg" or method=="tree" or method=="forest"),\
            "Failed to initialize classifier: allowed methods are " +\
            "'logreg' (Logistic Regression), 'tree' (Deicision Tree) and 'forest' (Random Forest)."

        # instantiate object that implements selected learning method
        clf = None
        if method == "logreg":
            clf = logreg_clf()
        elif method == "tree":
            clf = tree_clf()
        elif method == "forest":
            clf = forest_clf(n_estimators=100)

        assert (np.all(feature_indices >= 0) and np.all(feature_indices <= 7)),\
            "Failed to classify data: incorrect feature indices specified."

        # get relevant data subsets
        X_train, y_train = self.data.get_train_data()
        X_train = X_train[:, feature_indices]

        X_test, y_test = self.data.get_test_data()
        X_test = X_test[:, feature_indices]

        # fit classifier, make predictions, record accuracies
        clf.fit(X_train, y_train)
        train_prediction = clf.predict(X_train)
        train_acc = accuracy_score(y_train, train_prediction)
        test_prediction = clf.predict(X_test)
        test_acc = accuracy_score(y_test, test_prediction)
        accuracies = [train_acc, test_acc]

        if verbose:
            # print accuracies
            print("Training Accuracy: {:.3f}".format(train_acc))
            print("Validation Accuracy: {:.3f}".format(test_acc))

        data_features = [X_train, X_test, y_train, y_test]
        return data_features, clf, accuracies



if __name__=="__main__":
    # example: use logistic regression to predict diabetes based on pregnancy and glucose
    classifier = DataClassifier("../data/diabetes.csv")
    classifier.fit(method="logreg", verbose=True)