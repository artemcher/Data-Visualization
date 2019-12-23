import numpy as np
import io
import base64
from sklearn.linear_model import LogisticRegression as logreg_clf
from sklearn.tree import DecisionTreeClassifier as tree_clf
from sklearn.ensemble import RandomForestClassifier as forest_clf
from sklearn.metrics import accuracy_score


from data import *
from fitting import *


class ClassificationVisualizer:
    '''Module for visualizing diabetes data and classifier predictions.
       Supports visualization of only two data features at once.'''

    def __init__(self, data_file):
        self.classifier = DataClassifier(data_file)


    def get_classifier(self):
        '''
        Returns DataClassifier object instance.
        '''
        return self.classifier


    def _get_feature_names(self, X1_index, X2_index):
        '''
        Fetch names of selected features.
        Args:
            X1_index: index of the first selected data feature (between 0 and 7)
            X2_index: index of the second selected data featue (between 0 and 7)
        '''
        feature_names = self.get_classifier().get_data_reader().feature_names
        X1_name = feature_names[X1_index]
        X2_name = feature_names[X2_index]
        return X1_name, X2_name


    def _get_feature_units(self, X1_index, X2_index):
        '''
        Fetch descriptions/units of selected features.
        Args:
            X1_index: index of the first selected data feature (between 0 and 7)
            X2_index: index of the second selected data featue (between 0 and 7)
        '''
        feature_units = self.get_classifier().get_data_reader().feature_units
        X1_units = feature_units[X1_index]
        X2_units = feature_units[X2_index]
        return X1_units, X2_units


    def _get_method_name(self, method):
        '''
        Get full name of the selected learning method.
        Args:
            method: abbreviated name of selected learning method
                    (allowed values are "logreg", "tree", "forest")
        '''
        if method == "logreg":
            return "Logistic Regression"
        elif method == "tree":
            return "Decision Tree"
        elif method == "forest":
            return "Random Forest"
        else:
            raiseValueError("Undefined classification method.")


    def visualize(self, X1_index=0, X2_index=1, method="logreg", make_url=False):
        '''
        Plot selected VALIDATION data subsets along with decision boundaries
        of selected supervised learning method.

        Args:
            X1_index: index of the first feature to plot (between 0 and 7)
            X2_index: index of the second feature to plot (between 0 and 7)
              method: supervised learning method to use
                      (allowed values are "logreg", "tree", "forest")
            make_url: if toggled, encode resulting plot as a base 64 string
        
        Returns accuracies and plot (as base 64 string), if "make_url" is toggled.
        '''

        # get names of learning method and data features
        X1_name, X2_name = self._get_feature_names(X1_index, X2_index)
        X1_units, X2_units = self._get_feature_units(X1_index, X2_index)
        method_name = self._get_method_name(method)

        # fit classifier, extract validation data
        feature_indices = np.array([X1_index, X2_index])
        data_features, clf, accuracies = self.classifier.fit(feature_indices,
                                        method=method, verbose=False)
        X_test, y_test = data_features[1], data_features[3]

        # set minimum and maximum axis values for plot
        X1_min, X1_max = X_test[:,0].min()-1, X_test[:,0].max()+1
        X2_min, X2_max = X_test[:,1].min()-1, X_test[:,1].max()+1

        X1_min, X1_max = X1_min - 0.1*X1_min, X1_max + 0.1*X1_max
        X2_min, X2_max = X2_min - 0.1*X2_min, X2_max + 0.1*X2_max

        # turn data subset into a mesh for plotting
        X1_mesh, X2_mesh = np.meshgrid(np.arange(X1_min, X1_max, np.abs(0.005*X1_max)),
                                       np.arange(X2_min, X2_max, np.abs(0.005*X2_max)))

        # get classifier prediction for every value in mesh
        Z = clf.predict(np.c_[X1_mesh.ravel(), X2_mesh.ravel()])
        Z = Z.reshape(X1_mesh.shape)

        # make plot with decision boundaries
        plot_title = "Values of \"{}\" and \"{}\" features:\n".format(X1_name, X2_name) +\
                     "diabetes occurence ({} decision boundary)".format(method_name)
        plt.figure(figsize=[6.4,4.8])
        plt.contourf(X1_mesh, X2_mesh, Z, cmap="bwr", alpha=0.25)
        plt.contour(X1_mesh, X2_mesh, Z, colors='k', linewidths=0.1)
        plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap="bwr", edgecolors='k')
        plt.title(plot_title)
        plt.xlabel("{} {}".format(X1_name, X1_units))
        plt.ylabel("{} {}".format(X2_name, X2_units))

        # set up colorbar
        cbar = plt.colorbar()
        cbar.ax.set_yticklabels(['No','', '', '', '', 'Yes'])
        cbar.set_label("Diabetes", rotation=270)

        if not make_url:
            # print accuracies and display the plot
            print("Training Accuracy: {:.3f}".format(accuracies[0]))
            print("Validation Accuracy: {:.3f}".format(accuracies[1]))
            plt.show()
        else:
            # save plot as bytes and encode result as a base 64 string
            bytes_plot = io.BytesIO()
            plt.savefig(bytes_plot, format="png")
            bytes_plot.seek(0)
            plot_url = base64.b64encode(bytes_plot.getvalue()).decode()
            return plot_url, accuracies


if __name__=="__main__":
    # example: plot logistic regression predictions based on pregnancies and glucose
    visualizer = ClassificationVisualizer("../data/diabetes.csv")
    visualizer.visualize()