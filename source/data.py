import numpy as np
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class DataReader:
    """Module for reading and accessing data from diabetes dataset."""

    def __init__(self, data_file):
        self.df = pd.read_csv(data_file).dropna()
        self.feature_names = self.df.columns.values[1:]
        self.feature_units = ["(number of times)", "(in plasma)", "(mm Hg)",\
                             "(skin thickness, mm)", "(μU/ml)", "(BMI, kg/m^2)", " ", "(years)"]
        self._get_data()
        self._split_data()


    def _get_data(self, include_id=False, binary_labels=True):
        """
        Extract data as numpy array from diabetes dataframe.
        Args:
               include_id: retain data sample IDs on "True"
            binary_labels: transform labels into binary format (zeros and ones) on "True"
        """
        self.dataset = self.df.values
        if not include_id:
            self.dataset = self.dataset[:, 1:]
        if binary_labels:
            self.dataset[:, -1] = (self.dataset[:,-1] == "pos").astype("int")


    def _split_data(self):
        """
        Split data into training and validation sets with equal class representation.
        """
        X = self.dataset[:, :-1]
        y = self.dataset[:, -1].astype("int")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,
                                                            stratify=y, test_size=0.2)


    def get_train_data(self):
        """
        Returns training data and labels.
        """
        return self.X_train, self.y_train


    def get_test_data(self):
        """
        Returns validation data and labels.
        """
        return self.X_test, self.y_test


    def get_all_data(self):
        """
        Returns all data and labels.
        """
        return self.dataset[:, :-1], self.dataset[:, -1].astype("int")


    def plot_features(self, X1_index=0, X2_index=1, make_url=False):
        """
        Plot two selected data features against labels (diabetes occurence).
        Args:
            X1_index: index of the first feature to plot (between 0 and 7)
            X2_index: index of the second feature to plot (between 0 and 7)
            make_url: on "True", encode resulting plot as URL (for web visualization)
        Returns feature plot as URL if "make_url" is toggled.
        """
        assert (X1_index >= 0 and X1_index <= 7 and X2_index >= 0 and X2_index <= 7),\
         "Failed to plot data: invalid features selected."
        
        # get relevant subset of data
        X1 = self.dataset[:, int(X1_index)]
        X2 = self.dataset[:, int(X2_index)]
        y  = self.dataset[:, -1]

        # get names and descriptions of selected data features
        X1_name = self.feature_names[int(X1_index)]
        X2_name = self.feature_names[int(X2_index)]

        # make plot
        plot_title = "Graph of \"{}\" and \"{}\" feature values\n".format(X1_name, X2_name) +\
                     "against diabetes occurence"
        plt.figure(figsize=[6.4,4.8])
        plt.title(plot_title)
        plt.scatter(X1, X2, c=y, vmin=0, vmax=1, cmap="bwr")
        plt.xlabel("{} {}".format(X1_name, self.feature_units[X1_index]))
        plt.ylabel("{} {}".format(X2_name, self.feature_units[X2_index]))

        # set up color bar
        cbar = plt.colorbar()
        cbar.ax.set_yticklabels(['No','', '', '', '', 'Yes'])
        cbar.set_label("Diabetes", rotation=270)

        if not make_url:
            plt.show()
        else:
            # save plot as bytes, encode result in base 64
            bytes_plot = io.BytesIO()
            plt.savefig(bytes_plot, format="png")
            bytes_plot.seek(0)
            plot_url = base64.b64encode(bytes_plot.getvalue()).decode()
            return plot_url



if __name__=="__main__":
    # example: plots pregnancy and glucose against diabetes
    reader = DataReader("../data/diabetes.csv")
    reader.plot_features(X1_index=0, X2_index=1)