import numpy as np
import io
from visualize import *

from flask import Flask, send_file, render_template, request
app = Flask(__name__)


'''Web app for displaying diabetes data and its analysis.'''

# instantiate relevant objects
visualizer = ClassificationVisualizer("../data/diabetes.csv")
classifier = visualizer.get_classifier()
data = classifier.get_data_reader()


@app.route("/")
def index():
    '''
    Render starting page.
    '''
    return render_template("index.html", data_plot=None, class_plot=None,
        train_acc=None, val_acc=None)


@app.route("/show_data", methods=['GET', 'POST'])
def show_data():
    '''
    Show plot of selected data features (from user input) against diabetes occurence.
    '''
    X1_index = int(request.args.get("df1"))
    X2_index = int(request.args.get("df2"))
    data_plot_url = data.plot_features(X1_index=X1_index, X2_index=X2_index, make_url=True)
    return render_template('index.html', data_plot=data_plot_url, class_plot=None)


@app.route("/show_analysis", methods=['GET', 'POST'])
def show_analysis():
    '''
    Predict diabetes from data features (from user input) and display results.
    '''
    X1_index = int(request.args.get("af1"))
    X2_index = int(request.args.get("af2"))
    method = request.args.get("amethod")
    feature_indices = np.array([X1_index, X2_index]).astype("int")
    classifier_plot_url, accuracies = visualizer.visualize(X1_index=X1_index, X2_index=X2_index,
                                                            method=method, make_url=True)
    train_acc = "Training accuracy: {:.1f}%".format(accuracies[0]*100)
    val_acc = "Validation accuracy: {:.1f}%".format(accuracies[1]*100)
    return render_template("index.html", data_plot=None, class_plot=classifier_plot_url,
        train_acc=train_acc, val_acc=val_acc)


@app.route("/help")
def show_help():
    '''
    Render help page.
    '''
    return render_template("help.html", module_doc=None)


@app.route("/show_documentation")
def show_documentation():
    '''
    Show documentation of selected Python module (from user input).
    '''
    selected_module = request.args.get("module")
    f = open("docs/{}.html".format(selected_module))
    module_doc = f.read()
    f.close()
    return render_template("help.html", module_doc=module_doc)




if __name__=="__main__":
    # run the app
    app.run(debug=False)
    # OPEN "localhost:5000" IN BROWSER TO VIEW THE APP