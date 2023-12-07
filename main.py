# coding=utf-8
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print("Hi, {0}".format(name))  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# Import the necessary libraries
import os
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from joblib import dump

def get_data(base_dir):
    # Get a list of file names ending with '.csv' in the specified directory
    data_file_names = [x for x in os.listdir(base_dir) if x.endswith('.csv')]
    data = {}
    for name in data_file_names:
        # Read each CSV file into a Pandas DataFrame and store it in a dictionary
        path_file = os.path.join(base_dir, name)
        data[name] = pd.read_csv(path_file)
    return data

def split_data(data, test_size=0.2, random_state=42):
    # Extract features (X) and target variable (y) from the DataFrame
    df = data['iris.csv']
    X = df.drop('Species', axis=1)
    y = df['Species']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

def train_model(X_train, y_train):
    # Initialize a Logistic Regression classifier
    clf = LogisticRegression(max_iter=1000, random_state=200)
    
    # Use GridSearchCV to perform hyperparameter tuning
    mod = GridSearchCV(clf, param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100]})
    
    # Fit the model to the training data
    mod.fit(X_train, y_train)
    
    # Get the best estimator from the grid search
    m = mod.best_estimator_
    return m

def save_model(m):
    # Save the trained model to a joblib file
    dump(m, 'model/Regress_log.joblib')
    print("Model saved")

def create_metrics(X_test, y_test, clf):
    # Generate classification report and compute AUC score
    clf_report = classification_report(y_test, clf.predict(X_test))
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    return {'auc': auc, 'clf_report': clf_report}

def save_metrics(metrics):
    # Save the computed metrics to a JSON file
    with open('../assets/model/logistic_regression_metrics.json', 'w') as out_file:
        json.dump(metrics, out_file, sort_keys=True, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # Specify the base directory containing the data files
    base_dir = r'C:\Users\Gaurav Singh\PycharmProjects\pythonProject'
    
    # Retrieve data from CSV files in the specified directory
    data = get_data(base_dir)

    # Split the data into training and testing sets
    split_data = split_data(data['iris.csv'])

    # Train a Logistic Regression model on the training set
    m = train_model(split_data['X_train'], split_data['y_train'])
    
    # Evaluate the model and save the computed metrics
    metrics = create_metrics(split_data['X_test'], split_data['y_test'], m)
    save_model(m)
    save_metrics(metrics)
