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

import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from joblib import dump
import sys
import pandas as pd


def get_data(base_dir):
    data_file_names = [x for x in os.listdir(base_dir) if x.endswith('.csv')]
    data = {}
    for name in data_file_names:
        path_file = os.path.join(base_dir, name)
        data[name] = pd.read_csv("model/iris.csv")
    return data


def split_data(data, test_size=0.2, random_state=42):
    df = data['iris.csv']
    X = df.drop('Species', axis=1)
    y = df['Species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}


def train_model(X_train, y_train):
    clf = LogisticRegression(max_iter=1000, random_state=200)
    mod = GridSearchCV(clf, param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100]})
    mod.fit(X_train, y_train)
    m = mod.best_estimator_
    return m


def save_model(m):
    dump(m, 'model/Regress_log.joblib')
    print("Model saved")


def create_metrics(X_test, y_test, clf):
    clf_report = classification_report(y_test, clf.predict(X_test))
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    return {'auc': auc, 'clf_report': clf_report}


def save_metrics(metrics):
    with open('../assets/model/logistic_regression_metrics.json', 'w') as out_file:
        json.dump(metrics, out_file, sort_keys=True, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    base_dir = r'C:\Users\Gaurav Singh\PycharmProjects\pythonProject'
    data = get_data(base_dir)

    split_data = split_data(data['iris.csv'])

    m = train_model(split_data['X_train'], split_data['y_train'])
    metrics = create_metrics(split_data['X_test'], split_data['y_test'], m)
    save_model(m)
    save_metrics(metrics)
