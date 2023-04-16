import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def get_dateset(data_name='wiki'):
    with open('../tests/dataset.json', 'r') as dataset_file:
        dataset_data = json.load(dataset_file)
        return dataset_data[data_name]

def get_labels(filename, skip_head = False):
    labels = {}
    with open(filename, 'r') as reader:
        for line in reader:
            parts = line.strip().split()
            labels[parts[0]] = int(parts[1])
    return labels

def node_classification(embeddings, labels, test_size):
    keys = sorted(embeddings.keys())
    X = np.array([embeddings[node] for node in keys])
    Y = np.array([labels[node] for node in keys])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    clf.fit(X_train, Y_train)

    # Test Accuracy
    print("Test score: {:.5f}".format(clf.score(X_test, Y_test)))

    # Get report
    Y_pred = clf.predict(X_test)
    print(classification_report(Y_test, Y_pred, zero_division=1))

