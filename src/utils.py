import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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

def node_visualization(embeddings, labels):
    keys = sorted(embeddings.keys())
    X = np.array([embeddings[node] for node in keys])
    y = np.array([labels[node] for node in keys])

    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=0)
    X_tsne = tsne.fit_transform(X)

    labels, counts = np.unique(y, return_counts=True)
    top_labels = labels[np.argsort(counts)][::-1][:3]

    mask = np.isin(y, top_labels)
    X_top = X_tsne[mask]
    y_top = y[mask]

    colors = ['#108831', '#880c7f', '#4e8ab5']
    color_map = dict(zip(top_labels, colors))
    c = np.array([color_map[label] for label in y_top])

    plt.figure(figsize=(5, 5))
    plt.scatter(X_top[:, 0], X_top[:, 1], c=c, s=3)
    plt.axis('off')
    plt.show()