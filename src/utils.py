import json
import sys

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
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

def k_fold_cross_validation(embeddings, labels, test_size, k, info, file_name):
    result_file = f'../tests/{file_name}.txt'
    keys = sorted(labels.keys())
    X = np.array([embeddings[node] for node in keys])
    Y = np.array([labels[node] for node in keys])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    clf = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=1000)
    f1_micro = cross_val_score(clf, X_train, Y_train, cv=k, scoring='f1_micro', n_jobs=8)
    f1_micro = f1_micro.sum() / k
    f1_macro = cross_val_score(clf, X_train, Y_train, cv=k, scoring='f1_macro', n_jobs=8)
    f1_macro = f1_macro.sum() / k
    result = f"f1_micro = {f1_micro}, f1_macro = {f1_macro}\n"
    with open(result_file, 'a') as f:
        f.write(info + result)
    return

def node_classification(embeddings, labels, test_size):
    keys = sorted(labels.keys())
    X = np.array([embeddings[node] for node in keys])
    Y = np.array([labels[node] for node in keys])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    clf = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=1000)
    clf.fit(X_train, Y_train)

    # Get report
    Y_pred = clf.predict(X_test)
    # print(classification_report(Y_test, Y_pred, zero_division=1))
    micro_f1 = f1_score(Y_test, Y_pred, average='micro')
    macro_f1 = f1_score(Y_test, Y_pred, average='macro')
    return micro_f1, macro_f1

def get_f1_scores(embeddings, labels, test_size):
    micro_f1_list = []
    macro_f1_list = []

    for _ in range(7):
        micro_f1, macro_f1 = node_classification(embeddings, labels, test_size)
        micro_f1_list.append(micro_f1)
        macro_f1_list.append(macro_f1)

    # remove the max and the min
    micro_f1_list.remove(max(micro_f1_list))
    micro_f1_list.remove(min(micro_f1_list))
    macro_f1_list.remove(max(macro_f1_list))
    macro_f1_list.remove(min(macro_f1_list))

    micro_mean = sum(micro_f1_list) / len(micro_f1_list)
    macro_mean = sum(macro_f1_list) / len(macro_f1_list)
    return micro_mean, macro_mean

def node_visualization_3(embeddings, labels, top_labels=[], title=''):
    keys = sorted(labels.keys())
    X = np.array([embeddings[node] for node in keys])
    y = np.array([labels[node] for node in keys])

    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=0)
    X_tsne = tsne.fit_transform(X)

    # choose top labels
    min_top_counts = sys.maxsize
    if len(top_labels) != 3:
        labels, counts = np.unique(y, return_counts=True)
        top_labels = labels[np.argsort(counts)][::-1][:3]
        min_top_counts = counts[np.argsort(counts)][::-1][0]
        print(top_labels)
    else:
        for label in top_labels:
            label_indices = np.where(y == label)[0]
            min_top_counts = min(len(label_indices), min_top_counts)

    mask = []
    if min_top_counts > 500:
        mask = np.zeros_like(y, dtype=bool)
        for label in top_labels:
            label_indices = np.where(y == label)[0]
            np.random.shuffle(label_indices)
            mask[label_indices[:500]] = True
    else:
        mask = np.isin(y, top_labels)

    X_top = X_tsne[mask]
    y_top = y[mask]

    colors = ['#108831', '#880c7f', '#4e8ab5']
    color_map = dict(zip(top_labels, colors))
    c = np.array([color_map[label] for label in y_top])

    plt.figure(figsize=(5, 5))
    plt.title(title)
    plt.scatter(X_top[:, 0], X_top[:, 1], c=c, s=5)
    plt.axis('off')
    plt.show()
    return