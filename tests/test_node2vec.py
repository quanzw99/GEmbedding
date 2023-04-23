import sys

sys.path.append("..")
from src.utils import get_dateset
from src.utils import get_labels
from src.utils import node_classification
from src.utils import node_visualization_3
from src.utils import k_fold_cross_validation
from src import Node2Vec
import networkx as nx

data_info = get_dateset(data_name='wiki')
plot_title = 'Node2Vec'
# optional_para = [0.25, 0.5]
optional_para = [0.25, 0.5, 1, 2, 4]

def cross_validation(test_size=0.2, k=10):
    with open("./node2vec_result.txt", "w") as f:
        f.truncate(0)
    G = nx.read_edgelist(data_info['edges'], create_using=nx.DiGraph(), data=[('weight', int)])
    labels = get_labels(data_info['labels'])
    for p in optional_para:
        for q in optional_para:
            model = Node2Vec(G, walk_num=80, walk_len=10, p=p, q=q, walkers=1)
            model.train()
            embeddings = model.get_embedding()
            info = f"{plot_title} - (p = {p}, q = {q}): "
            k_fold_cross_validation(embeddings, labels, test_size, k, info)

if __name__ == "__main__":
    cross_validation()
    '''
    # init and train
    G = nx.read_edgelist(data_info['edges'], create_using=nx.Graph(), data=[('weight', int)])
    model = Node2Vec(G, walk_num=80, walk_len=10, p=0.25, q=4, walkers=1)
    model.train()
    embeddings = model.get_embedding()

    # evaluation
    labels = get_labels(data_info['labels'])
    node_classification(embeddings, labels, 0.2)
    node_visualization_3(embeddings, labels, [1, 4, 7], plot_title)
    '''
