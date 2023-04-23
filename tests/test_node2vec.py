import sys

sys.path.append("..")
from src.utils import get_dateset
from src.utils import get_labels
from src.utils import node_classification
from src.utils import node_visualization_3
from src import Node2Vec
import networkx as nx

data_info = get_dateset(data_name='wiki')

if __name__ == "__main__":
    # init and train
    G = nx.read_edgelist(data_info['edges'], create_using=nx.DiGraph(), data=[('weight', int)])
    model = Node2Vec(G, walk_num=80, walk_len=10, p=1.0, q=1.0, walkers=1)
    model.train()
    embeddings = model.get_embedding()

    # evaluation
    labels = get_labels(data_info['labels'])
    node_classification(embeddings, labels, 0.2)
    node_visualization_3(embeddings, labels)