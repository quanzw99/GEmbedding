import sys
sys.path.append("..")
from src.utils import get_dateset
from src import DeepWalk
import networkx as nx
data_info = get_dateset(data_name='wiki')

if __name__ == "__main__":
    G = nx.read_edgelist(data_info['edges'], create_using=nx.DiGraph())
    model = DeepWalk(G, walk_num=10, walk_len=10, walkers=1)
    model.train()
    embeddings = model.get_embedding()