import sys
sys.path.append("..")
from src.utils import get_dateset
from src.utils import get_labels
from src.utils import node_classification
from src.utils import node_visualization_3
from src import DeepWalk
from src.utils import get_f1_scores
import networkx as nx

data_info = get_dateset(data_name='tmall')
plot_title = 'DeepWalk'

if __name__ == "__main__":
    # DIGraph for wiki, cora; otherwise for dblp and blogCatalog
    G = nx.read_edgelist(data_info['edges'], create_using=nx.Graph())
    model = DeepWalk(G, walk_num=80, walk_len=10, walkers=1)
    model.train()
    embeddings = model.get_embedding()
    labels = get_labels(data_info['labels'])

    micro_f1, macro_f1 = get_f1_scores(embeddings, labels, 0.2)
    print("micro_f1: {:.5f}".format(micro_f1))
    print("macro_f1: {:.5f}".format(macro_f1))
    # node_visualization_3(embeddings, labels, [1, 4, 7], title='DeepWalk-dblp')
    # node_visualization_3(embeddings, labels, title='DeepWalk-wiki')