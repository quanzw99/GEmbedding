from .walker import RandomWalker
from gensim.models import Word2Vec

class Node2Vec:
    def __init__(self, graph, walk_num, walk_len, p=1.0, q=1.0, walkers=1):
        self.graph = graph
        self.model = None
        self._embed = {}
        self.walker = RandomWalker(graph, p=p, q=q)
        print('cal_transition_probs start')
        self.walker.cal_transition_probs()
        print('cal_transition_probs finished')
        self.sentences = self.walker.simulate_walk(walk_num, walk_len, walkers, verbose=1)

    def train(self, embed_size=128, window_size=5, workers=3, epochs=5, **kwargs):
        kwargs["vector_size"] = embed_size
        kwargs["window"] = window_size
        kwargs["workers"] = workers
        kwargs["epochs"] = epochs
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["sg"] = 1

        # according  to the paper, node2vec uses negative sampling
        # instead of hierarchical softmax
        kwargs["hs"] = 0

        # print(f"len of the sentence = {len(self.sentences)}")
        print("Stating w2v...")
        model = Word2Vec(**kwargs)
        print("Finishing w2v...")
        self.model = model
        return model

    def get_embedding(self):
        if self.model is None:
            print("Please run train function...")
            return {}

        for node in self.graph.nodes():
            self._embed[node] = self.model.wv[node]
        return self._embed