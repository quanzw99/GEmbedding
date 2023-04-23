import random
from joblib import Parallel, delayed
import itertools
from .alias import create_alias_table
from .alias import alias_sample

class RandomWalker:
    def __init__(self, G, p=1.0, q=1.0):
        self.G = G
        self.p = p
        self.q = q

    def simulate_walk(self, walk_num, walk_len, walkers=1, verbose=1):
        nodes = list(self.G.nodes())
        nums = self._slipt_walk_num(walk_num, walkers)
        tmp_path = Parallel(n_jobs=walkers, verbose=verbose)(
            delayed(self._simulate_walk)(num, walk_len, nodes) for num in nums
        )
        paths = list(itertools.chain(*tmp_path))
        return paths

    def _simulate_walk(self, walk_num, walk_len, nodes):
        paths = []
        for _ in range(walk_num):
            random.shuffle(nodes)
            for v in nodes:
                if self.p == 1 and self.q == 1:
                    paths.append(self._deepwalk_walk(walk_len, v))
                else:
                    paths.append(self._node2vec_walk(walk_len, v))
        return paths

    def _deepwalk_walk(self, walk_len, cur_node):
        path = [cur_node]
        while len(path) < walk_len:
            cur_node = path[-1]
            cur_nb = list(self.G.neighbors(cur_node))
            if len(cur_nb) > 0:
                path.append(random.choice(cur_nb))
            else:
                break
        return path

    def _node2vec_walk(self, walk_len, cur_node):
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        path = [cur_node]
        while len(path) < walk_len:
            cur_node = path[-1]
            cur_nb = list(self.G.neighbors(cur_node))
            if len(cur_nb) > 0:
                if len(path) == 1:
                    nxt_node_idx = alias_sample(alias_nodes[cur_node][0], alias_nodes[cur_node][1])
                    path.append(cur_nb[nxt_node_idx])
                else:
                    edge = (path[-2], cur_node)
                    nxt_node_idx = alias_sample(alias_edges[edge][0], alias_edges[edge][1])
                    path.append(cur_nb[nxt_node_idx])
            else:
                break
        return path

    def _slipt_walk_num(self, num, walkers):
        if num % walkers == 0:
            return [num // walkers] * walkers
        else:
            return [num // walkers] * walkers + [num % walkers]

    # t is the previous node
    # v is the current node
    # x is the next node
    def get_edge_alias(self, t, v):
        G = self.G
        p = self.p
        q = self.q
        unnormalized_probs = []
        for x in G.neighbors(v):
            weight = G[v][x].get('weight', 1.0)
            if x == t: # d_tx = 0
                unnormalized_probs.append(weight / p)
            elif G.has_edge(x, t): # d_tx = 1
                unnormalized_probs.append(weight)
            else: # d_tx > 1
                unnormalized_probs.append(weight / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(prob) / norm_const for prob in unnormalized_probs]
        return create_alias_table(normalized_probs)

    def cal_transition_probs(self):
        G = self.G

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr].get('weight', 1.0) for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(prob) / norm_const for prob in unnormalized_probs]
            alias_nodes[node] = create_alias_table(normalized_probs)
        self.alias_nodes = alias_nodes

        alias_edges = {}
        for edge in G.edges():
            alias_edges[(edge[0], edge[1])] = self.get_edge_alias(edge[0], edge[1])
            if not G.is_directed():
                alias_edges[(edge[1], edge[0])] = self.get_edge_alias(edge[1], edge[0])
        self.alias_edges = alias_edges

        return