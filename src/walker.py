import random
from joblib import Parallel, delayed
import itertools

class RandomWalker:
    def __init__(self, G, p=1, q=1):
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

    def _slipt_walk_num(self, num, walkers):
        if num % walkers == 0:
            return [num // walkers] * walkers
        else:
            return [num // walkers] * walkers + [num % walkers]

