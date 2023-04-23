import numpy as np
import matplotlib.pyplot as plt

def gen_prob_dist(N):
    p = np.random.randint(0, 100, N)
    return p / np.sum(p)

def create_alias_table(area_ratio):
    N = len(area_ratio)
    probs, alias = [0] * N, [0] * N
    small, large = [], []
    area_ratio = np.array(area_ratio) * N

    for i, prob in enumerate(area_ratio):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        probs[small_idx] = area_ratio[small_idx]
        alias[small_idx] = large_idx
        area_ratio[large_idx] = area_ratio[large_idx] - (1 - area_ratio[small_idx])
        if area_ratio[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        probs[large_idx] = 1
    while small:
        small_idx = small.pop()
        probs[small_idx] = 1

    return probs, alias

def alias_sample(probs, alias):
    N = len(probs)
    random1 = np.random.randint(0, N)
    random2 = np.random.random()
    if random2 < probs[random1]:
        return random1
    else:
        return alias[random1]

def test_sample(N, k):
    truth_dis = gen_prob_dist(N)
    probs, alias = create_alias_table(truth_dis)
    ans = np.zeros(N)
    for _ in range(k):
        tmp = alias_sample(probs, alias)
        ans[tmp] += 1
    return ans/np.sum(ans), truth_dis

def plot_hist(alias_sample, truth_dis):
    x = np.arange(len(alias_sample))
    width = 0.35
    plt.bar(x - width/2, alias_sample, width, label='alias')
    plt.bar(x + width/2, truth_dis, width, label='truth')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    alias_sample, truth_dis = test_sample(N=30, k=10000)
    plot_hist(alias_sample, truth_dis)