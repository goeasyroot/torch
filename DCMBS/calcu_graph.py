
import numpy as np
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize

top_k = 10


def construct_graph(features, label, method='heat'):
    file_name = 'graph/cite2k_graph.txt'
    num = len(label)
    dist = None

    if method == 'heat':
        dist = -0.5 * pair(features) ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos':
        features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)

    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(top_k + 1))[-(top_k + 1):]
        inds.append(ind)

    f = open(file_name, 'w')
    A = np.zeros_like(dist)
    for i, v in enumerate(inds):
        f.write('{} {}\n'.format(i, v[0]))
        mutual_knn = False
        for vv in v:
            if vv == i:
                pass
            else:
                if vv != v[0] and dist[i][vv] < 0.006:
                    f.write('{} {}\n'.format(i, vv))
    f.close()


data = np.loadtxt('data/cite.txt', dtype=float)
label = np.loadtxt('data/cite_label.txt', dtype=int)
if __name__ == '__main__':
    construct_graph(data, label, 'ncos')
