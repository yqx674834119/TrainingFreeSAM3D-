"""Turn edge merge probabilities into a mutually exclusive grouping."""
import numpy as np


def average_linkage(num_nodes, edges, prob, threshold):
    """Greedy agglomeration: repeatedly merge the cluster pair with the highest
    mean edge probability while it exceeds ``threshold``.

    Averaging over all edges between two clusters prevents a single confident
    edge from chaining unrelated groups together (the connected-components
    failure mode). Dense (G,G) formulation — G ≈ 70, so the matrices are tiny
    and each merge is two row/column adds (2.7 ms/frame; the dict-of-links
    version was 8.6 ms and a torch version 13 ms from launch overhead).
    """
    edges, prob = np.asarray(edges), np.asarray(prob, float)
    S = np.zeros((num_nodes, num_nodes))          # summed probability between clusters
    C = np.zeros((num_nodes, num_nodes))          # edge count between clusters
    if len(edges):
        a, b = edges[:, 0], edges[:, 1]
        np.add.at(S, (a, b), prob)
        np.add.at(S, (b, a), prob)
        np.add.at(C, (a, b), 1)
        np.add.at(C, (b, a), 1)
    alive = np.ones(num_nodes, bool)
    owner = np.arange(num_nodes)
    while True:
        with np.errstate(invalid='ignore', divide='ignore'):
            A = np.where(C > 0, S / C, -np.inf)
        A[~alive] = -np.inf
        A[:, ~alive] = -np.inf
        i, j = divmod(int(A.argmax()), num_nodes)
        if A[i, j] < threshold:
            break
        i, j = min(i, j), max(i, j)
        S[i] += S[j]
        S[:, i] += S[:, j]
        C[i] += C[j]
        C[:, i] += C[:, j]
        S[i, i] = C[i, i] = 0
        alive[j] = False
        owner[owner == j] = i
    _, out = np.unique(owner, return_inverse=True)
    return out


def connected_components(num_nodes, edges, prob, threshold):
    parent = list(range(num_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for (a, b), p in zip(edges, prob):
        if p >= threshold:
            parent[find(a)] = find(b)
    roots = np.array([find(i) for i in range(num_nodes)])
    _, out = np.unique(roots, return_inverse=True)
    return out


LINKAGE = {'average': average_linkage, 'single': connected_components}


def merge_groups(groups, edges, prob, threshold, linkage='average', return_mapping=False):
    """Point-level group ids -> merged point-level ids (and optionally the (G,) node -> cluster map)."""
    num = int(groups.max()) + 1
    mapping = LINKAGE[linkage](num, np.asarray(edges), np.asarray(prob), threshold)
    return (mapping[groups], mapping) if return_mapping else mapping[groups]
