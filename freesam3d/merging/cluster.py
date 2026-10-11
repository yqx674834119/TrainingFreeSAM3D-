"""Turn edge merge probabilities into a mutually exclusive grouping."""
import numpy as np


def average_linkage(num_nodes, edges, prob, threshold):
    """Greedy agglomeration: repeatedly merge the cluster pair with the highest
    mean edge probability while it exceeds ``threshold``.

    Averaging over all edges between two clusters prevents a single confident
    edge from chaining unrelated groups together (the connected-components
    failure mode).
    """
    parent = list(range(num_nodes))
    members = {i: {i} for i in range(num_nodes)}
    # cluster-pair -> (sum, count)
    links = {}
    for (a, b), p in zip(edges, prob):
        key = (min(a, b), max(a, b))
        s, c = links.get(key, (0.0, 0))
        links[key] = (s + float(p), c + 1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    while links:
        key, (s, c) = max(links.items(), key=lambda kv: kv[1][0] / kv[1][1])
        if s / c < threshold:
            break
        i, j = key
        parent[j] = i
        members[i] |= members.pop(j)
        new = {}
        for (a, b), v in links.items():
            if (a, b) == key:
                continue
            a, b = (i if a == j else a), (i if b == j else b)
            if a == b:
                continue
            k = (min(a, b), max(a, b))
            s0, c0 = new.get(k, (0.0, 0))
            new[k] = (s0 + v[0], c0 + v[1])
        links = new
    out = np.empty(num_nodes, np.int64)
    for k, (root, mem) in enumerate(members.items()):
        out[list(mem)] = k
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


def merge_groups(groups, edges, prob, threshold, linkage='average'):
    """Point-level group ids -> merged point-level ids."""
    num = int(groups.max()) + 1
    mapping = LINKAGE[linkage](num, np.asarray(edges), np.asarray(prob), threshold)
    return mapping[groups]
