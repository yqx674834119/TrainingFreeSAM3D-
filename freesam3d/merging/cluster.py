"""Turn edge merge probabilities into a mutually exclusive grouping."""
import heapq

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


def average_linkage_sparse(num_nodes, edges, prob, threshold):
    """``average_linkage`` with identical semantics (same merge order and tie-breaking:
    highest mean first, then smallest node ids; merges while mean >= threshold), but
    only cluster pairs that actually have edges are tracked (dict + heap + adjacency
    sets), so the cost is O(E log E) instead of O(G^2) per merge. Used by the window
    memory, where the node count grows past a few hundred.
    """
    edges = np.asarray(edges, np.int64)
    prob = np.asarray(prob, float)
    num_nodes = int(num_nodes)
    S, C = {}, {}                                  # (a, b), a < b -> summed probability / edge count
    adj = [set() for _ in range(num_nodes)]
    for (a, b), p in zip(edges, prob):
        a, b = int(a), int(b)
        if a == b:
            continue
        if a > b:
            a, b = b, a
        S[(a, b)] = S.get((a, b), 0.0) + float(p)
        C[(a, b)] = C.get((a, b), 0) + 1
        adj[a].add(b)
        adj[b].add(a)
    heap = [(-S[k] / C[k], k[0], k[1]) for k in S]
    heapq.heapify(heap)
    alive = np.ones(num_nodes, bool)
    owner = np.arange(num_nodes)
    while heap:
        negmean, a, b = heapq.heappop(heap)
        if not (alive[a] and alive[b]):
            continue
        key = (a, b)
        if key not in S or -S[key] / C[key] != negmean:
            continue                               # stale entry (pair changed after a merge)
        if -negmean < threshold:
            break
        for k in list(adj[b]):                     # merge b into a
            if k == a or not alive[k]:
                continue
            kb, ka = (min(k, b), max(k, b)), (min(k, a), max(k, a))
            S[ka] = S.get(ka, 0.0) + S.pop(kb)
            C[ka] = C.get(ka, 0) + C.pop(kb)
            adj[a].add(k)
            adj[k].discard(b)
            adj[k].add(a)
            heapq.heappush(heap, (-S[ka] / C[ka], ka[0], ka[1]))
        adj[b].clear()
        alive[b] = False
        owner[owner == b] = a
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


LINKAGE = {'average': average_linkage, 'single': connected_components, 'average_sparse': average_linkage_sparse}


def merge_groups(groups, edges, prob, threshold, linkage='average', return_mapping=False):
    """Point-level group ids -> merged point-level ids (and optionally the (G,) node -> cluster map)."""
    num = int(groups.max()) + 1
    mapping = LINKAGE[linkage](num, np.asarray(edges), np.asarray(prob), threshold)
    return (mapping[groups], mapping) if return_mapping else mapping[groups]
