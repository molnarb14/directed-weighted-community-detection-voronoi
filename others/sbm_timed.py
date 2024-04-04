import numpy as np
import graph_tool.all as gt
import sys
import os
import time

if(len(sys.argv) != 8):
    print("> [U] : python {} [number of nodes] [average degree] [mixing parameter] [distribution type] [ins] [outs] [count]".format(sys.argv[0]))
    sys.exit(1)

start = time.time()

N = int(sys.argv[1])
k = int(sys.argv[2])
mu = sys.argv[3]
dtype = sys.argv[4]
ins = sys.argv[5]
outs = sys.argv[6]
count = int(sys.argv[7])

foldername = "N{}_k{}_mu{}_{}{}{}".format(N, k, mu, dtype, ins, outs)

if not os.path.exists("results/sbm2/{}".format(foldername)):
    os.makedirs("results/sbm2/{}".format(foldername))

total = 0
times = np.zeros(count + 1)
diffs_alone = np.zeros(count + 1)

for nid in range(1, count + 1):
    one_start = time.time()

    g = gt.Graph(directed=True)
    g.add_vertex(N)

    src, tgt, weights = np.loadtxt("../benchmarks/{}/test_{}.txt".format(foldername, nid), dtype=float, unpack=True)
    src = src.astype(int)
    src -= 1
    tgt = tgt.astype(int)
    tgt -= 1

    for i in range(0, len(weights)):
        g.add_edge(src[i], tgt[i])

    edge_weights = g.new_edge_property('double')
    g.edge_properties['weights'] = edge_weights
    for index, e in enumerate(g.edges()):
        edge_weights[e] = weights[index]

    # vertex_ids = g.add_edge_list(edgelist, hashed=True)

    state = gt.minimize_blockmodel_dl(g, state_args=dict(recs=[g.ep.weights], rec_types=["real-exponential"]))
    b = state.get_blocks()

    cluster_indexes = []
    for node in range(0, N):
        if b[node] not in cluster_indexes:
            cluster_indexes.append(b[node])

    transformed_clustering = np.zeros(N)
    for node in range(0, N):
        transformed_clustering[node] = cluster_indexes.index(b[node]) + 1
    transformed_clustering = transformed_clustering.astype(int)

    q = gt.modularity(g, b, weight=g.ep.weights)

    with open("results/sbm2/{}/info_{}_timed.txt".format(foldername, nid), "w") as f:
        f.write("{}\t{}\n".format(len(cluster_indexes), q))
    f.close()

    with open("results/sbm2/{}/test_{}_timed.txt".format(foldername, nid), "w") as f:
        for node in range(0, N):
            f.write("{}\n".format(transformed_clustering[node]))
    f.close()

    one_end = time.time()
    one_diff = (one_end - one_start)
    total += one_diff
    times[nid - 1] = one_diff
    diffs_alone[nid - 1] = one_diff

end = time.time()
diff = end - start
diff_std = np.std(times)

with open("results/sbm2/{}_times.txt".format(foldername), "w") as f:
    for i in range(1, count + 1):
        f.write("{} {:.2f}\n".format(i, diffs_alone[i]))

    f.write("overall: {:.2f}\n".format(diff))
    f.write("average per network: {:.2f}\n".format(diff/count))
    f.write("standard deviation: {:.2f}\n".format(diff_std))
f.close()
