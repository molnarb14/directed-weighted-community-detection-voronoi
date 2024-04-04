import numpy as np
import igraph as ig
import sys
import leidenalg as la
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

if not os.path.exists("results/leiden/{}".format(foldername)):
    os.makedirs("results/leiden/{}".format(foldername))

total = 0
times = np.zeros(count + 1)
diffs_alone = np.zeros(count + 1)

for nid in range(1, count + 1):
    one_start = time.time()

    g = ig.Graph.Read_Ncol("../benchmarks/{}/test_{}.txt".format(foldername, nid))

    clustering = la.find_partition(g, la.ModularityVertexPartition, weights=g.es['weight'])

    transformed_clustering = np.zeros(N)
    for cidx, c in enumerate(clustering):
        for vertex in g.vs[clustering[cidx]]['name']:
            transformed_clustering[int(vertex) - 1] = cidx
    transformed_clustering = transformed_clustering.astype(int)
    # transformed_clustering += 1 # we start indexing the clusters with 1, not 0

    q = g.modularity(clustering, directed=True, weights=g.es['weight'])

    with open("results/leiden/{}/info_{}_timed.txt".format(foldername, nid), "w") as f:
        f.write("{}\t{}\n".format(len(clustering), q))
    f.close()

    with open("results/leiden/{}/test_{}_timed.txt".format(foldername, nid), "w") as f:
        for c in transformed_clustering:
            f.write("{}\n".format(c))
    f.close()

    with open("results/leiden/{}/test_{}_clusters_timed.txt".format(foldername, nid), "w") as f:
        for c in clustering:
            f.write("{}\n".format(c))
    f.close()

    one_end = time.time()
    one_diff = (one_end - one_start)
    total += one_diff
    times[nid - 1] = one_diff
    diffs_alone[nid - 1] = one_diff

end = time.time()
diff = end - start
diff_std = np.std(times)

with open("results/leiden/{}_times.txt".format(foldername), "w") as f:
    for i in range(1, count + 1):
        f.write("{} {:.2f}\n".format(i, diffs_alone[i]))

    f.write("overall: {:.2f}\n".format(diff))
    f.write("average per network: {:.2f}\n".format(diff/count))
    f.write("standard deviation: {:.2f}\n".format(diff_std))
f.close()