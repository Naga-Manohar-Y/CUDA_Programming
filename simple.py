import networkx as nx
import numpy as np
import time
import sys
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from sklearn.metrics import adjusted_rand_score
from community import community_louvain  # Louvain algorithm implementation
from networkx.algorithms.community import girvan_newman


graph_file=sys.argv[1]
label_file=sys.argv[2]
alg=sys.argv[3]
st=float(sys.argv[4])

# Step 1: Load the graph from the edges file (space-separated values)
G = nx.Graph()  # Use undirected graph from the start

with open(graph_file, "r") as f:
    for line in f:
        source, target = line.strip().split()  # Use space as separator
        G.add_edge(source, target)

# Step 2: Load the labels and add them as node attributes (space-separated values)
with open(label_file, "r") as f:
    for line in f:
        node, label = line.strip().split(",")  # Use space as separator
        if node in G:  # Only add label if the node exists in the graph
            G.nodes[node]["label"] = int(label)

# Display graph information to verify
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# Display a sample of nodes with their attributes to confirm labels
print("Sample node attributes:")
for node, attrs in list(G.nodes(data=True))[:5]:  # Adjust sample size as needed
    print(f"Node {node}: {attrs}")
    
    
# graph="dataset/"+graph
# edges = np.loadtxt(graph+".edges", dtype=int)
# n=edges.max()
# G = nx.Graph()
# G.add_edges_from(edges)

print("==>", G.number_of_nodes(), G.number_of_edges(), time.time())
orc_ATD = OllivierRicci(G, alpha=0.5, method=alg, verbose="INFO")
t0 = time.time()
orc_ATD.compute_ricci_flow(iterations=5, step=st)
cc=orc_ATD.ricci_community()
# l_true, l_pred = gen_labels(graph+".node_labels", cc[1])
# ari_score = adjusted_rand_score(l_true, l_pred)
# louv_pred = louv_labels(G)
# louv_score = adjusted_rand_score(l_true, louv_pred)
# gn_pred = gn_labels(G)
# gn_score = adjusted_rand_score(l_true, gn_pred)


# l_pred=gen_labels1(cc[1])
# louv_pred = louv_labels(G)
# louv_ricci_score = adjusted_rand_score(louv_pred,l_pred)
# gn_pred = gn_labels(G)
# gn_ricci_score = adjusted_rand_score(l_pred, gn_pred)
# print("<==", graph, gn_ricci_score, louv_ricci_score, time.time()-t0)
# print("<==", graph, louv_ricci_score, time.time()-t0)