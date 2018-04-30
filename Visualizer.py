import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)

def plot_network(adj_mat, labels):
    G = nx.Graph()

    edges = []
    pos_edge = []
    neg_edge = []
    for (i, j) in zip(*np.where(abs(adj_mat) > 0)):
        if i > j:
            e = (labels[i], labels[j])
            if adj_mat[i][j] > 0:
                pos_edge.append(e)
            else:
                neg_edge.append(e)
            edges.append(e)

    G.add_edges_from(edges)

    pos = nx.spring_layout(G)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=pos_edge, width=1)
    nx.draw_networkx_edges(G, pos, edgelist=neg_edge, width=1, alpha=0.5, edge_color='b', style='dashed')

    # labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

    plt.axis('off')
    plt.show()