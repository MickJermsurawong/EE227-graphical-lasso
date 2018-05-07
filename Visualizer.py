import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import Counter

plt.rcParams["figure.figsize"] = (10,10)

from nltk import tokenize


def plot_network(adj_mat, labels, save_file_name=None):
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


    if save_file_name:
        plt.savefig(save_file_name)

    plt.show()

def sentiments_by_sources(total_introductions, key='sentiment'):
    before_size = plt.rcParams["figure.figsize"]

    source_ent_sent = {}

    for intro in total_introductions:
        p = intro['person']
        s = intro['source']
        source_ent_sent.setdefault(s, {})
        source_ent_sent[s].setdefault(p, [])
        source_ent_sent[s][p].append(intro[key])

    plt.rcParams["figure.figsize"] = (20, 20)

    f, axes = plt.subplots(5, 5, sharex=True, sharey=True)

    source_list = list(source_ent_sent.keys())
    source_ent_sent_list = [np.hstack(list(source_ent_sent[source].values())) for source in source_list]
    for i in range(5):
        for j in range(5):

            idx = (i * 5) + j
            if idx >= len(source_ent_sent):
                break

            axes[i][j].hist(source_ent_sent_list[idx])
            axes[i][j].set_title(source_list[idx])

    plt.suptitle("Introduction Sentiments across Sources")

    plt.show()

    plt.rcParams["figure.figsize"] = before_size
