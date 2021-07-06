import networkx as nx
import numpy as np


def show_graph(G):
    import pylab
    pos = nx.shell_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='white', edge_color='red', node_size=400, alpha=0.5)
    pylab.title('Self_Define Net', fontsize=15)
    pylab.show()


def multi_target_shortest_path(G, source, target_list):
    best_dist = np.inf
    best_path = None
    dist_dict, path_dict = \
        nx.algorithms.shortest_paths.weighted.single_source_dijkstra(G, source=source)
    if path_dict is None:
        return np.inf, None
    for target in target_list:
        try:
            dist = dist_dict[target]
            path = path_dict[target]
            if dist < best_dist:
                best_dist = dist
                best_path = path
        except:
            continue
    return best_dist, best_path


def get_nodes_value(info_dict, node_path, info_key):
    values = []
    for node in node_path:
        k, inx = node.split("_")
        values.append(info_dict[int(k)][info_key][int(inx)])
    return values
