import uuid
import itertools
import networkx as nx
from tqdm import tqdm

def build_ttreg_graph(ttpg_col):
    """
    build graph from ttpg_col
    :param ttpg_col:
    :return:
    """
    graph = nx.Graph()
    global_nodes_by_gid = [[] for _ in range(len(ttpg_col.reference_grasp_collection))]
    for ttpg in ttpg_col:
        local_nodes = []
        placement_pose = ttpg_col.get_placement_pose_by_ttpg(ttpg)
        print(len(ttpg.feasible_gids))
        for gid in ttpg.feasible_gids:
            grasp = ttpg_col.reference_grasp_collection[gid]
            local_nodes.append(uuid.uuid4())
            graph.add_node(local_nodes[-1],
                           placement_pose=placement_pose,
                           grasp=grasp)
            global_nodes_by_gid[gid].append(local_nodes[-1])
        for global_nodes in global_nodes_by_gid:
            print(len(global_nodes))
        for node_pair in itertools.combinations(local_nodes, 2):
            graph.add_edge(node_pair[0], node_pair[1], edgetype='transit')
    for global_nodes in tqdm(global_nodes_by_gid):
        print(len(global_nodes))
        for global_node_pair in itertools.combinations(global_nodes, 2):
            graph.add_edge(global_node_pair[0], global_node_pair[1], edgetype='transfer')
    return graph


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import manipulation.placement as mpl

    ttpg_col =mpl.TTPGCollection.load_from_disk("x6wg2_bunny_ttpg_col.pickle")
    ttreg_graph = build_ttreg_graph(ttpg_col)
    nx.draw(ttreg_graph, with_labels=True, node_color='skyblue', node_size=1000, font_size=12, font_weight='bold')
    # Show the plot
    plt.show()