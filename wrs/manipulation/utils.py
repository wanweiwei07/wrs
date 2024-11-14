import matplotlib.pyplot as plt


def draw_graph(graph):
    for node_tuple in graph.edges:
        node1_plot_xy = graph.nodes[node_tuple[0]]['plot_xy']
        node2_plot_xy = graph.nodes[node_tuple[1]]['plot_xy']
        print(node1_plot_xy, node2_plot_xy)
        if graph.edges[node_tuple]['type'] == 'transit':
            plt.plot([node1_plot_xy[0], node2_plot_xy[0]], [node1_plot_xy[1], node2_plot_xy[1]], 'c-')
        elif graph.edges[node_tuple]['type'] == 'transfer':
            plt.plot([node1_plot_xy[0], node2_plot_xy[0]], [node1_plot_xy[1], node2_plot_xy[1]], 'k-')
        elif graph.edges[node_tuple]['type'] == 'handover':
            plt.plot([node1_plot_xy[0], node2_plot_xy[0]], [node1_plot_xy[1], node2_plot_xy[1]], 'y-')
        elif graph.edges[node_tuple]['type'] == 'start_transit':
            plt.plot([node1_plot_xy[0], node2_plot_xy[0]], [node1_plot_xy[1], node2_plot_xy[1]], 'c-')
        elif graph.edges[node_tuple]['type'] == 'start_transfer':
            plt.plot([node1_plot_xy[0], node2_plot_xy[0]], [node1_plot_xy[1], node2_plot_xy[1]], 'r-')
        elif graph.edges[node_tuple]['type'] == 'goal_transit':
            plt.plot([node1_plot_xy[0], node2_plot_xy[0]], [node1_plot_xy[1], node2_plot_xy[1]], 'c-')
        elif graph.edges[node_tuple]['type'] == 'goal_transfer':
            plt.plot([node1_plot_xy[0], node2_plot_xy[0]], [node1_plot_xy[1], node2_plot_xy[1]], 'b-')
    plt.gca().set_aspect('equal', adjustable='box')

def draw_path(graph, path):
    n_nodes_on_path = len(path)
    for i in range(1, n_nodes_on_path):
        node1_plot_xy = graph.nodes[path[i]]['plot_xy']
        node2_plot_xy = graph.nodes[path[i - 1]]['plot_xy']
        plt.plot([node1_plot_xy[0], node2_plot_xy[0]], [node1_plot_xy[1], node2_plot_xy[1]], 'r-', linewidth=2)

def show_graph(graph):
    draw_graph(graph)
    plt.show()

def show_graph_with_path(graph, path_list):
    draw_graph(graph)
    for path in path_list:
        draw_path(graph, path)
    plt.show()