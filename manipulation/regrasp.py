import uuid
import itertools
import numpy as np
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt


class FSRegraspPlanner(object):

    def __init__(self, fsregspot_col=None):
        if fsregspot_col is not None:
            self.build_fsreg_graph(fsregspot_col)
        self.fsreg_graph = nx.Graph()
        # self.fsregspot_col = None

    @property
    def robot(self):
        return self.freregspot_col

    def build_fsreg_graph(self, fsregspot_col):
        """
        build graph from ttpg_col
        :param ttpg_col:
        :return:
        """
        # self.fsregspot_col = fsregspot_col
        plot_grasp_radius = .01
        plot_pose_radius = 0.05
        n_reference_poses = len(fsregspot_col.reference_fsp_poses)
        n_reference_grasps = len(fsregspot_col.reference_grasp_collection)
        p_angle_interval = np.pi * 2 / n_reference_poses
        g_angle_interval = np.pi * 2 / n_reference_grasps
        global_nodes_by_gid = [[] for _ in range(len(fsregspot_col.reference_grasp_collection))]
        for fsregspot in fsregspot_col:
            spot_x = fsregspot.spot_pos[0]
            spot_y = fsregspot.spot_pos[1]
            for fspg in fsregspot.fspgs:
                plot_pose_x = spot_x + plot_pose_radius * np.sin(fspg.fsp_pose_id * p_angle_interval)
                plot_pose_y = spot_y + plot_pose_radius * np.cos(fspg.fsp_pose_id * p_angle_interval)
                local_nodes = []
                obj_pose = fspg.obj_pose
                for gid, grasp, jnt_values in zip(fspg.feasible_gids, fspg.feasible_grasps, fspg.feasible_jv_list):
                    local_nodes.append(uuid.uuid4())
                    plot_grasp_x = plot_pose_x + plot_grasp_radius * np.sin(gid * g_angle_interval)
                    plot_grasp_y = plot_pose_y + plot_grasp_radius * np.cos(gid * g_angle_interval)
                    self.fsreg_graph.add_node(local_nodes[-1],
                                              obj_pose=obj_pose,
                                              grasp=grasp,
                                              jnt_values=jnt_values,
                                              plot_xy=(plot_grasp_x, plot_grasp_y))
                    global_nodes_by_gid[gid].append(local_nodes[-1])
                for node_pair in itertools.combinations(local_nodes, 2):
                    self.fsreg_graph.add_edge(node_pair[0], node_pair[1], type='transit')
        for global_nodes in tqdm(global_nodes_by_gid):
            for global_node_pair in itertools.combinations(global_nodes, 2):
                self.fsreg_graph.add_edge(global_node_pair[0], global_node_pair[1], type='transfer')

    def draw_fsreg_graph(self):
        # for fsregspot in self.fsregspot_col:
        #     spot_x = fsregspot.spot_pos[0]
        #     spot_y = fsregspot.spot_pos[1]
        #     plt.plot(spot_x, spot_y, 'ko')
        for node_tuple in self.fsreg_graph.edges:
            node1_plot_xy = self.fsreg_graph.nodes[node_tuple[0]]['plot_xy']
            node2_plot_xy = self.fsreg_graph.nodes[node_tuple[1]]['plot_xy']
            if self.fsreg_graph.edges[node_tuple]['type'] == 'transit':
                plt.plot([node1_plot_xy[0], node2_plot_xy[0]], [node1_plot_xy[1], node2_plot_xy[1]], 'r-')
            elif self.fsreg_graph.edges[node_tuple]['type'] == 'transfer':
                plt.plot([node1_plot_xy[0], node2_plot_xy[0]], [node1_plot_xy[1], node2_plot_xy[1]], 'b-')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def plan(self, start_fsp_pose_id, goal_fsp_pos_id):
        """

        :param start_fsp_pose_id:
        :param goal_fsp_pos_id:
        :return:
        """

        pass

    def gen_regrasp_motion(robot, path_on_fsreg_graph):
        """
        """
        pass


if __name__ == '__main__':
    import robot_sim.robots.xarmlite6_wg.x6wg2 as x6wg2
    import matplotlib.pyplot as plt
    import manipulation.placement as mpl

    robot = x6wg2.XArmLite6WG2(enable_cc=True)
    fspg_col = mpl.FSPGCollection.load_from_disk("x6wg2_bunny_fspg_col.pickle")
    regrasp_planner = FSRegraspPlanner(robot)
    regrasp_planner.build_fsreg_graph(fspg_col)

    # nx.draw(ttreg_graph, with_labels=True, node_color='skyblue', node_size=1000, font_size=12, font_weight='bold')
    # Show the plot

    plt.show()
