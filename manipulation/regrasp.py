import uuid
import itertools
import networkx as nx
from tqdm import tqdm
import grasping.reasoner as gr


class FSRegraspPlanner(object):

    def __init__(self, robot):
        self.robot = robot
        self.grasp_reasoner = gr.GraspReasoner(robot)
        self.fsreg_graph = nx.Graph()

    def build_fsreg_graph(self, fsregspot_col):
        """
        build graph from ttpg_col
        :param ttpg_col:
        :return:
        """
        global_nodes_by_gid = [[] for _ in range(len(fsregspot_col.reference_grasp_collection))]
        for fsregspot in fsregspot_col:
            for fspg in fsregspot.fspg_list:
                local_nodes = []
                obj_pose = fspg.obj_pose
                for gid, grasp, jnt_values in zip(fspg.feasible_gids, fspg.feasible_grasps, fspg.feasible_jv_list):
                    local_nodes.append(uuid.uuid4())
                    self.fsreg_graph.add_node(local_nodes[-1],
                                              obj_pose=obj_pose,
                                              grasp=grasp,
                                              jnt_values=jnt_values)
                    global_nodes_by_gid[gid].append(local_nodes[-1])
                for node_pair in itertools.combinations(local_nodes, 2):
                    self.fsreg_graph.add_edge(node_pair[0], node_pair[1], edgetype='transit')
        for global_nodes in tqdm(global_nodes_by_gid):
            for global_node_pair in itertools.combinations(global_nodes, 2):
                self.fsreg_graph.add_edge(global_node_pair[0], global_node_pair[1], edgetype='transfer')

    def plan(self, start_obj_pose, goal_obj_pose, reference_grasp_collection):
        """
        plan regrasp motion
        :param start_obj_pose: (start_obj_pos, start_obj_rotmat)
        :param goal_obj_pose: (goal_obj_pos, goal_obj_rotmat)
        :return:
        """
        start_feasible_gids, start_feasible_grasps, start_feasible_jv_list = self.grasp_reasoner.find_feasible_gids(
            reference_grasp_collection=reference_grasp_collection,
            obstacle_list=[],
            goal_pose=start_obj_pose,
            consider_robot=True,
            toggle_keep=True,
            toggle_dbg=False)
        goal_feasible_gids, goal_feasible_grasps, goal_feasible_jv_list = self.grasp_reasoner.find_feasible_gids(
            reference_grasp_collection=reference_grasp_collection,
            obstacle_list=[],
            goal_pose=start_obj_pose,
            consider_robot=True,
            toggle_keep=True,
            toggle_dbg=False)
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
