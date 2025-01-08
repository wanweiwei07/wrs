import uuid, networkx, itertools
import matplotlib.pyplot as plt
import networkx

import wrs.basis.robot_math as rm
import wrs.motion.motion_data as motd
import wrs.manipulation.utils as mp_utils
import wrs.manipulation.pick_place as ppp
import wrs.manipulation.placement.common as mp_gp
import wrs.manipulation.placement.handover as mp_hop
import wrs.manipulation.flatsurface_regrasp as fsreg


class HandoverPlanner(object):
    def __init__(self, obj_cmodel, sender_robot, receiver_robot, sender_reference_gc, receiver_reference_gc):
        self.obj_cmodel = obj_cmodel
        self.sender_robot = sender_robot
        self.receiver_robot = receiver_robot
        self._graph = networkx.Graph()
        self._plot_x_offset = .05
        self._plot_y_offset = .001
        self._hopg_collection = mp_hop.HOPGCollection(obj_cmodel=obj_cmodel,
                                                      sender_robot=sender_robot, receiver_robot=receiver_robot,
                                                      sender_reference_gc=sender_reference_gc,
                                                      receiver_reference_gc=receiver_reference_gc)
        self.sender_fsreg_planner = fsreg.FSRegraspPlanner(robot=sender_robot,
                                                           obj_cmodel=obj_cmodel,
                                                           fs_reference_poses=None,
                                                           reference_gc=sender_reference_gc)
        self.receiver_fsreg_planner = fsreg.FSRegraspPlanner(robot=receiver_robot,
                                                             obj_cmodel=obj_cmodel,
                                                             fs_reference_poses=None,
                                                             reference_gc=receiver_reference_gc)
        # dictionary for backtracking sender global connection
        self._sender_gl_nodes_by_gid = {}
        for id in range(len(sender_reference_gc)):
            self._sender_gl_nodes_by_gid[id] = []
        # dictionary for backtracking sender global connection
        self._receiver_gl_nodes_by_gid = {}
        for id in range(len(receiver_reference_gc)):
            self._receiver_gl_nodes_by_gid[id] = []

    @property
    def sender_reference_gc(self):
        return self._hopg_collection.sender_reference_gc

    @property
    def receiver_reference_gc(self):
        return self._hopg_collection.receiver_reference_gc

    def plan_by_obj_poses(self, start_pose, goal_pose, obstacle_list=None, toggle_dbg=False):
        """
        :param start_pose: (pos, rotmat)
        :param goal_pose: (pos, rotmat)
        :return:
        """
        # connect start and goal to sender and receiver graphs respectively
        start_node_list = self.sender_fsreg_planner.add_start_pose(obj_pose=start_pose)
        goal_node_list = self.sender_fsreg_planner.add_goal_pose(obj_pose=goal_pose)
        start_node_list += self.receiver_fsreg_planner.add_start_pose(obj_pose=start_pose)
        goal_node_list += self.receiver_fsreg_planner.add_goal_pose(obj_pose=goal_pose)
        # merge graphs
        # add labels to subgraphs
        networkx.set_node_attributes(self.sender_fsreg_planner.graph, name="robot_name", values=self.sender_robot.name)
        networkx.set_node_attributes(self.receiver_fsreg_planner.graph, name="robot_name",
                                     values=self.receiver_robot.name)
        # add nodes and edges from subgraphs to self._graph
        self._graph.add_nodes_from(self.sender_fsreg_planner.graph.nodes(data=True))
        self._graph.add_edges_from(self.sender_fsreg_planner.graph.edges(data=True))
        self._graph.add_nodes_from(self.receiver_fsreg_planner.graph.nodes(data=True))
        self._graph.add_edges_from(self.receiver_fsreg_planner.graph.edges(data=True))
        # connect subgraphs by adding new edges
        for i in range(len(self.sender_reference_gc)):
            for node_pair in itertools.product(self._sender_gl_nodes_by_gid[i],
                                               self.sender_fsreg_planner.gl_nodes_by_gid[i]):
                self._graph.add_edge(node_pair[0], node_pair[1], type='transfer')
                # add edges to the prepared nodes for easier motion generation
                self.sender_fsreg_planner.graph.add_edge(node_pair[0], node_pair[1], type='transfer')
        for i in range(len(self.receiver_reference_gc)):
            for node_pair in itertools.product(self._receiver_gl_nodes_by_gid[i],
                                               self.receiver_fsreg_planner.gl_nodes_by_gid[i]):
                self._graph.add_edge(node_pair[0], node_pair[1], type='transfer')
                # add edges to the prepared nodes for easier motion generation
                self.receiver_fsreg_planner.graph.add_edge(node_pair[0], node_pair[1], type='transfer')
        self.show_graph()
        self.sender_fsreg_planner.show_graph()
        self.receiver_fsreg_planner.show_graph()
        # search paths
        while True:
            min_path = None
            for start in start_node_list:
                for goal in goal_node_list:
                    try:
                        path = networkx.shortest_path(self._graph, source=start, target=goal)
                        min_path = path if min_path is None else path if len(path) < len(min_path) else min_path
                    except networkx.NetworkXNoPath:
                        # print(f"No path exists between {start} and {goal}")
                        continue
            if min_path is None:
                return None
            # segment the path
            sub_paths = []
            current_sub_path = [min_path[0]]
            current_robot_name = self._graph.nodes[min_path[0]]["robot_name"]
            for node_uuid in min_path[1:]:
                robot_name = self._graph.nodes[node_uuid]["robot_name"]
                if robot_name == current_robot_name:
                    current_sub_path.append(node_uuid)
                else:
                    sub_paths.append(current_sub_path)
                    current_sub_path = [node_uuid]
                    current_robot_name = robot_name
            else:
                sub_paths.append(current_sub_path)
            # gen regrasp motion
            print(sub_paths)
            motion_list = []
            for i, path in enumerate(sub_paths):
                if self._graph.nodes[path[0]]["robot_name"] == self.sender_robot.name:
                    reg_planner = self.sender_fsreg_planner
                else:
                    reg_planner = self.receiver_fsreg_planner
                result = reg_planner.gen_regrasp_motion(path=path, obstacle_list=obstacle_list,
                                                        toggle_start_approach=False,
                                                        toggle_end_depart=False,
                                                        toggle_dbg=toggle_dbg)
                print(result)
                if result[0][0] == 's':  # success
                    motion_list.append(result[1])
                if result[0][0] == 'n':  # node failure
                    print("Node failure at node: ", result[1])
                    self._graph.remove_node(result[1])
                    if result[1] in start_node_list:
                        start_node_list.remove(result[1])
                    break
                elif result[0][0] == 'e':  # edge failure
                    print("Edge failure at edge: ", result[1])
                    self._graph.remove_edge(*result[1])
                    self._graph.remove_node(result[1][0])
                    self._graph.remove_node(result[1][1])
                    if result[1][0] in start_node_list:
                        start_node_list.remove(result[1][0])
                    if result[1][1] in goal_node_list:
                        goal_node_list.remove(result[1][1])
                    break
            else:
                # correct handover details
                return motion_list
            print("####################### Update Graph #######################")

    def add_hopg_collection_from_disk(self, file_name):
        hopg_collection = self._hopg_collection.copy()
        hopg_collection.load_from_disk(file_name)
        self._hopg_collection += hopg_collection
        self._add_hopg_collection_to_graph(hopg_collection)

    def _add_hopg_collection_to_graph(self, hopg_collection):
        for hopg in hopg_collection:
            # each hopg includes several sid-rid_list pairs, the rid_lists may have repeatitions
            pos_x = hopg.obj_pose[0][1]
            pos_y = -hopg.obj_pose[0][0]
            # sender
            sender_node = uuid.uuid4()
            plot_pos_x = pos_x + rm.sign(self.sender_robot.pos[1]) * self._plot_x_offset
            plot_pos_y = pos_y + self._plot_y_offset * hopg.sender_gid
            self._graph.add_node(sender_node,
                                 obj_pose=hopg.obj_pose,
                                 grasp=hopg.sender_grasp,
                                 jnt_values=hopg.sender_conf,
                                 plot_xy=(plot_pos_x, plot_pos_y),
                                 robot_name=self.sender_robot.name)
            # prepare a copy for easier motion generation, edges will be added later
            self.sender_fsreg_planner.graph.add_node(sender_node, **self._graph.nodes[sender_node])
            self._sender_gl_nodes_by_gid[hopg.sender_gid].append(sender_node)
            # receiver
            receiver_gid2node = {}
            for rid, receiver_gid in enumerate(hopg.receiver_gids):
                if receiver_gid not in receiver_gid2node.keys():
                    receiver_gid2node[receiver_gid] = uuid.uuid4()
                    plot_pos_x = pos_x + rm.sign(self.receiver_robot.pos[1]) * self._plot_x_offset
                    plot_pos_y = pos_y + self._plot_y_offset * receiver_gid
                    self._graph.add_node(receiver_gid2node[receiver_gid],
                                         obj_pose=hopg.obj_pose,
                                         grasp=hopg.receiver_grasps[rid],
                                         jnt_values=hopg.receiver_confs[rid],
                                         plot_xy=(plot_pos_x, plot_pos_y),
                                         robot_name=self.receiver_robot.name)
                    # prepare a copy for easier motion generation, edges will be added later
                    self.receiver_fsreg_planner.graph.add_node(receiver_gid2node[receiver_gid],
                                                               **self._graph.nodes[receiver_gid2node[receiver_gid]])
                    self._receiver_gl_nodes_by_gid[receiver_gid].append(receiver_gid2node[receiver_gid])
                self._graph.add_edge(sender_node, receiver_gid2node[receiver_gid], type='handover')

    def show_graph(self):
        mp_utils.show_graph(self._graph)
