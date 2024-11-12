import uuid, networkx, itertools
import matplotlib.pyplot as plt
import networkx as nx

import wrs.basis.robot_math as rm
import wrs.motion.motion_data as motd
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
        self.sender_fsreg_planner.add_start_pose(obj_pose=start_pose)
        self.sender_fsreg_planner.add_goal_pose(obj_pose=goal_pose)
        self.receiver_fsreg_planner.add_start_pose(obj_pose=start_pose)
        self.receiver_fsreg_planner.add_goal_pose(obj_pose=goal_pose)
        # merge graphs
        nx.set_node_attributes(self._graph, name="graph_id", values="handover")
        nx.set_node_attributes(self.sender_fsreg_planner.graph, name="graph_id", values="sender")
        nx.set_node_attributes(self.receiver_fsreg_planner.graph, name="graph_id", values="receiver")
        # for node in self._graph.nodes():
        #     self._graph[node]['graph_id'] = "handover"
        # for node in self.sender_fsreg_planner.graph.nodes():
        #     self.sender_fsreg_planner.graph[node]['graph_id']="sender"
        # for node in self.receiver_fsreg_planner.graph.nodes():
        #     self.receiver_fsreg_planner.graph[node]['graph_id']="receiver"
        self._graph.add_nodes_from(self.sender_fsreg_planner.graph.nodes(data=True))
        self._graph.add_edges_from(self.sender_fsreg_planner.graph.edges(data=True))
        self._graph.add_nodes_from(self.receiver_fsreg_planner.graph.nodes(data=True))
        self._graph.add_edges_from(self.receiver_fsreg_planner.graph.edges(data=True))
        for i in range(len(self.sender_reference_gc)):
            for node_pair in itertools.product(self._sender_gl_nodes_by_gid[i], self.sender_fsreg_planner.gl_nodes_by_gid[i]):
                self._graph.add_edge(node_pair[0], node_pair[1], type='transfer')
        for i in range(len(self.receiver_reference_gc)):
            for node_pair in itertools.product(self._receiver_gl_nodes_by_gid[i],
                                               self.receiver_fsreg_planner.gl_nodes_by_gid[i]):
                self._graph.add_edge(node_pair[0], node_pair[1], type='transfer')

        self.show_graph()


        start_node_list = self.add_start_pose(obj_pose=start_pose, obstacle_list=obstacle_list)
        goal_node_list = self.add_goal_pose(obj_pose=goal_pose, obstacle_list=obstacle_list)
        self.show_graph()
        while True:
            min_path = None
            for start in start_node_list:
                for goal in goal_node_list:
                    path = networkx.shortest_path(self._graph, source=start, target=goal)
                    min_path = path if min_path is None else path if len(path) < len(min_path) else min_path
            result = self.gen_regrasp_motion(path=min_path, obstacle_list=obstacle_list, toggle_dbg=toggle_dbg)
            print(result)
            if result[0][0] == 's':  # success
                return result[1]
            if result[0][0] == 'n':  # node failure
                print("Node failure at node: ", result[1])
                self._graph.remove_node(result[1])
                if result[1] in start_node_list:
                    start_node_list.remove(result[1])
            elif result[0][0] == 'e':  # edge failure
                print("Edge failure at edge: ", result[1])
                self._graph.remove_edge(*result[1])
                self._graph.remove_node(result[1][0])
                self._graph.remove_node(result[1][1])
                if result[1][0] in start_node_list:
                    start_node_list.remove(result[1][0])
                if result[1][1] in goal_node_list:
                    goal_node_list.remove(result[1][1])
            print("####################### Update Graph #######################")

    @ppp.adp.mpi.InterplatedMotion.keep_states_decorator
    def gen_regrasp_motion(self, path, obstacle_list,
                           linear_distance=.05,
                           granularity=.03,
                           toggle_dbg=False):
        regraps_motion = motd.MotionData(robot=self.robot)
        for i, node in enumerate(path):
            obj_pose = self._graph.nodes[node]['obj_pose']
            jnt_values = self._graph.nodes[node]['jnt_values']
            # make a copy to keep original movement
            obj_cmodel_copy = self.obj_cmodel.copy()
            obj_cmodel_copy.pose = obj_pose
            # self.robot.goto_given_conf(jnt_values=jnt_values, ee_values=grasp.ee_values)
            if toggle_dbg:
                self.robot.gen_meshmodel().attach_to(base)
            if i == 0:
                jnt_values = self._graph.nodes[path[i]]['jnt_values']
                start_jnt_values = self.robot.get_jnt_values()
                goal_jnt_values = jnt_values
                pick = self.pp_planner.gen_approach_to_given_conf(goal_jnt_values=goal_jnt_values,
                                                                  start_jnt_values=start_jnt_values,
                                                                  linear_distance=.05,
                                                                  ee_values=self.robot.end_effector.jaw_range[1],
                                                                  obstacle_list=obstacle_list,
                                                                  object_list=[obj_cmodel_copy],
                                                                  use_rrt=True,
                                                                  toggle_dbg=toggle_dbg)
                if pick is None:
                    return (f"node failure at {i}", path[i])
                for robot_mesh in pick.mesh_list:
                    obj_cmodel_copy.attach_to(robot_mesh)
                regraps_motion += pick
            if i >= 1:
                prev_node = path[i - 1]
                prev_grasp = self._graph.nodes[prev_node]['grasp']
                prev_jnt_values = self._graph.nodes[prev_node]['jnt_values']
                if self._graph.edges[(prev_node, node)]['type'].endswith('transit'):
                    if toggle_dbg:
                        # show transit poses
                        tmp_obj = obj_cmodel_copy.copy()
                        tmp_obj.rgb = rm.const.pink
                        tmp_obj.alpha = .3
                        tmp_obj.attach_to(base)
                        tmp_obj.show_cdprim()
                    prev2current = self.pp_planner.gen_depart_approach_with_given_conf(start_jnt_values=prev_jnt_values,
                                                                                       end_jnt_values=jnt_values,
                                                                                       depart_direction=None,
                                                                                       depart_distance=.05,
                                                                                       depart_ee_values=
                                                                                       self.robot.end_effector.jaw_range[
                                                                                           1],
                                                                                       approach_direction=None,
                                                                                       approach_distance=.05,
                                                                                       approach_ee_values=
                                                                                       self.robot.end_effector.jaw_range[
                                                                                           1],
                                                                                       granularity=granularity,
                                                                                       obstacle_list=obstacle_list,
                                                                                       object_list=[obj_cmodel_copy],
                                                                                       use_rrt=True,
                                                                                       toggle_dbg=False)
                    if prev2current is None:
                        return (f"edge failure at transit {i - 1}-{i}", (path[i - 1], path[i]))
                    for robot_mesh in prev2current.mesh_list:
                        obj_cmodel_copy.attach_to(robot_mesh)
                    regraps_motion += prev2current
                if self._graph.edges[(prev_node, node)]['type'].endswith('transfer'):
                    self.robot.goto_given_conf(prev_jnt_values)
                    obj_cmodel_copy.pose = self._graph.nodes[prev_node]['obj_pose']
                    self.robot.hold(obj_cmodel=obj_cmodel_copy, jaw_width=prev_grasp.ee_values)
                    regraps_motion.extend(jv_list=[prev_jnt_values],
                                          ev_list=[prev_grasp.ee_values],
                                          mesh_list=[self.robot.gen_meshmodel()])
                    prev2current = self.pp_planner.gen_depart_approach_with_given_conf(start_jnt_values=prev_jnt_values,
                                                                                       end_jnt_values=jnt_values,
                                                                                       depart_direction=rm.const.z_ax,
                                                                                       depart_distance=linear_distance,
                                                                                       approach_direction=-rm.const.z_ax,
                                                                                       approach_distance=linear_distance,
                                                                                       granularity=granularity,
                                                                                       obstacle_list=obstacle_list,
                                                                                       use_rrt=True,
                                                                                       toggle_dbg=False)
                    # prev2current = self.pp_planner.im_planner.gen_interplated_between_given_conf(
                    #     start_jnt_values=prev_jnt_values,
                    #     end_jnt_values=jnt_values,
                    #     obstacle_list=[])
                    if prev2current is None:
                        return (f"edge failure at transfer {i - 1}-{i}", (path[i - 1], path[i]))
                    regraps_motion += prev2current
                    self.robot.goto_given_conf(prev2current.jv_list[-1])
                    if toggle_dbg:
                        self.robot.gen_meshmodel(rgb=rm.const.red).attach_to(base)
                    self.robot.release(obj_cmodel=obj_cmodel_copy, jaw_width=self.robot.end_effector.jaw_range[1])
                    mesh = self.robot.gen_meshmodel()
                    obj_cmodel_copy.attach_to(mesh)
                    regraps_motion.extend(jv_list=[prev2current.jv_list[-1]],
                                          ev_list=[self.robot.end_effector.jaw_range[1]],
                                          mesh_list=[mesh])
        return ("success", regraps_motion)

    def add_hopg_collection_from_disk(self, file_name):
        hopg_collection = self._hopg_collection.copy()
        hopg_collection.load_from_disk(file_name)
        self._hopg_collection += hopg_collection
        self._add_hopg_collection_to_graph(hopg_collection)

    def _add_hopg_collection_to_graph(self, hopg_collection):
        # new_sender_gl_nodes_by_gid = {}
        # for id in range(len(self.sender_reference_gc)):
        #     new_sender_gl_nodes_by_gid[id] = []
        # new_receiver_gl_nodes_by_gid = {}
        # for id in range(len(self.receiver_reference_gc)):
        #     new_receiver_gl_nodes_by_gid[id] = []
        for hopg in hopg_collection:
            # each hopg includes several sid-rid_list pairs, the rid_lists may have repeatitions
            pos_x = hopg.obj_pose[0][0]
            pos_y = hopg.obj_pose[0][1]
            # sender
            sender_node = uuid.uuid4()
            plot_pos_x = pos_x + self._plot_x_offset
            plot_pos_y = pos_y + self._plot_y_offset * hopg.sender_gid
            self._graph.add_node(sender_node,
                                 obj_pose=hopg.obj_pose,
                                 grasp=hopg.sender_grasp,
                                 jnt_values=hopg.sender_conf,
                                 plot_xy=(plot_pos_x, plot_pos_y))
            self._sender_gl_nodes_by_gid[hopg.sender_gid].append(sender_node)
            # receiver
            receiver_gid2node = {}
            for rid, receiver_gid in enumerate(hopg.receiver_gids):
                if receiver_gid not in receiver_gid2node.keys():
                    receiver_gid2node[receiver_gid] = uuid.uuid4()
                    plot_pos_x = pos_x - self._plot_x_offset
                    plot_pos_y = pos_y + self._plot_y_offset * receiver_gid
                    self._graph.add_node(receiver_gid2node[receiver_gid],
                                         obj_pose=hopg.obj_pose,
                                         grasp=hopg.receiver_grasps[rid],
                                         jnt_values=hopg.receiver_confs[rid],
                                         plot_xy=(plot_pos_x, plot_pos_y))
                    self._receiver_gl_nodes_by_gid[receiver_gid].append(receiver_gid2node[receiver_gid])
                self._graph.add_edge(sender_node, receiver_gid2node[receiver_gid], type='handover')
        # for i in range(len(self.sender_reference_gc)):
        #     for node_pair in itertools.product(new_sender_gl_nodes_by_gid[i], self._sender_gl_nodes_by_gid[i]):
        #         if node_pair[0] != node_pair[1]:
        #             self._graph.add_edge(node_pair[0], node_pair[1], type='transit')
        #     self._sender_gl_nodes_by_gid[i].extend(new_sender_gl_nodes_by_gid[i])
        # for i in range(len(self.receiver_reference_gc)):
        #     for node_pair in itertools.product(new_receiver_gl_nodes_by_gid[i], self._receiver_gl_nodes_by_gid[i]):
        #         if node_pair[0] != node_pair[1]:
        #             self._graph.add_edge(node_pair[0], node_pair[1], type='transit')
        #     self._receiver_gl_nodes_by_gid[receiver_gid].extend(new_receiver_gl_nodes_by_gid[i)

    def draw_graph(self):
        # for regspot in self.regspot_col:
        #     spot_x = regspot.spot_pos[0]
        #     spot_y = regspot.spot_pos[1]
        #     plt.plot(spot_x, spot_y, 'ko')
        for node_tuple in self._graph.edges:
            print(self._graph.nodes[node_tuple[0]])
            node1_plot_xy = self._graph.nodes[node_tuple[0]]['plot_xy']
            node2_plot_xy = self._graph.nodes[node_tuple[1]]['plot_xy']
            print(node1_plot_xy, node2_plot_xy)
            if self._graph.edges[node_tuple]['type'] == 'transit':
                plt.plot([node1_plot_xy[0], node2_plot_xy[0]], [node1_plot_xy[1], node2_plot_xy[1]], 'c-')
            elif self._graph.edges[node_tuple]['type'] == 'transfer':
                plt.plot([node1_plot_xy[0], node2_plot_xy[0]], [node1_plot_xy[1], node2_plot_xy[1]], 'k-')
            elif self._graph.edges[node_tuple]['type'] == 'handover':
                plt.plot([node1_plot_xy[0], node2_plot_xy[0]], [node1_plot_xy[1], node2_plot_xy[1]], 'y-')
            elif self._graph.edges[node_tuple]['type'] == 'start_transit':
                plt.plot([node1_plot_xy[0], node2_plot_xy[0]], [node1_plot_xy[1], node2_plot_xy[1]], 'c-')
            elif self._graph.edges[node_tuple]['type'] == 'start_transfer':
                plt.plot([node1_plot_xy[0], node2_plot_xy[0]], [node1_plot_xy[1], node2_plot_xy[1]], 'r-')
            elif self._graph.edges[node_tuple]['type'] == 'goal_transit':
                plt.plot([node1_plot_xy[0], node2_plot_xy[0]], [node1_plot_xy[1], node2_plot_xy[1]], 'c-')
            elif self._graph.edges[node_tuple]['type'] == 'goal_transfer':
                plt.plot([node1_plot_xy[0], node2_plot_xy[0]], [node1_plot_xy[1], node2_plot_xy[1]], 'b-')
        plt.gca().set_aspect('equal', adjustable='box')

    def show_graph(self):
        self.draw_graph()
        plt.show()
