import uuid, networkx, itertools
import matplotlib.pyplot as plt
import wrs.manipulation.pick_place as ppp
import wrs.manipulation.placement.handover as mp_hop


class HandoverPlanner(object):
    def __init__(self, obj_cmodel, sender_robot, receiver_robot,
                 sender_reference_grasps, receiver_reference_grasps):
        self.obj_cmodel = obj_cmodel
        self.sender_robot = sender_robot
        self.receiver_robot = receiver_robot
        self._graph = networkx.Graph()
        self._plot_x_offset = .05
        self._plot_y_offset = .001
        self._hopg_collection = mp_hop.HOPGCollection(obj_cmodel=obj_cmodel,
                                                      sender_robot=sender_robot, receiver_robot=receiver_robot,
                                                      sender_reference_gc=sender_reference_grasps,
                                                      receiver_reference_gc=receiver_reference_grasps)
        self.sender_ppp = ppp.PickPlacePlanner(sender_robot)
        # dictionary for backtracking sender global connection
        self._sender_gl_nodes_by_gid = {}
        for id in range(len(sender_reference_grasps)):
            self._sender_gl_nodes_by_gid[id] = []
        # dictionary for backtracking sender global connection
        self._receiver_gl_nodes_by_gid = {}
        for id in range(len(receiver_reference_grasps)):
            self._receiver_gl_nodes_by_gid[id] = []

    @property
    def sender_reference_grasp_collection(self):
        return self._hopg_collection.sender_reference_gc

    @property
    def receiver_reference_grasp_collection(self):
        return self._hopg_collection.receiver_reference_gc

    def add_hopg_collection_from_disk(self, file_name):
        hopg_collection = self._hopg_collection.copy()
        hopg_collection.load_from_disk(file_name)
        self._hopg_collection += hopg_collection
        self._add_hopg_collection_to_graph(hopg_collection)

    def _add_hopg_collection_to_graph(self, hopg_collection):
        new_sender_gl_nodes_by_gid = {}
        for id in range(len(self.sender_reference_grasp_collection)):
            new_sender_gl_nodes_by_gid[id] = []
        new_receiver_gl_nodes_by_gid = {}
        for id in range(len(self.receiver_reference_grasp_collection)):
            new_receiver_gl_nodes_by_gid[id] = []
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
            new_sender_gl_nodes_by_gid[hopg.sender_gid].append(sender_node)
            # self._sender_gl_nodes_by_gid[hopg.sender_gid].append(sender_node)
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
                    new_receiver_gl_nodes_by_gid[receiver_gid].append(receiver_gid2node[receiver_gid])
                    self._receiver_gl_nodes_by_gid[receiver_gid].append(receiver_gid2node[receiver_gid])
                self._graph.add_edge(sender_node, receiver_gid2node[receiver_gid], type='handover')
        for i in range(len(self.sender_reference_grasp_collection)):
            new_sender_gl_nodes = new_sender_gl_nodes_by_gid[i]
            original_sender_gl_nodes = self._sender_gl_nodes_by_gid[i]
            for node_pair in itertools.product(new_sender_gl_nodes, original_sender_gl_nodes):
                if node_pair[0] != node_pair[1]:
                    self._graph.add_edge(node_pair[0], node_pair[1], type='transit')
        for i in range(len(self.receiver_reference_grasp_collection)):
            new_receiver_gl_nodes = new_receiver_gl_nodes_by_gid[i]
            original_receiver_gl_nodes = self._receiver_gl_nodes_by_gid[i]
            for node_pair in itertools.product(new_receiver_gl_nodes, original_receiver_gl_nodes):
                if node_pair[0] != node_pair[1]:
                    self._graph.add_edge(node_pair[0], node_pair[1], type='transit')

    # def _add_fspg_to_graph(self, fspg, plot_pose_xy, prefix=''):
    #     """
    #     add a fspg to the regrasp graph
    #     :param fspg:
    #     :param plot_pose_xy: specify where to plot the fspg
    #     :return:
    #     """
    #     new_global_nodes_by_gid = [[] for _ in range(len(self.reference_gc))]
    #     local_nodes = []
    #     obj_pose = fspg.obj_pose
    #     for gid, grasp, jnt_values in zip(fspg.feasible_gids, fspg.feasible_grasps, fspg.feasible_confs):
    #         local_nodes.append(uuid.uuid4())
    #         plot_grasp_x = plot_pose_xy[0] + self._plot_g_radius * rm.sin(gid * self._g_angle_interval)
    #         plot_grasp_y = plot_pose_xy[1] + self._plot_g_radius * rm.cos(gid * self._g_angle_interval)
    #         self._graph.add_node(local_nodes[-1],
    #                              obj_pose=obj_pose,
    #                              grasp=grasp,
    #                              jnt_values=jnt_values,
    #                              plot_xy=(plot_grasp_x, plot_grasp_y))
    #         new_global_nodes_by_gid[gid].append(local_nodes[-1])
    #         # self._global_nodes_by_gid[gid].append(local_nodes[-1])
    #     for node_pair in itertools.combinations(local_nodes, 2):
    #         self._graph.add_edge(node_pair[0], node_pair[1], type=prefix + '_transit')
    #     for i in range(len(self.reference_gc)):
    #         new_global_nodes = new_global_nodes_by_gid[i]
    #         original_global_nodes = self._global_nodes_by_gid[i]
    #         for node_pair in itertools.product(new_global_nodes, original_global_nodes):
    #             self._graph.add_edge(node_pair[0], node_pair[1], type=prefix + '_transfer')
    #         original_global_nodes.extend(new_global_nodes)
    #     return local_nodes

    def draw_graph(self):
        # for regspot in self.regspot_col:
        #     spot_x = regspot.spot_pos[0]
        #     spot_y = regspot.spot_pos[1]
        #     plt.plot(spot_x, spot_y, 'ko')
        for node_tuple in self._graph.edges:
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
