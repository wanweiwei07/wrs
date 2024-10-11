import uuid
import pickle
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.grasping.reasoner as gr
import wrs.manipulation.pick_place_planner as ppp
import wrs.manipulation.placement.flat_surface_placement as mpfsp
import wrs.manipulation.placement.general_placement as mpgp
import wrs.modeling.model_collection as mmc


class FSRegSpotCollection(object):
    def __init__(self, robot, obj_cmodel, fs_reference_poses, reference_grasp_collection):
        """
        :param robot:
        :param reference_fsp_poses: an instance of ReferenceFSPPoses
        :param reference_grasp_collection: an instance of GraspCollection
        """
        self.robot = robot
        self.obj_cmodel = obj_cmodel
        self.grasp_reasoner = gr.GraspReasoner(robot)
        self.fs_reference_poses = fs_reference_poses
        self.reference_grasp_collection = reference_grasp_collection
        self._fsregspot_list = []  # list of FSRegSpot

    def load_from_disk(self, file_name="fsregspot_collection.pickle"):
        with open(file_name, 'rb') as file:
            self._fsregspot_list = pickle.load(file)

    def save_to_disk(self, file_name='fsregspot_collection.pickle'):
        """
        :param file_name
        :return:
        """
        with open(file_name, 'wb') as file:
            pickle.dump(self._fsregspot_list, file)

    def __getitem__(self, index):
        return self._fsregspot_list[index]

    def __len__(self):
        return len(self._fsregspot_list)

    def __iter__(self):
        return iter(self._fsregspot_list)

    def __add__(self, other):
        self._fsregspot_list += other._fsregspot_list
        return self

    def add_new_spot(self, spot_pos, spot_rotz, barrier_z_offset=.0, consider_robot=True, toggle_dbg=False):
        fs_regspot = mpfsp.FSRegSpot(spot_pos, spot_rotz)
        barrier_obstacle = mcm.gen_surface_barrier(spot_pos[2] + barrier_z_offset)
        for pose_id, pose in enumerate(self.fs_reference_poses):
            pos = pose[0] + spot_pos
            rotmat = rm.rotmat_from_euler(0, 0, spot_rotz) @ pose[1]
            feasible_gids, feasible_grasps, feasible_jv_list = self.grasp_reasoner.find_feasible_gids(
                reference_grasp_collection=self.reference_grasp_collection,
                obstacle_list=[barrier_obstacle],
                goal_pose=(pos, rotmat),
                consider_robot=consider_robot,
                toggle_keep=True,
                toggle_dbg=False)
            if feasible_gids is not None:
                fs_regspot.add_fspg(mpfsp.FSPG(fs_pose_id=pose_id,
                                               obj_pose=(pos, rotmat),
                                               feasible_gids=feasible_gids,
                                               feasible_grasps=feasible_grasps,
                                               feasible_jv_list=feasible_jv_list))
            if toggle_dbg:
                for grasp, jnt_values in zip(feasible_grasps, feasible_jv_list):
                    self.robot.goto_given_conf(jnt_values=jnt_values, ee_values=grasp.ee_values)
                    self.robot.gen_meshmodel().attach_to(base)
                base.run()
        self._fsregspot_list.append(fs_regspot)

    # TODO keep robot state
    def gen_meshmodel(self):
        """
        TODO do not use explicit obj_cmodel
        :param robot:
        :param fspg_col:
        :return:
        """
        meshmodel_list = []
        for fsreg_spot in self._fsregspot_list:
            for fspg in fsreg_spot:
                m_col = mmc.ModelCollection()
                obj_pose = fspg.obj_pose
                feasible_grasps = fspg.feasible_grasps
                feasible_jv_list = fspg.feasible_jv_list
                obj_cmodel_copy = self.obj_cmodel.copy()
                obj_cmodel_copy.pose = obj_pose
                obj_cmodel_copy.attach_to(m_col)
                for grasp, jnt_values in zip(feasible_grasps, feasible_jv_list):
                    self.robot.goto_given_conf(jnt_values=jnt_values, ee_values=grasp.ee_values)
                    self.robot.gen_meshmodel().attach_to(m_col)
                meshmodel_list.append(m_col)
        return meshmodel_list


class FSRegraspPlanner(object):

    def __init__(self, robot, obj_cmodel, fs_reference_poses, reference_grasp_collection):
        self._graph = nx.Graph()
        self._fsregspot_collection = FSRegSpotCollection(robot=robot,
                                                         obj_cmodel=obj_cmodel,
                                                         fs_reference_poses=fs_reference_poses,
                                                         reference_grasp_collection=reference_grasp_collection)
        self.pp_planner = ppp.PickPlacePlanner(robot)
        self._global_nodes_by_gid = [[] for _ in range(len(reference_grasp_collection))]
        self._plot_g_radius = .01
        self._plot_p_radius = 0.05
        self._n_fs_reference_poses = len(fs_reference_poses)
        self._n_reference_grasps = len(reference_grasp_collection)
        self._p_angle_interval = rm.pi * 2 / self._n_fs_reference_poses
        self._g_angle_interval = rm.pi * 2 / self._n_reference_grasps

    def add_fsregspot_collection_from_disk(self, file_name):
        fsregspot_collection = FSRegSpotCollection(robot=self.robot,
                                                   obj_cmodel=self.obj_cmodel,
                                                   fs_reference_poses=self.fs_reference_poses,
                                                   reference_grasp_collection=self.reference_grasp_collection)
        fsregspot_collection.load_from_disk(file_name)
        self._fsregspot_collection += fsregspot_collection
        self._add_fsregspot_collection_to_graph(fsregspot_collection)

    @property
    def robot(self):
        return self._fsregspot_collection.robot

    @property
    def obj_cmodel(self):
        return self._fsregspot_collection.obj_cmodel

    @property
    def fs_reference_poses(self):
        return self._fsregspot_collection.fs_reference_poses

    @property
    def reference_grasp_collection(self):
        return self._fsregspot_collection.reference_grasp_collection

    # def load_spotfspgs_col_from_disk(self, file_name):
    #     self.regspot_col.load_from_disk(file_name)
    #
    # def save_spotfspgs_col_to_disk(self, file_name):
    #     self.regspot_col.save_to_disk(file_name)

    def save_to_disk(self, file_name):
        pass

    def load_from_disk(self, file_name):
        pass

    def create_add_fsregspot(self, spot_pos, spot_rotz, barrier_z_offset=-.01, consider_robot=True, toggle_dbg=False):
        self._fsregspot_collection.add_new_spot(spot_pos, spot_rotz, barrier_z_offset, consider_robot, toggle_dbg)
        self._add_fsregspot_to_graph(self._fsregspot_collection[-1])

    def add_start_pose(self, obj_pose, obstacle_list=None, plot_pose_xy=None):
        start_pg = mpgp.PG.create_from_pose(self.robot,
                                            self.reference_grasp_collection,
                                            obj_pose,
                                            obstacle_list=obstacle_list,
                                            consider_robot=True)
        if start_pg is None:
            print("No feasible grasps found at the start pose")
            return None
        if plot_pose_xy is None:
            plot_pose_xy = obj_pose[0][:2]
        return self._add_fspg_to_graph(start_pg, plot_pose_xy=plot_pose_xy, prefix='start')

    def add_goal_pose(self, obj_pose, obstacle_list=None, plot_pose_xy=None):
        goal_pg = mpgp.PG.create_from_pose(self.robot,
                                           self.reference_grasp_collection,
                                           obj_pose,
                                           obstacle_list=obstacle_list,
                                           consider_robot=True)
        if goal_pg is None:
            print("No feasible grasps found at the goal pose")
            return None
        if plot_pose_xy is None:
            plot_pose_xy = obj_pose[0][:2]
        return self._add_fspg_to_graph(goal_pg, plot_pose_xy=plot_pose_xy, prefix='goal')

    def _add_fsregspot_collection_to_graph(self, fsregspot_collection):
        """
        add regspot collection to the regrasp graph
        :param fsregspot_collection: an instance of FSRegSpotCollection or a list of FSRegSpot
        :return:
        """
        new_global_nodes_by_gid = [[] for _ in range(len(self.reference_grasp_collection))]
        for fsregspot in fsregspot_collection:
            spot_x = fsregspot.pos[0]
            spot_y = fsregspot.pos[1]
            for fspg in fsregspot.fspg_list:
                plot_pose_x = spot_x + self._plot_p_radius * rm.sin(fspg.fs_pose_id * self._p_angle_interval)
                plot_pose_y = spot_y + self._plot_p_radius * rm.cos(fspg.fs_pose_id * self._p_angle_interval)
                local_nodes = []
                obj_pose = fspg.obj_pose
                for gid, grasp, jnt_values in zip(fspg.feasible_gids, fspg.feasible_grasps, fspg.feasible_jv_list):
                    local_nodes.append(uuid.uuid4())
                    plot_grasp_x = plot_pose_x + self._plot_g_radius * rm.sin(gid * self._g_angle_interval)
                    plot_grasp_y = plot_pose_y + self._plot_g_radius * rm.cos(gid * self._g_angle_interval)
                    self._graph.add_node(local_nodes[-1],
                                         obj_pose=obj_pose,
                                         grasp=grasp,
                                         jnt_values=jnt_values,
                                         plot_xy=(plot_grasp_x, plot_grasp_y))
                    new_global_nodes_by_gid[gid].append(local_nodes[-1])
                    self._global_nodes_by_gid[gid].append(local_nodes[-1])
                for node_pair in itertools.combinations(local_nodes, 2):
                    self._graph.add_edge(node_pair[0], node_pair[1], type='transit')
        for i in range(len(self.reference_grasp_collection)):
            new_global_nodes = new_global_nodes_by_gid[i]
            original_global_nodes = self._global_nodes_by_gid[i]
            for node_pair in itertools.product(new_global_nodes, original_global_nodes):
                self._graph.add_edge(node_pair[0], node_pair[1], type='transfer')
        # for global_nodes in tqdm(self._global_nodes_by_gid):
        #     for global_node_pair in itertools.combinations(global_nodes, 2):
        #         self.fsreg_graph.add_edge(global_node_pair[0], global_node_pair[1], type='transfer')

    def _add_fsregspot_to_graph(self, fsregspot):
        """
        add a spotfspgs to the regrasp graph
        :param fsregspot:
        :return:
        """
        new_global_nodes_by_gid = [[] for _ in range(len(self.reference_grasp_collection))]
        spot_x = fsregspot.pos[0]
        spot_y = fsregspot.pos[1]
        for fspg in fsregspot.fspgs:
            plot_pose_x = spot_x + self._plot_p_radius * rm.sin(fspg.fsp_pose_id * self._p_angle_interval)
            plot_pose_y = spot_y + self._plot_p_radius * rm.cos(fspg.fsp_pose_id * self._p_angle_interval)
            local_nodes = []
            obj_pose = fspg.obj_pose
            for gid, grasp, jnt_values in zip(fspg.feasible_gids, fspg.feasible_grasps, fspg.feasible_jv_list):
                local_nodes.append(uuid.uuid4())
                plot_grasp_x = plot_pose_x + self._plot_g_radius * rm.sin(gid * self._g_angle_interval)
                plot_grasp_y = plot_pose_y + self._plot_g_radius * rm.cos(gid * self._g_angle_interval)
                self._graph.add_node(local_nodes[-1],
                                     obj_pose=obj_pose,
                                     grasp=grasp,
                                     jnt_values=jnt_values,
                                     plot_xy=(plot_grasp_x, plot_grasp_y))
                new_global_nodes_by_gid[gid].append(local_nodes[-1])
                self._global_nodes_by_gid[gid].append(local_nodes[-1])
            for node_pair in itertools.combinations(local_nodes, 2):
                self._graph.add_edge(node_pair[0], node_pair[1], type='transit')
        for i in range(len(self.reference_grasp_collection)):
            new_global_nodes = new_global_nodes_by_gid[i]
            original_global_nodes = self._global_nodes_by_gid[i]
            for node_pair in itertools.product(new_global_nodes, original_global_nodes):
                self._graph.add_edge(node_pair[0], node_pair[1], type='transfer')

    def _add_fspg_to_graph(self, fspg, plot_pose_xy, prefix=''):
        """
        add a fspg to the regrasp graph
        :param fspg:
        :param plot_pose_xy: specify where to plot the fspg
        :return:
        """
        new_global_nodes_by_gid = [[] for _ in range(len(self.reference_grasp_collection))]
        local_nodes = []
        obj_pose = fspg.obj_pose
        for gid, grasp, jnt_values in zip(fspg.feasible_gids, fspg.feasible_grasps, fspg.feasible_jv_list):
            local_nodes.append(uuid.uuid4())
            plot_grasp_x = plot_pose_xy[0] + self._plot_g_radius * rm.sin(gid * self._g_angle_interval)
            plot_grasp_y = plot_pose_xy[1] + self._plot_g_radius * rm.cos(gid * self._g_angle_interval)
            self._graph.add_node(local_nodes[-1],
                                 obj_pose=obj_pose,
                                 grasp=grasp,
                                 jnt_values=jnt_values,
                                 plot_xy=(plot_grasp_x, plot_grasp_y))
            new_global_nodes_by_gid[gid].append(local_nodes[-1])
            self._global_nodes_by_gid[gid].append(local_nodes[-1])
        for node_pair in itertools.combinations(local_nodes, 2):
            self._graph.add_edge(node_pair[0], node_pair[1], type=prefix + '_transit')
        for i in range(len(self.reference_grasp_collection)):
            new_global_nodes = new_global_nodes_by_gid[i]
            original_global_nodes = self._global_nodes_by_gid[i]
            for node_pair in itertools.product(new_global_nodes, original_global_nodes):
                self._graph.add_edge(node_pair[0], node_pair[1], type=prefix + '_transfer')
        return local_nodes

    def draw_graph(self):
        # for regspot in self.regspot_col:
        #     spot_x = regspot.spot_pos[0]
        #     spot_y = regspot.spot_pos[1]
        #     plt.plot(spot_x, spot_y, 'ko')
        for node_tuple in self._graph.edges:
            node1_plot_xy = self._graph.nodes[node_tuple[0]]['plot_xy']
            node2_plot_xy = self._graph.nodes[node_tuple[1]]['plot_xy']
            if self._graph.edges[node_tuple]['type'] == 'transit':
                plt.plot([node1_plot_xy[0], node2_plot_xy[0]], [node1_plot_xy[1], node2_plot_xy[1]], 'c-')
            elif self._graph.edges[node_tuple]['type'] == 'transfer':
                plt.plot([node1_plot_xy[0], node2_plot_xy[0]], [node1_plot_xy[1], node2_plot_xy[1]], 'k-')
            elif self._graph.edges[node_tuple]['type'] == 'start_transit':
                plt.plot([node1_plot_xy[0], node2_plot_xy[0]], [node1_plot_xy[1], node2_plot_xy[1]], 'm-')
            elif self._graph.edges[node_tuple]['type'] == 'start_transfer':
                plt.plot([node1_plot_xy[0], node2_plot_xy[0]], [node1_plot_xy[1], node2_plot_xy[1]], 'g-')
            elif self._graph.edges[node_tuple]['type'] == 'goal_transit':
                plt.plot([node1_plot_xy[0], node2_plot_xy[0]], [node1_plot_xy[1], node2_plot_xy[1]], 'y-')
            elif self._graph.edges[node_tuple]['type'] == 'goal_transfer':
                plt.plot([node1_plot_xy[0], node2_plot_xy[0]], [node1_plot_xy[1], node2_plot_xy[1]], 'b-')
        plt.gca().set_aspect('equal', adjustable='box')

    def draw_path(self, path):
        n_nodes_on_path = len(path)
        for i in range(1, n_nodes_on_path):
            node1_plot_xy = self._graph.nodes[path[i]]['plot_xy']
            node2_plot_xy = self._graph.nodes[path[i - 1]]['plot_xy']
            plt.plot([node1_plot_xy[0], node2_plot_xy[0]], [node1_plot_xy[1], node2_plot_xy[1]], 'r-', linewidth=2)

    def show_graph(self):
        self.draw_graph()
        plt.show()

    def show_graph_with_path(self, path):
        self.draw_graph()
        self.draw_path(path)
        plt.show()

    def plan(self, start_fsp_pose_id, goal_fsp_pos_id):
        """

        :param start_fsp_pose_id:
        :param goal_fsp_pos_id:
        :return:
        """
        pass

    def gen_regrasp_motion(self, path, obstacle_list,
                           linear_distance=.05,
                           granularity=.03,
                           toggle_dbg=False):
        """
        """
        mesh_list = []
        for i, node in enumerate(path):
            obj_pose = self._graph.nodes[node]['obj_pose']
            grasp = self._graph.nodes[node]['grasp']
            jnt_values = self._graph.nodes[node]['jnt_values']
            m_col = mmc.ModelCollection()
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
                                                                  linear_distance=.1,
                                                                  ee_values = self.robot.end_effector.jaw_range[1],
                                                                  obstacle_list=obstacle_list,
                                                                  object_list=[obj_cmodel_copy],
                                                                  use_rrt=True)
                mesh_list += pick.mesh_list
            if i >= 1:
                prev_node = path[i - 1]
                prev_obj_pose = self._graph.nodes[prev_node]['obj_pose']
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
                                                                                       use_rrt=True)
                    if prev2current is None:
                        pass
                    for robot_mesh in prev2current.mesh_list:
                        obj_cmodel_copy.attach_to(robot_mesh)
                    mesh_list += prev2current.mesh_list
                if self._graph.edges[(prev_node, node)]['type'].endswith('transfer'):
                    self.robot.hold(obj_cmodel=obj_cmodel_copy, jaw_width=prev_grasp.ee_values)
                    prev2current = self.pp_planner.gen_depart_approach_with_given_conf(start_jnt_values=prev_jnt_values,
                                                                                       end_jnt_values=jnt_values,
                                                                                       depart_direction=rm.const.z_ax,
                                                                                       depart_distance=linear_distance,
                                                                                       approach_direction=-rm.const.z_ax,
                                                                                       approach_distance=linear_distance,
                                                                                       granularity=granularity,
                                                                                       obstacle_list=obstacle_list,
                                                                                       use_rrt=True)
                    # prev2current = self.pp_planner.im_planner.gen_interplated_between_given_conf(
                    #     start_jnt_values=prev_jnt_values,
                    #     end_jnt_values=jnt_values,
                    #     obstacle_list=[])
                    if toggle_dbg:
                        self.robot.gen_meshmodel(rgb=rm.const.red).attach_to(base)
                    self.robot.release(obj_cmodel=obj_cmodel_copy, jaw_width=self.robot.end_effector.jaw_range[1])
                    mesh_list += prev2current.mesh_list
        return mesh_list
