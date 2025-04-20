import uuid, itertools, networkx
import matplotlib.pyplot as plt
import wrs.basis.robot_math as rm
import wrs.motion.motion_data as motd
import wrs.manipulation.utils as mp_utils
import wrs.manipulation.pick_place as ppp
import wrs.manipulation.placement.common as mp_gp
import wrs.manipulation.placement.flatsurface as mp_fsp


class FSRegraspPlanner(object):

    def __init__(self, robot, obj_cmodel, fs_reference_poses, reference_gc):
        self._graph = networkx.Graph()
        self._fsregspot_collection = mp_fsp.FSRegSpotCollection(robot=robot,
                                                                obj_cmodel=obj_cmodel,
                                                                fs_reference_poses=fs_reference_poses,
                                                                reference_gc=reference_gc)
        self.pp_planner = ppp.PickPlacePlanner(robot)
        self._gl_nodes_by_gid = {}
        for id in range(len(reference_gc)):
            self._gl_nodes_by_gid[id] = []
        self._plot_g_radius = .01
        self._plot_p_radius = 0.05
        self._n_fs_reference_poses = len(fs_reference_poses) if fs_reference_poses is not None else 1
        self._n_reference_grasps = len(reference_gc)
        self._p_angle_interval = rm.pi * 2 / self._n_fs_reference_poses
        self._g_angle_interval = rm.pi * 2 / self._n_reference_grasps

    @property
    def robot(self):
        return self._fsregspot_collection.robot

    @property
    def obj_cmodel(self):
        return self._fsregspot_collection.obj_cmodel

    @property
    def graph(self):
        return self._graph

    @property
    def fs_reference_poses(self):
        return self._fsregspot_collection.fs_reference_poses

    @property
    def reference_gc(self):
        return self._fsregspot_collection.reference_gc

    @property
    def gl_nodes_by_gid(self):
        return self._gl_nodes_by_gid

    def add_fsregspot_collection_from_disk(self, file_name):
        fsregspot_collection = mp_fsp.FSRegSpotCollection(robot=self.robot,
                                                          obj_cmodel=self.obj_cmodel,
                                                          fs_reference_poses=self.fs_reference_poses,
                                                          reference_gc=self.reference_gc)
        fsregspot_collection.load_from_disk(file_name)
        self._fsregspot_collection += fsregspot_collection
        self._add_fsregspot_collection_to_graph(fsregspot_collection)

    def save_to_disk(self, file_name):
        pass

    def load_from_disk(self, file_name):
        pass

    def create_add_fsregspot(self, spot_pos, spot_rotz, barrier_z_offset=-.01, consider_robot=True, toggle_dbg=False):
        self._fsregspot_collection.add_new_spot(spot_pos, spot_rotz, barrier_z_offset, consider_robot, toggle_dbg)
        self._add_fsregspot_to_graph(self._fsregspot_collection[-1])

    def add_start_pose(self, obj_pose, obstacle_list=None, plot_pose_xy=None, toggle_dbg=False):
        start_pg = mp_gp.GPG.create_from_pose(self.robot,
                                              self.reference_gc,
                                              obj_pose,
                                              obstacle_list=obstacle_list,
                                              consider_robot=True, toggle_dbg=toggle_dbg)
        if start_pg is None:
            print("No feasible grasps found at the start pose")
            return None
        if plot_pose_xy is None:
            plot_pose_xy = [obj_pose[0][1], -obj_pose[0][0]]
        return self._add_fspg_to_graph(start_pg, plot_pose_xy=plot_pose_xy, prefix='start')

    def add_goal_pose(self, obj_pose, obstacle_list=None, plot_pose_xy=None, toggle_dbg=False):
        goal_pg = mp_gp.GPG.create_from_pose(self.robot,
                                             self.reference_gc,
                                             obj_pose,
                                             obstacle_list=obstacle_list,
                                             consider_robot=True, toggle_dbg=toggle_dbg)
        if goal_pg is None:
            print("No feasible grasps found at the goal pose")
            return None
        if plot_pose_xy is None:
            plot_pose_xy = [obj_pose[0][1], -obj_pose[0][0]]
        return self._add_fspg_to_graph(goal_pg, plot_pose_xy=plot_pose_xy, prefix='goal')

    def _add_fsregspot_collection_to_graph(self, fsregspot_collection):
        """
        add regspot collection to the regrasp graph
        :param fsregspot_collection: an instance of FSRegSpotCollection or a list of FSRegSpot
        :return:
        """
        new_gl_nodes_by_gid = {}
        for id in range(len(self.reference_gc)):
            new_gl_nodes_by_gid[id] = []
        for fsregspot in fsregspot_collection:
            spot_x = fsregspot.pos[1]
            spot_y = -fsregspot.pos[0]
            for fspg in fsregspot.fspg_list:
                plot_pose_x = spot_x + self._plot_p_radius * rm.sin(fspg.fs_pose_id * self._p_angle_interval)
                plot_pose_y = spot_y + self._plot_p_radius * rm.cos(fspg.fs_pose_id * self._p_angle_interval)
                local_nodes = []
                obj_pose = fspg.obj_pose
                for gid, grasp, jnt_values in zip(fspg.feasible_gids, fspg.feasible_grasps, fspg.feasible_confs):
                    local_nodes.append(uuid.uuid4())
                    plot_grasp_x = plot_pose_x + self._plot_g_radius * rm.sin(gid * self._g_angle_interval)
                    plot_grasp_y = plot_pose_y + self._plot_g_radius * rm.cos(gid * self._g_angle_interval)
                    self._graph.add_node(local_nodes[-1],
                                         obj_pose=obj_pose,
                                         grasp=grasp,
                                         jnt_values=jnt_values,
                                         plot_xy=(plot_grasp_x, plot_grasp_y))
                    new_gl_nodes_by_gid[gid].append(local_nodes[-1])
                    self._gl_nodes_by_gid[gid].append(local_nodes[-1])
                for node_pair in itertools.combinations(local_nodes, 2):
                    self._graph.add_edge(node_pair[0], node_pair[1], type='transit')
        for i in range(len(self.reference_gc)):
            for node_pair in itertools.product(new_gl_nodes_by_gid[i], self._gl_nodes_by_gid[i]):
                if node_pair[0] != node_pair[1]:
                    self._graph.add_edge(node_pair[0], node_pair[1], type='transfer')
        # for global_nodes in tqdm(self._gl_nodes_by_gid):
        #     for global_node_pair in itertools.combinations(global_nodes, 2):
        #         self.fsreg_graph.add_edge(global_node_pair[0], global_node_pair[1], type='transfer')

    def _add_fsregspot_to_graph(self, fsregspot):
        """
        add a spotfspgs to the regrasp graph
        :param fsregspot:
        :return:
        """
        new_gl_nodes_by_gid = {}
        for id in range(len(self.reference_gc)):
            new_gl_nodes_by_gid[id] = []
        spot_x = fsregspot.pos[1]
        spot_y = -fsregspot.pos[0]
        for fspg in fsregspot.fspg_list:
            plot_pose_x = spot_x + self._plot_p_radius * rm.sin(fspg.fsp_pose_id * self._p_angle_interval)
            plot_pose_y = spot_y + self._plot_p_radius * rm.cos(fspg.fsp_pose_id * self._p_angle_interval)
            local_nodes = []
            obj_pose = fspg.obj_pose
            for gid, grasp, jnt_values in zip(fspg.feasible_gids, fspg.feasible_grasps, fspg.feasible_confs):
                local_nodes.append(uuid.uuid4())
                plot_grasp_x = plot_pose_x + self._plot_g_radius * rm.sin(gid * self._g_angle_interval)
                plot_grasp_y = plot_pose_y + self._plot_g_radius * rm.cos(gid * self._g_angle_interval)
                self._graph.add_node(local_nodes[-1],
                                     obj_pose=obj_pose,
                                     grasp=grasp,
                                     jnt_values=jnt_values,
                                     plot_xy=(plot_grasp_x, plot_grasp_y))
                new_gl_nodes_by_gid[gid].append(local_nodes[-1])
                self._gl_nodes_by_gid[gid].append(local_nodes[-1])
            for node_pair in itertools.combinations(local_nodes, 2):
                self._graph.add_edge(node_pair[0], node_pair[1], type='transit')
        for i in range(len(self.reference_gc)):
            for node_pair in itertools.product(new_gl_nodes_by_gid[i], self._gl_nodes_by_gid[i]):
                if node_pair[0] != node_pair[1]:
                    self._graph.add_edge(node_pair[0], node_pair[1], type='transfer')

    def _add_fspg_to_graph(self, fspg, plot_pose_xy, prefix=''):
        """
        add a fspg to the regrasp graph
        :param fspg:
        :param plot_pose_xy: specify where to plot the fspg
        :return:
        """
        new_gl_nodes_by_gid = {}
        for id in range(len(self.reference_gc)):
            new_gl_nodes_by_gid[id] = []
        local_nodes = []
        obj_pose = fspg.obj_pose
        for gid, grasp, jnt_values in zip(fspg.feasible_gids, fspg.feasible_grasps, fspg.feasible_confs):
            local_nodes.append(uuid.uuid4())
            plot_grasp_x = plot_pose_xy[0] + self._plot_g_radius * rm.sin(gid * self._g_angle_interval)
            plot_grasp_y = -plot_pose_xy[1] + self._plot_g_radius * rm.cos(gid * self._g_angle_interval)
            self._graph.add_node(local_nodes[-1],
                                 obj_pose=obj_pose,
                                 grasp=grasp,
                                 jnt_values=jnt_values,
                                 plot_xy=(plot_grasp_x, plot_grasp_y))
            new_gl_nodes_by_gid[gid].append(local_nodes[-1])
            # self._gl_nodes_by_gid[gid].append(local_nodes[-1])
        # for node_pair in itertools.combinations(local_nodes, 2):
        #     self._graph.add_edge(node_pair[0], node_pair[1], type=prefix + '_transit')
        for i in range(len(self.reference_gc)):
            for node_pair in itertools.product(new_gl_nodes_by_gid[i], self._gl_nodes_by_gid[i]):
                self._graph.add_edge(node_pair[0], node_pair[1], type=prefix + '_transfer')
            self._gl_nodes_by_gid[i] += new_gl_nodes_by_gid[i]
        return local_nodes

    def plan_by_obj_poses(self, start_pose, goal_pose, obstacle_list=None, linear_distance=.07, toggle_dbg=False):
        """
        :param start_pose: (pos, rotmat)
        :param goal_pose: (pos, rotmat)
        :return:
        """
        start_node_list = self.add_start_pose(obj_pose=start_pose, obstacle_list=obstacle_list)
        goal_node_list = self.add_goal_pose(obj_pose=goal_pose, obstacle_list=obstacle_list)
        self.show_graph()
        while True:
            min_path = None
            for start in start_node_list:
                for goal in goal_node_list:
                    try:
                        path = networkx.shortest_path(self._graph, source=start, target=goal)
                        min_path = path if min_path is None else path if len(path) < len(min_path) else min_path
                    except networkx.NetworkXNoPath:
                        print(f"No path exists between {start} and {goal}")
                        continue
            result = self.gen_regrasp_motion(path=min_path, obstacle_list=obstacle_list,
                                             start_jnt_values=self.robot.get_jnt_values(),
                                             linear_distance=linear_distance,
                                             toggle_dbg=toggle_dbg)
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
            # self.show_graph()

    @ppp.adp.mpi.InterplatedMotion.keep_states_decorator
    def gen_regrasp_motion(self, path, obstacle_list, start_jnt_values=None, linear_distance=.03,
                           granularity=.03, toggle_start_approach=True, toggle_end_depart=True, toggle_dbg=False):
        """
        :param path:
        :param obstacle_list:
        :param start_jnt_values:
        :param linear_distance:
        :param granularity:
        :param rtoggle_start_approach:
        :param toggle_end_depart:
        :param toggle_dbg:
        :return:
        """
        regraps_motion = motd.MotionData(robot=self.robot)
        for i, node in enumerate(path):
            obj_pose = self._graph.nodes[node]['obj_pose']
            curr_jnt_values = self._graph.nodes[node]['jnt_values']
            # make a copy to keep original movement
            obj_cmodel_copy = self.obj_cmodel.copy()
            obj_cmodel_copy.pose = obj_pose
            if toggle_dbg:
                self.robot.gen_meshmodel().attach_to(base)
            if i == 0 and toggle_start_approach:
                approach = self.pp_planner.gen_approach_to_given_conf(goal_jnt_values=curr_jnt_values,
                                                                      start_jnt_values=start_jnt_values,
                                                                      linear_distance=linear_distance,
                                                                      ee_values=self.robot.end_effector.jaw_range[1],
                                                                      obstacle_list=obstacle_list,
                                                                      object_list=[obj_cmodel_copy],
                                                                      use_rrt=True,
                                                                      toggle_dbg=toggle_dbg)
                if approach is None:
                    return (f"node failure at {i}", path[i])
                for robot_mesh in approach.mesh_list:
                    obj_cmodel_copy.attach_to(robot_mesh)
                regraps_motion += approach
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
                                                                                       end_jnt_values=curr_jnt_values,
                                                                                       depart_direction=None,
                                                                                       depart_distance=linear_distance,
                                                                                       depart_ee_values=
                                                                                       self.robot.end_effector.jaw_range[
                                                                                           1],
                                                                                       approach_direction=None,
                                                                                       approach_distance=linear_distance,
                                                                                       approach_ee_values=
                                                                                       self.robot.end_effector.jaw_range[
                                                                                           1],
                                                                                       linear_granularity=granularity,
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
                    obj_cmodel_copy.pose = self._graph.nodes[prev_node]['obj_pose']
                    self.robot.goto_given_conf(prev_jnt_values)
                    self.robot.hold(obj_cmodel=obj_cmodel_copy, jaw_width=prev_grasp.ee_values)
                    # regraps_motion.extend(jv_list=[prev_jnt_values],
                    #                       ev_list=[prev_grasp.ee_values],
                    #                       mesh_list=[self.robot.gen_meshmodel()])
                    prev2current = self.pp_planner.gen_depart_approach_with_given_conf(start_jnt_values=prev_jnt_values,
                                                                                       end_jnt_values=curr_jnt_values,
                                                                                       depart_direction=rm.const.z_ax,
                                                                                       depart_distance=linear_distance,
                                                                                       approach_direction=-rm.const.z_ax,
                                                                                       approach_distance=linear_distance,
                                                                                       linear_granularity=granularity,
                                                                                       obstacle_list=obstacle_list,
                                                                                       use_rrt=True,
                                                                                       toggle_dbg=False)
                    if prev2current is None:
                        return (f"edge failure at transfer {i - 1}-{i}", (path[i - 1], path[i]))
                    regraps_motion += prev2current
                    self.robot.goto_given_conf(prev2current.jv_list[-1])
                    if toggle_dbg:
                        self.robot.gen_meshmodel(rgb=rm.const.red).attach_to(base)
                    self.robot.release(obj_cmodel=obj_cmodel_copy, jaw_width=self.robot.end_effector.jaw_range[1])
                    # it seems we do not need to explicitly add release models
                    # mesh = self.robot.gen_meshmodel()
                    # obj_cmodel_copy.attach_to(mesh)
                    # regraps_motion.extend(jv_list=[prev2current.jv_list[-1]],
                    #                       ev_list=[self.robot.end_effector.jaw_range[1]],
                    #                       mesh_list=[mesh])
            if i == len(path) - 1 and toggle_end_depart:
                obj_cmodel_copy.pose = obj_pose
                retract = self.pp_planner.gen_depart_from_given_conf(start_jnt_values=curr_jnt_values,
                                                                     end_jnt_values=start_jnt_values,
                                                                     linear_distance=linear_distance,
                                                                     ee_values=self.robot.end_effector.jaw_range[1],
                                                                     obstacle_list=obstacle_list,
                                                                     object_list=[obj_cmodel_copy],
                                                                     use_rrt=True,
                                                                     toggle_dbg=False)
                if retract is None:
                    return (f"node failure at {i}", path[i])
                for robot_mesh in retract.mesh_list:
                    obj_cmodel_copy.attach_to(robot_mesh)
                regraps_motion += retract
        return ("success", regraps_motion)

    def show_graph(self):
        mp_utils.show_graph(self._graph)

    def show_graph_with_path(self, path):
        mp_utils.show_graph_with_path(self._graph, path)
