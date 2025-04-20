import numpy as np
import pickle
import wrs.basis.robot_math as rm
import wrs.basis.trimesh as trm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc
import wrs.modeling._ode_cdhelper as moh
import wrs.manipulation.placement.common as mp_pg
import wrs.grasping.planning.segmentation as seg
import wrs.grasping.reasoner as gr


class FSReferencePoses(object):

    def __init__(self, obj_cmodel=None, poses=None, stability_threshhold=.1, boundary_radius=.0025,
                 gravity_arrow_radius=.0025):
        self.obj_cmodel = obj_cmodel
        if obj_cmodel is not None and poses is None:
            self._poses, self._support_surfaces, self._stability_values = self.compute_reference_poses(
                obj_cmodel, stability_threshhold=stability_threshhold, boundary_radius=boundary_radius,
                gravity_arrow_radius=gravity_arrow_radius,
                toggle_support_facets=True)
        else:
            self._poses = poses
            self._support_surfaces = None
            self._stability_values = None

    @staticmethod
    def load_from_disk(file_name="fs_reference_poses.pickle"):
        with open(file_name, 'rb') as file:
            poses = pickle.load(file)
            return FSReferencePoses(poses=poses)

    def save_to_disk(self, file_name="fs_reference_poses.pickle"):
        with open(file_name, 'wb') as file:
            pickle.dump(self._poses, file)

    @staticmethod
    def compute_reference_poses(obj_cmodel, stability_threshhold=.1, boundary_radius=.0025, gravity_arrow_radius=.0025,
                                toggle_support_facets=False):
        """
        find all placements on a flat surface (z axis is the surface normal; no consideration on symmetry)
        the result is called a reference flat surface placement (reference fsp)
        :param obj_cmodel:
        :param stability_threshhold: the ratio of (com_projection to support boundary)/(com to com_projection)
        :return:
        author: weiwei
        date: 20161213, 20240321osaka
        """
        convex_trm = obj_cmodel.trm_mesh.convex_hull
        seg_result = seg.overlapped_segmentation(model=convex_trm, max_normal_bias_angle=np.pi / 64)
        seg_nested_face_id_list, seg_nested_edge_list, seg_seed_face_id_list, seg_normal_list, _ = seg_result
        pose_list = []
        support_facet_list = []
        stability_value_list = []
        for id, seg_face_id in enumerate(seg_seed_face_id_list):
            seed_face_normal = convex_trm.face_normals[seg_face_id]
            seed_face_z = -seed_face_normal
            seed_face_y = rm.orthogonal_vector(seed_face_z)
            seed_face_x = np.cross(seed_face_y, seed_face_z)
            seed_face_rotmat = np.column_stack((seed_face_x, seed_face_y, seed_face_z))
            seed_face_pos = np.mean(convex_trm.vertices[convex_trm.faces[seg_face_id]], axis=0)
            placement_pos, placement_rotmat = rm.rel_pose(seed_face_pos,
                                                          seed_face_rotmat,
                                                          np.zeros(3),
                                                          np.eye(3))
            normals = seg_normal_list[id]
            faces = convex_trm.faces[seg_nested_face_id_list[id]]
            facet = mcm.CollisionModel(
                initor=trm.Trimesh(vertices=convex_trm.vertices, faces=faces, face_normals=normals),
                toggle_twosided=True, rgb=rm.const.tab20_list[0], alpha=.5)
            # show edge
            for edge in seg_nested_edge_list[id]:
                mgm.gen_stick(spos=edge[0], epos=edge[1], type="round", radius=boundary_radius).attach_to(facet)
            com = obj_cmodel.trm_mesh.center_mass
            result = moh.rayhit_closet(spos=com, epos=com + seed_face_normal,
                                       target_cmodel=facet)
            if result is not None:
                contact_point, contact_normal = result
                min_contact_distance = np.linalg.norm(contact_point - com)
                min_edge_distance, min_edge_projection = rm.min_distance_point_edge_list(contact_point,
                                                                                         seg_nested_edge_list[id])
                stability_value = min_edge_distance / min_contact_distance
                if stability_value < stability_threshhold:
                    continue
                # show contact point to edge projection
                mgm.gen_stick(spos=contact_point, epos=min_edge_projection, radius=boundary_radius,
                              type="round").attach_to(facet)
                pose_list.append((placement_pos, placement_rotmat))
                mgm.gen_arrow(spos=com, epos=contact_point, stick_radius=gravity_arrow_radius).attach_to(facet)
                support_facet_list.append(facet)
                stability_value_list.append(stability_value)
        if toggle_support_facets:
            combined_lists = list(zip(pose_list, support_facet_list, stability_value_list))
            sorted_combined_lists = sorted(combined_lists, key=lambda x: x[2], reverse=True)
            pose_list_sorted, support_facet_list_sorted, stability_value_list_sorted = zip(*sorted_combined_lists)
            return pose_list_sorted, support_facet_list_sorted, stability_value_list_sorted
        return pose_list

    @property
    def support_surfaces(self):
        return self._support_surfaces

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._poses[index]
        elif isinstance(index, list):
            return [self._poses[i] for i in index]
        else:
            raise Exception("Index type not supported.")

    def __len__(self):
        return len(self._poses)

    def __iter__(self):
        return iter(self._poses)


class FSPG(mp_pg.GPG):
    """
    A container class that holds the placement pose id, and associated feasible grasps id
    """

    def __init__(self,
                 fs_pose_id=None,
                 obj_pose=None,
                 feasible_gids=None,
                 feasible_grasps=None,
                 feasible_confs=None):
        super().__init__(obj_pose=obj_pose,
                         feasible_gids=feasible_gids,
                         feasible_grasps=feasible_grasps,
                         feasible_confs=feasible_confs)
        self._fs_pose_id = fs_pose_id

    @property
    def fs_pose_id(self):
        return self._fs_pose_id

    def __str__(self):
        return f"pose id = {repr(self._fs_pose_id)}, with gids= {repr(self._feasible_gids)}"


class FSRegSpot(object):
    def __init__(self, pos=None, rotz=None):
        self.pos = pos
        self.rotz = rotz
        self.fspg_list = []

    def __iter__(self):
        return iter(self.fspg_list)


class FSRegSpotCollection(object):
    def __init__(self, robot, obj_cmodel, fs_reference_poses, reference_gc):
        """
        :param robot:
        :param reference_fsp_poses: an instance of ReferenceFSPPoses
        :param reference_gc: an instance of GraspCollection
        """
        self.robot = robot
        self.obj_cmodel = obj_cmodel
        self.fs_reference_poses = fs_reference_poses
        self._fsregspot_list = []  # list of FSRegSpot
        self.grasp_reasoner = gr.GraspReasoner(robot=robot, reference_gc=reference_gc)

    @property
    def reference_gc(self):
        return self.grasp_reasoner.reference_gc

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
        fs_regspot = FSRegSpot(spot_pos, spot_rotz)
        if barrier_z_offset is not None:
            obstacle_list = [mcm.gen_surface_barrier(spot_pos[2] + barrier_z_offset)]
        else:
            obstacle_list = []
        for pose_id, pose in enumerate(self.fs_reference_poses):
            pos = pose[0] + spot_pos
            rotmat = rm.rotmat_from_euler(0, 0, spot_rotz) @ pose[1]
            mgm.gen_frame(pos=pos, rotmat=rotmat).attach_to(base)
            feasible_gids, feasible_grasps, feasible_confs = self.grasp_reasoner.find_feasible_gids(
                goal_pose=(pos, rotmat),
                obstacle_list=obstacle_list,
                consider_robot=consider_robot,
                toggle_dbg=False)
            if feasible_gids is not None:
                fs_regspot.fspg_list.append(FSPG(fs_pose_id=pose_id,
                                                 obj_pose=(pos, rotmat),
                                                 feasible_gids=feasible_gids,
                                                 feasible_grasps=feasible_grasps,
                                                 feasible_confs=feasible_confs))
            if toggle_dbg:
                for grasp, jnt_values in zip(feasible_grasps, feasible_confs):
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
                feasible_confs = fspg.feasible_confs
                obj_cmodel_copy = self.obj_cmodel.copy()
                obj_cmodel_copy.pose = obj_pose
                obj_cmodel_copy.attach_to(m_col)
                for grasp, conf in zip(feasible_grasps, feasible_confs):
                    self.robot.goto_given_conf(jnt_values=conf, ee_values=grasp.ee_values)
                    self.robot.gen_meshmodel().attach_to(m_col)
                meshmodel_list.append(m_col)
        return meshmodel_list


if __name__ == '__main__':
    import os
    import time
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    obj_path = os.path.join(os.path.dirname(rm.__file__), 'objects', 'bunnysim.stl')
    ground = mcm.gen_box(xyz_lengths=[.5, .5, .01], pos=np.array([0, 0, -0.01]))
    ground.attach_to(base)
    bunny = mcm.CollisionModel(obj_path)

    fs_reference_poses = FSReferencePoses(obj_cmodel=bunny)


    class AnimeData(object):
        def __init__(self, poses):
            self.counter = 0
            self.model = fs_reference_poses.obj_cmodel
            self.poses = fs_reference_poses
            self.support_facets = fs_reference_poses.support_surfaces


    anime_data = AnimeData(poses=fs_reference_poses)


    def update(anime_data, task):
        if anime_data.counter >= len(anime_data.poses):
            anime_data.model.detach()
            anime_data.support_facets[anime_data.counter - 1].detach()
            anime_data.counter = 0
        if base.inputmgr.keymap["space"] is True:
            time.sleep(.1)
            anime_data.model.detach()
            print(anime_data.poses[anime_data.counter])
            anime_data.model.pose = anime_data.poses[anime_data.counter]
            anime_data.model.rgb = rm.const.tab20_list[1]
            anime_data.model.alpha = .3
            anime_data.model.attach_to(base)
            if (anime_data.support_facets is not None):
                if anime_data.counter > 0:
                    anime_data.support_facets[anime_data.counter - 1].detach()
                anime_data.support_facets[anime_data.counter].pose = anime_data.poses[anime_data.counter]
                anime_data.support_facets[anime_data.counter].attach_to(base)
            anime_data.counter += 1
        return task.cont


    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[anime_data],
                          appendTask=True)
    base.run()
