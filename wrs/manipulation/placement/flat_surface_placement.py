import numpy as np
import pickle
import wrs.basis.robot_math as rm
import wrs.basis.trimesh as trm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.modeling._ode_cdhelper as moh
import wrs.grasping.planning.segmentation as seg
import wrs.manipulation.placement.general_placement as mpgp


class FSReferencePoses(object):

    def __init__(self, obj_cmodel=None, poses=None):
        self.obj_cmodel = obj_cmodel
        if obj_cmodel is not None and poses is None:
            self._poses, self._support_surfaces, self._stability_values = self.comptue_reference_poses(
                obj_cmodel, toggle_support_facets=True)
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
    def comptue_reference_poses(obj_cmodel, stability_threshhold=.1, toggle_support_facets=False):
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
                mgm.gen_stick(spos=edge[0], epos=edge[1], type="round").attach_to(facet)
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
                mgm.gen_stick(spos=contact_point, epos=min_edge_projection, type="round").attach_to(facet)
                pose_list.append((placement_pos, placement_rotmat))
                mgm.gen_arrow(spos=com, epos=contact_point).attach_to(facet)
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


class FSPG(mpgp.PG):
    """
    A container class that holds the placement pose id, and associated feasible grasps id
    """

    def __init__(self,
                 fs_pose_id=None,
                 obj_pose=None,
                 feasible_gids=None,
                 feasible_grasps=None,
                 feasible_jv_list=None):
        super().__init__(obj_pose=obj_pose,
                         feasible_gids=feasible_gids,
                         feasible_grasps=feasible_grasps,
                         feasible_jv_list=feasible_jv_list)
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

    def add_fspg(self, fspg):
        self.fspg_list.append(fspg)

    def __iter__(self):
        return iter(self.fspg_list)


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
