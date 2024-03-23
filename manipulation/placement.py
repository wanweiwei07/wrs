import numpy as np
import pickle
import basis.robot_math as rm
import basis.trimesh.base as trm
import modeling.geometric_model as mgm
import modeling.collision_model as mcm
import grasping.planning.segmentation as seg
import modeling._ode_cdhelper as moh
import grasping.reasoner as gr


class FSPwGIdx(object):
    """
    A container class that holds the flat surface placement pose id and its associated feasible grasps id
    """

    def __init__(self, pos=None, rotz=None, fsp_pose_id=None, feasible_gids=None):
        self._tgt_pos = pos
        self._tgt_rotz = rotz
        self._placement_pose_id = fsp_pose_id
        self._feasible_gids = feasible_gids

    @property
    def tgt_pos(self):
        return self._tgt_pos

    @property
    def tgt_rotz(self):
        return self._tgt_rotz

    @property
    def placement_pose_id(self):
        return self._placement_pose_id

    @property
    def feasible_gids(self):
        return self._feasible_gids

    def __str__(self):
        out_str = f"Tabletop Placement at {repr(self._tgt_pos)}, with theta={repr(self._tgt_rotz)}\n"
        out_str += f"   Placement id = {repr(self._placement_pose_id)}, with gids= {repr(self._feasible_gids)}"
        return out_str


class FSPwGIdxCollection(object):
    def __init__(self, reference_fsp_pose_list, reference_grasp_collection):
        """I am learning users to ensure placements and grasps are defined for the same objecd"""
        self.reference_fsp_poses = reference_fsp_pose_list
        self.reference_grasp_collection = reference_grasp_collection
        self._fspwg_list = []

    @classmethod
    def load_from_disk(cls, file_name="fspg_collection.pickle"):
        with open(file_name, 'rb') as file:
            obj = pickle.load(file)
            if not isinstance(obj, cls):
                raise TypeError(f"Object in {file_name} is not an instance of {cls.__name__}")
            return obj

    def append(self, ttpg):
        self._fspwg_list.append(ttpg)

    def get_placement_pose_by_fspwg(self, fspwg):
        return self.reference_fsp_poses[fspwg._placement_pose_id]

    def get_grasp_list(self, fspwg):
        return self.reference_grasp_collection[fspwg._feasible_gids]

    def __getitem__(self, index):
        return self._fspwg_list[index]

    def __setitem__(self, index, fspg):
        self._fspwg_list[index] = fspg

    def __len__(self):
        return len(self._fspwg_list)

    def __iter__(self):
        return iter(self._fspwg_list)

    def __add__(self, other):
        """
        leave it to the user to ensure the two collections share the same reference placements and grasps
        :param other:
        :return:
        """
        self._fspwg_list += other._fspwg_list
        return self

    def __str__(self):
        out_str = f"FSPGCollection of {len(self._fspwg_list)} FSPGs\n"
        for fspg in self._fspwg_list:
            out_str += str(fspg) + "\n"
        return out_str

    def save_to_disk(self, file_name='fspg_collection.pickle'):
        """
        :param reference_grasp_collection:
        :param file_name
        :return:
        """
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)


def get_reference_fsp_pose_list(obj_cmodel, stability_threshhold=.1, toggle_support_facets=False):
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
    fsp_pose_list = []
    support_facet_list = []
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
        facet = mcm.CollisionModel(initor=trm.Trimesh(vertices=convex_trm.vertices, faces=faces, face_normals=normals),
                                   toggle_twosided=True, rgb=rm.bc.tab20_list[0], alpha=.5)
        # show edge
        for edge in seg_nested_edge_list[id]:
            mgm.gen_stick(spos=edge[0], epos=edge[1], type="round").attach_to(facet)
        com = obj_cmodel.trm_mesh.center_mass
        contact_point, contact_normal = moh.rayhit_closet(spos=com, epos=com + seed_face_normal, target_cmodel=facet)
        if contact_point is not None:
            min_contact_distance = np.linalg.norm(contact_point - com)
            min_edge_distance, min_edge_projection = rm.min_distance_point_edge_list(contact_point,
                                                                                     seg_nested_edge_list[id])
            if min_edge_distance / min_contact_distance < stability_threshhold:
                continue
            # show contact point to edge projection
            mgm.gen_stick(spos=contact_point, epos=min_edge_projection, type="round").attach_to(facet)
            fsp_pose_list.append((placement_pos, placement_rotmat))
            mgm.gen_arrow(spos=com, epos=contact_point).attach_to(facet)
            support_facet_list.append(facet)
    if toggle_support_facets:
        return fsp_pose_list, support_facet_list
    return fsp_pose_list


def fs_placements_with_grasps(robot,
                              reference_fsp_pose_list,
                              reference_grasp_collection,
                              tgt_xyz,
                              tgt_rotz=0,
                              barrier_z_offset=-.01,
                              consider_robot=True,
                              toggle_dbg=False):
    fspg_collection = FSPwGIdxCollection(reference_fsp_pose_list, reference_grasp_collection)
    grasp_reasoner = gr.GraspReasoner(robot=robot)
    barrier_obstacle = mcm.gen_surface_barrier(tgt_xyz[2]+barrier_z_offset)
    for pid, reference_fsp_pose in enumerate(reference_fsp_pose_list):
        pos = reference_fsp_pose[0] + tgt_xyz
        rotmat = reference_fsp_pose[1]
        feasible_gids = grasp_reasoner.find_feasible_gids(reference_grasp_collection=reference_grasp_collection,
                                                          obstacle_list=[barrier_obstacle],
                                                          goal_pose=(pos, rotmat),
                                                          consider_robot=consider_robot,
                                                          toggle_keep=True,
                                                          toggle_dbg=False)
        if feasible_gids is not None:
            fspg_collection.append(FSPwGIdx(pos=pos,
                                            rotz=tgt_rotz,
                                            fsp_pose_id=pid,
                                            feasible_gids=feasible_gids))
        if toggle_dbg:
            for f_gid in feasible_gids:
                print(f_gid)
                grasp = reference_grasp_collection[f_gid]
                jnt_values = robot.ik(tgt_pos=pos + rotmat @ grasp.ac_pos, tgt_rotmat=rotmat @ grasp.ac_rotmat)
                if jnt_values is not None:
                    robot.goto_given_conf(jnt_values=jnt_values, ee_values=grasp.ee_values)
                robot.gen_meshmodel().attach_to(base)
            base.run()
    return fspg_collection


if __name__ == '__main__':
    import os
    import time
    import basis
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    obj_path = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
    ground = mcm.gen_box(xyz_lengths=[.5, .5, .01], pos=np.array([0, 0, -0.01]))
    ground.attach_to(base)
    bunny = mcm.CollisionModel(obj_path)
    placement_pose_list, support_facets = flat_placements(bunny, toggle_support_facets=True)


    # for id, placement_pose in enumerate(reference_fsp_pose_list):
    #     bgm_copy = bunny_gmodel.copy()
    #     bgm_copy.pose = placement_pose
    #     bgm_copy.pos = bgm_copy.pos+np.array([id*.05, 0, 0])
    #     bgm_copy.attach_to(base)

    class AnimeData(object):
        def __init__(self, placement_pose_list, support_facets=None):
            self.counter = 0
            self.model = bunny
            self.placement_pose_list = placement_pose_list
            self.support_facets = support_facets


    anime_data = AnimeData(placement_pose_list, support_facets)

    print(len(anime_data.placement_pose_list))


    def update(anime_data, task):
        if anime_data.counter >= len(anime_data.placement_pose_list):
            anime_data.model.detach()
            anime_data.support_facets[anime_data.counter - 1].detach()
            anime_data.counter = 0
        if base.inputmgr.keymap["space"] is True:
            time.sleep(.1)
            anime_data.model.detach()
            print(anime_data.placement_pose_list[anime_data.counter])
            anime_data.model.pose = anime_data.placement_pose_list[anime_data.counter]
            anime_data.model.rgb = rm.bc.tab20_list[1]
            anime_data.model.alpha = .3
            anime_data.model.attach_to(base)
            if (anime_data.support_facets is not None):
                if anime_data.counter > 0:
                    anime_data.support_facets[anime_data.counter - 1].detach()
                anime_data.support_facets[anime_data.counter].pose = anime_data.placement_pose_list[
                    anime_data.counter]
                anime_data.support_facets[anime_data.counter].attach_to(base)
            anime_data.counter += 1
        return task.cont


    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[anime_data],
                          appendTask=True)
    base.run()
