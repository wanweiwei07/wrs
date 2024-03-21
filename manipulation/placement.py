import numpy as np
import basis.robot_math as rm
import basis.trimesh.base as trm
import modeling.geometric_model as mgm
import modeling.collision_model as mcm
import grasping.planning.segmentation as seg
import modeling._ode_cdhelper as moh


def tabletop_placements(obj_cmodel, toggle_support_facets=False):
    """
    find all placements on a table (z axis is the plane normal; no consideration on symmetry)
    :param obj_cmodel:
    :return:
    author: weiwei
    """
    convex_trm = obj_cmodel.trm_mesh.convex_hull
    seg_result = seg.overlapped_segmentation(model=convex_trm, max_normal_bias_angle=np.pi / 32)
    seg_nested_face_id_list, seg_nested_edge_list, seg_seed_face_id_list, seg_normal_list, _ = seg_result
    placement_pose_list = []
    support_facet_list = []
    for id, seg_face_id in enumerate(seg_seed_face_id_list):
        seed_face_normal = convex_trm.face_normals[seg_face_id]
        seed_face_z = -seed_face_normal
        seed_face_y = rm.orthogonal_vector(seed_face_z)
        seed_face_x = np.cross(seed_face_y, seed_face_z)
        seed_face_rotmat = np.column_stack((seed_face_x, seed_face_y, seed_face_z))
        seed_face_pos = np.mean(convex_trm.vertices[convex_trm.faces[seg_face_id]], axis=0)
        placement_pos, placement_rotmat = rm.rel_pose(seed_face_pos, seed_face_rotmat, obj_cmodel.pos,
                                                      obj_cmodel.rotmat)
        normals = seg_normal_list[id]
        faces = convex_trm.faces[seg_nested_face_id_list[id]]
        facet = mcm.CollisionModel(initor=trm.Trimesh(vertices=convex_trm.vertices, faces=faces, face_normals=normals),
                                   toggle_twosided=True, rgb=rm.bc.tab20_list[0], alpha=.5)
        com = obj_cmodel.trm_mesh.center_mass
        contact_point, contact_normal = moh.rayhit_closet(spos=com, epos=seed_face_pos, target_cmodel=facet)
        if contact_point is not None:
            placement_pose_list.append((placement_pos, placement_rotmat))
            mgm.gen_arrow(spos=com, epos=contact_point).attach_to(facet)
            support_facet_list.append(facet)
    if toggle_support_facets:
        return placement_pose_list, support_facet_list
    return placement_pose_list


if __name__ == '__main__':
    import os
    import time
    import basis
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    obj_path = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
    ground = mcm.gen_box(xyz_lengths=[.5, .5, .01], pos=np.array([0, 0, -0.01]), rgba=[.3, .3, .3, 1])
    ground.attach_to(base)
    bunny_gmodel = mcm.CollisionModel(obj_path)
    placement_pose_list, support_facets = tabletop_placements(bunny_gmodel, toggle_support_facets=True)


    # for id, placement_pose in enumerate(placement_pose_list):
    #     bgm_copy = bunny_gmodel.copy()
    #     bgm_copy.pose = placement_pose
    #     bgm_copy.pos = bgm_copy.pos+np.array([id*.05, 0, 0])
    #     bgm_copy.attach_to(base)

    class AnimeData(object):
        def __init__(self, placement_pose_list, support_facets=None):
            self.counter = 0
            self.gmodel = bunny_gmodel
            self.placement_pose_list = placement_pose_list
            self.support_facets = support_facets


    anime_data = AnimeData(placement_pose_list, support_facets)


    def update(animation_data, task):
        if animation_data.counter >= len(animation_data.placement_pose_list):
            animation_data.counter = 0
        if base.inputmgr.keymap["space"] is True:
            time.sleep(.1)
            animation_data.counter += 1
            print(animation_data.counter)
            animation_data.gmodel.pose = animation_data.placement_pose_list[animation_data.counter]
            animation_data.gmodel.detach()
            animation_data.gmodel.rgb=rm.bc.tab20_list[1]
            animation_data.gmodel.alpha=.3
            animation_data.gmodel.attach_to(base)
            if (animation_data.support_facets is not None):
                if animation_data.counter > 1:
                    animation_data.support_facets[animation_data.counter - 1].detach()
                animation_data.support_facets[animation_data.counter].pose = animation_data.placement_pose_list[
                    animation_data.counter]
                animation_data.support_facets[animation_data.counter].attach_to(base)
        return task.cont


    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[anime_data],
                          appendTask=True)
    base.run()
