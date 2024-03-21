import numpy as np
import basis.robot_math as rm
import basis.trimesh.base as trm
import modeling.geometric_model as mgm
import modeling.collision_model as mcm
import modeling.model_collection as mmc
import grasping.planning.segmentation as seg
import modeling._ode_cdhelper as moh
import grasping.reasoner as gr


def tabletop_placements(obj_cmodel, stability_threshhold=.1, toggle_support_facets=False):
    """
    find all placements on a table (z axis is the plane normal; no consideration on symmetry)
    :param obj_cmodel:
    :param stability_threshhold: the ratio of (com_projection to support boundary)/(com to com_projection)
    :return:
    author: weiwei
    date: 20161213, 20240321osaka
    """
    convex_trm = obj_cmodel.trm_mesh.convex_hull
    seg_result = seg.overlapped_segmentation(model=convex_trm, max_normal_bias_angle=np.pi / 64)
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
            placement_pose_list.append((placement_pos, placement_rotmat))
            mgm.gen_arrow(spos=com, epos=contact_point).attach_to(facet)
            support_facet_list.append(facet)
    if toggle_support_facets:
        return placement_pose_list, support_facet_list
    return placement_pose_list


def tabletop_placements_and_grasps(tabletop_xy,
                                   obj_cmodel,
                                   robot,
                                   grasp_collection,
                                   stability_threshhold=.1,
                                   toggle_dbg=False):
    placement_pose_list, support_facet_list = tabletop_placements(obj_cmodel=obj_cmodel,
                                                                  stability_threshhold=stability_threshhold,
                                                                  toggle_support_facets=True)
    grasp_reasoner = gr.GraspReasoner(robot=robot)
    for placement_pose in placement_pose_list:
        pos = placement_pose[0] + np.array([tabletop_xy[0], tabletop_xy[1], 0])
        rotmat = placement_pose[1]
        feasible_gids = grasp_reasoner.find_feasible_gids(grasp_collection=grasp_collection,
                                                          obstacle_list=[mcm.gen_box(xyz_lengths=[5, 5, 0.1],
                                                                                     pos=np.array([0, 0, -0.1]),
                                                                                     rgba=[.7, .7, .7, .7])],
                                                          goal_pose=(pos, rotmat),
                                                          toggle_keep=True,
                                                          toggle_dbg=False)
        print(feasible_gids)
        if toggle_dbg:
            for f_gid in feasible_gids:
                print(f_gid)
                grasp = grasp_collection[f_gid]
                robot.end_effector.grip_at_by_pose(jaw_center_pos=pos + rotmat @ grasp.ac_pos,
                                                   jaw_center_rotmat=rotmat @ grasp.ac_rotmat,
                                                   jaw_width=grasp.ee_values)
                m_col = mmc.ModelCollection()
                obj_cmodel_copy = obj_cmodel.copy()
                obj_cmodel_copy.pose = (pos, rotmat)
                robot.end_effector.gen_meshmodel(rgb=rm.bc.green, alpha=.3).attach_to(m_col)
                obj_cmodel_copy.attach_to(m_col)
                m_col.attach_to(base)
                base.run()


if __name__ == '__main__':
    import os
    import time
    import basis
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    obj_path = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
    ground = mcm.gen_box(xyz_lengths=[.5, .5, .01], pos=np.array([0, 0, -0.01]), rgba=[.3, .3, .3, 1])
    ground.attach_to(base)
    bunny = mcm.CollisionModel(obj_path)
    placement_pose_list, support_facets = tabletop_placements(bunny, toggle_support_facets=True)


    # for id, placement_pose in enumerate(placement_pose_list):
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
