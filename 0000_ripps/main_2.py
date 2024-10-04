import os
import numpy as np
from wrs import basis as rm, robot_sim as cbtr, robot_sim as cbtg, motion as rrtc, modeling as cm, modeling as gm
import wrs.visualization.panda.world as wd
import utils
import wrs.motion.trajectory.piecewisepoly_toppra as trajp

if __name__ == '__main__':
    base = wd.World(cam_pos=[1.7, .7, .7], lookat_pos=[0, 0, 0])
    traj_gen = trajp.PiecewisePolyTOPPRA()
    max_vels = [np.pi * .6, np.pi * .4, np.pi, np.pi, np.pi, np.pi * 1.5]
    gm.gen_frame().attach_to(base)

    this_dir, this_filename = os.path.split(__file__)
    file_frame = os.path.join(this_dir, "meshes", "frame_bottom.stl")
    frame_bottom = cm.CollisionModel(file_frame)
    frame_bottom.set_rgba([.55, .55, .55, 1])
    frame_bottom.attach_to(base)

    table_plate = cm.gen_box(xyz_lengths=[.405, .26, .003])
    table_plate.set_pos([0.07 + 0.2025, .055, .0015])
    table_plate.set_rgba([.87, .87, .87, 1])
    table_plate.attach_to(base)

    file_dispose_box = os.path.join(this_dir, "objects", "tip_rack_cover.stl")
    dispose_box = cm.CollisionModel(file_dispose_box, ex_radius=.007)
    dispose_box.set_rgba([140 / 255, 110 / 255, 170 / 255, 1])
    dispose_box.set_pos(pos=np.array([.14, 0.07, .003]))
    dispose_box.attach_to(base)

    file_tip_rack = os.path.join(this_dir, "objects", "tip_rack.stl")
    tip_rack = utils.Rack96(file_tip_rack)
    tip_rack.set_rgba([140 / 255, 110 / 255, 170 / 255, 1])
    tip_rack.set_pose(pos=np.array([.3, 0.1, .003]), rotmat=rm.rotmat_from_axangle([0, 0, 1], np.pi/3))
    tip_rack.attach_to(base)

    file_tip = os.path.join(this_dir, "objects", "tip.stl")
    tip = cm.CollisionModel(file_tip)
    tip.set_rgba([200 / 255, 180 / 255, 140 / 255, 1])
    tip_cm_list = []
    for id_x in range(8):
        for id_y in range(12):
            pos, rotmat = tip_rack.get_rack_hole_pose(id_x=id_x, id_y=id_y)
            tip_new = tip.copy()
            tip_new.set_pose(pos, rotmat)
            # mgm.gen_frame(pos=pos, rotmat=rotmat).attach_to(base)
            tip_new.attach_to(base)
            tip_cm_list.append(tip_new)

    file_microplate = os.path.join(this_dir, "objects", "microplate_96.stl")
    microplate96 = utils.Microplate96(file_microplate)
    microplate96.set_rgba([200 / 255, 180 / 255, 140 / 255, 1])
    microplate96.set_pose(pos=np.array([.3, 0, .003]), rotmat=rm.rotmat_from_axangle([0, 0, 1], np.pi/3))
    microplate96.attach_to(base)

    file_microplate = os.path.join(this_dir, "objects", "microplate_24.stl")
    microplate24_0 = utils.Microplate96(file_microplate)
    microplate24_0.set_rgba([180 / 255, 180 / 255, 180 / 255, .7])
    # exactly 0.225, 0.31
    microplate24_0.set_pose(pos=np.array([.15, 0.23, -.03]), rotmat=rm.rotmat_from_axangle([0, 0, 1], np.pi / 2))
    microplate24_0.attach_to(base)
    microplate24_1 = microplate24_0.copy()
    microplate24_1.set_pose(pos=np.array([.3, 0.23, -.03]), rotmat=rm.rotmat_from_axangle([0, 0, 1], np.pi / 2))
    microplate24_1.attach_to(base)
    microplate24_2 = microplate24_0.copy()
    microplate24_2.set_pose(pos=np.array([.15, 0.32, -.03]), rotmat=rm.rotmat_from_axangle([0, 0, 1], np.pi / 2))
    microplate24_2.attach_to(base)
    microplate24_3 = microplate24_0.copy()
    microplate24_3.set_pose(pos=np.array([.3, 0.32, -.03]), rotmat=rm.rotmat_from_axangle([0, 0, 1], np.pi / 2))
    microplate24_3.attach_to(base)

    rbt_s = cbtr.CobottaRIPPS()
    init_joint_values = np.radians(np.asarray([0.0, 0.0, 70.0, 0.0, 90.0, 0.0]))
    component_name = "arm"
    rbt_s.fk(component_name=component_name, jnt_values=init_joint_values)
    rbt_s.gen_meshmodel().attach_to(base)

    planner = rrtc.RRTConnect(rbt_s)
    ee_s = cbtg.CobottaPipette()

    id_x = 0
    id_y = 0
    tip_pos, tip_rotmat = tip_rack.get_rack_hole_pose(id_x=id_x, id_y=id_y)
    z_offset = np.array([0, 0, .07])
    previous_jnt_values = rbt_s.get_jnt_values(component_name=component_name)
    goal_joint_values_attachment = utils.search_reachable_configuration(rbt_s=rbt_s,
                                                              ee_s=ee_s,
                                                              component_name=component_name,
                                                              tgt_pos=tip_pos + z_offset,
                                                              cone_axis=-tip_rotmat[:3, 2],
                                                              rotation_interval=np.radians(15),
                                                              obstacle_list=[frame_bottom])
    if goal_joint_values_attachment is not None:
        rbt_s.fk(component_name=component_name, jnt_values=goal_joint_values_attachment)
        rbt_s.gen_meshmodel().attach_to(base)
        path = planner.plan(component_name=component_name,
                            start_conf=init_joint_values,
                            goal_conf=goal_joint_values_attachment,
                            ext_dist=.02)
        n_path = len(path)
        previous_pos = None
        for id, jnt_values in enumerate(path):
            rbt_s.fk(component_name=component_name, jnt_values=jnt_values)
            rgba = rm.get_rgba_from_cmap(int(id * 256 / n_path), cm_name='cool', step=256)
            rgba[-1] = .3
            pos, rotmat = rbt_s.get_gl_tcp(manipulator_name=component_name)
            if previous_pos is not None:
                gm.gen_stick(previous_pos, pos, rgba=rgba).attach_to(base)
            previous_pos = pos
            # rbt_s.gen_meshmodel(rgba=rgba).attach_to(base)
        # interpolated_confs = \
        #     traj_gen.interpolate_by_max_spdacc(path,
        #                                        control_frequency=.008,
        #                                        max_vels=max_vels)
        # n_path = len(interpolated_confs)
        # print(n_path)
        # previous_pos = None
        # for id, jnt_values in enumerate(interpolated_confs):
        #     rbt_s.fk(component_name=component_name, jnt_values=jnt_values)
        #     rgba = rm.get_rgba_from_cmap(int(id * 256 / n_path), cm_name='cool', step=256)
        #     # rgba[-1] = 1
        #     pos, rotmat = rbt_s.get_gl_tcp(manipulator_name=component_name)
        #     # rgb_mat = np.eye(3)
        #     # rgb_mat[:, 0] = rgba[:3]
        #     # rgb_mat[:, 1] = rgba[:3]
        #     # rgb_mat[:, 2] = rgba[:3]
        #     # mgm.gen_frame(pos=pos, rotmat=rotmat, rgb_mat=rgb_mat, alpha=1, axis_length=.02, major_radius=.001).attach_to(base)
        #     # mgm.gen_sphere(pos=pos, major_radius=.005, rgba=rgba).attach_to(base)
        #     if previous_pos is not None:
        #         mgm.gen_stick(previous_pos, pos, rgba=rgba).attach_to(base)
        #     previous_pos = pos
        #     # rbt_s.gen_meshmodel(rgba=rgba).attach_to(base)
        new_joint_values = utils.search_reachable_configuration(rbt_s=rbt_s,
                                                                ee_s=ee_s,
                                                                component_name=component_name,
                                                                tgt_pos=tip_pos,
                                                                cone_axis=-tip_rotmat[:3, 2],
                                                                rotation_interval=np.radians(15),
                                                                obstacle_list=[frame_bottom],
                                                                seed_jnt_values=np.zeros(6))
        # mgm.gen_frame(pos=tip_pos, major_radius=.001).attach_to(base)
        rbt_s.fk(component_name=component_name, jnt_values=new_joint_values)
        # rbt_s.gen_meshmodel().attach_to(base)
        hnd_name = "hnd"
        rbt_s.hold(hnd_name=hnd_name, objcm=tip_cm_list[id_x * 12 + id_y])
    well_pos, well_rotmat = microplate96.get_rack_hole_pose(id_x=id_x, id_y=id_y)
    z_offset = np.array([0, 0, .05])
    goal_joint_values_aspiration = utils.search_reachable_configuration(rbt_s=rbt_s,
                                                             ee_s=ee_s,
                                                             component_name=component_name,
                                                             tgt_pos=well_pos + z_offset,
                                                             cone_axis=-well_rotmat[:3, 2],
                                                             rotation_interval=np.radians(15),
                                                             obstacle_list=[frame_bottom],
                                                             seed_jnt_values=np.zeros(6))
    if goal_joint_values_aspiration is not None:
        rbt_s.fk(component_name=component_name, jnt_values=goal_joint_values_aspiration)
        rbt_s.gen_meshmodel().attach_to(base)
        path = planner.plan(component_name=component_name,
                            start_conf=goal_joint_values_attachment,
                            goal_conf=goal_joint_values_aspiration,
                            ext_dist=.02)
        n_path = len(path)
        previous_pos = None
        for id, jnt_values in enumerate(path):
            rbt_s.fk(component_name=component_name, jnt_values=jnt_values)
            rgba = rm.get_rgba_from_cmap(int(id * 256 / n_path), cm_name='cool', step=256)
            rgba[-1] = .3
            pos, rotmat = rbt_s.get_gl_tcp(manipulator_name=component_name)
            if previous_pos is not None:
                gm.gen_stick(previous_pos, pos, rgba=rgba).attach_to(base)
            previous_pos = pos
            # rbt_s.gen_meshmodel(rgba=rgba).attach_to(base)


    well_pos, well_rotmat = microplate24_1.get_rack_hole_pose(id_x=0, id_y=0)
    z_offset = np.array([0, 0, .03])
    goal_joint_values_dispense = utils.search_reachable_configuration(rbt_s=rbt_s,
                                                                      ee_s=ee_s,
                                                                      component_name=component_name,
                                                                      tgt_pos=well_pos + z_offset,
                                                                      cone_axis=-well_rotmat[:3, 2],
                                                                      rotation_interval=np.radians(15),
                                                                      obstacle_list=[frame_bottom],
                                                                      seed_jnt_values=np.zeros(6))
    if goal_joint_values_dispense is not None:
        # microplate96.show_cdprimit()
        rbt_s.fk(component_name=component_name, jnt_values=goal_joint_values_dispense)
        rbt_s.gen_meshmodel().attach_to(base)
        path = planner.plan(component_name=component_name,
                            start_conf=goal_joint_values_aspiration,
                            goal_conf=goal_joint_values_dispense,
                            ext_dist=.02,
                            obstacle_list=[microplate96])
        n_path = len(path)
        previous_pos = None
        for id, jnt_values in enumerate(path):
            rbt_s.fk(component_name=component_name, jnt_values=jnt_values)
            rgba = rm.get_rgba_from_cmap(int(id * 256 / n_path), cm_name='cool', step=256)
            rgba[-1] = .3
            pos, rotmat = rbt_s.get_gl_tcp(manipulator_name=component_name)
            if previous_pos is not None:
                gm.gen_stick(previous_pos, pos, rgba=rgba).attach_to(base)
            previous_pos = pos
            # rbt_s.gen_meshmodel(rgba=rgba).attach_to(base)

    goal_joint_values_dispose = None
    pos = dispose_box.get_pos() + np.array([0, 0.05, .02])
    z_offset = np.array([0, 0.0, .04])
    rotmat = rm.rotmat_from_axangle([1, 0, 0], -np.pi * 7 / 18).dot(rm.rotmat_from_axangle([0, 1, 0], -np.pi))
    goal_joint_values_dispose = utils.search_reachable_configuration(rbt_s=rbt_s,
                                                                     ee_s=ee_s,
                                                                     component_name=component_name,
                                                                     tgt_pos=pos + z_offset,
                                                                     cone_axis=rotmat[:3, 2],
                                                                     cone_angle=np.pi / 18,
                                                                     rotation_interval=np.radians(22.5),
                                                                     obstacle_list=[frame_bottom])
    if goal_joint_values_dispose is not None:
        rbt_s.fk(component_name=component_name, jnt_values=goal_joint_values_dispose)
        rbt_s.gen_meshmodel().attach_to(base)
        # rbt_s.show_cdprimit()
        # dispose_box.show_cdprimit()
        # base.run()
        path = planner.plan(component_name=component_name,
                            start_conf=goal_joint_values_dispense,
                            goal_conf=goal_joint_values_dispose,
                            ext_dist=.02,
                            obstacle_list=[dispose_box, microplate96])
        n_path = len(path)
        previous_pos = None
        for id, jnt_values in enumerate(path):
            rbt_s.fk(component_name=component_name, jnt_values=jnt_values)
            rgba = rm.get_rgba_from_cmap(int(id * 256 / n_path), cm_name='cool', step=256)
            rgba[-1] = .3
            pos, rotmat = rbt_s.get_gl_tcp(manipulator_name=component_name)
            if previous_pos is not None:
                gm.gen_stick(previous_pos, pos, rgba=rgba).attach_to(base)
            previous_pos = pos
            # rbt_s.gen_meshmodel(rgba=rgba).attach_to(base)

    #
    # for angle in np.linspace(0, np.pi * 2, 25):
    #     rotmat = rm.rotmat_from_axangle([0, 0, 1], angle).dot(rotmat)
    #     jnt_values = rbt_s.ik(component_name=component_name, tgt_pos=well_pos + z_offset, tgt_rotmat=rotmat,
    #                           seed_jnt_values=previous_jnt_values)
    #     if jnt_values is not None:
    #         rbt_s.fk(jnt_values=jnt_values)
    #         rbt_s.gen_meshmodel().attach_to(base)
    #         rbt_s.hold(hnd_name=hnd_name, obj_cmodel=tip_cm_list[id_x * 12 + id_y])
    #         previous_jnt_values = jnt_values
    #         goal_joint_values = jnt_values
    #         break
    #     else:
    #         ee_s.grip_at_with_jcpose(jaw_center_pos=pos + z_offset, jaw_center_rotmat=rotmat, ee_values=0)
    #         ee_s.gen_meshmodel(rgba=[1, 0, 0, .3]).attach_to(base)

    base.run()
