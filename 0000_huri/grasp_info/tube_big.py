import math
import numpy as np
from wrs import basis as rm, robot_sim as yg, modeling as cm
import wrs.grasping.annotation.gripping as gutil

if __name__ == '__main__':

    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[.5, .5, .3], lookat_pos=[0, 0, 0])
    gripper_instance = yg.YumiGripper(enable_cc=True, cdmesh_type='aabb')
    objcm = cm.CollisionModel('../objects/tubebig.stl', cdmesh_type='convex_hull')
    objcm.attach_to(base)
    objcm.show_local_frame()
    grasp_info_list = []
    for height in [.08, .095]:
        for roll_angle in [math.pi*.1, math.pi*.2]:
            gl_hndz = rm.rotmat_from_axangle(np.array([1,0,0]), roll_angle).dot(np.array([0,0,-1]))
            grasp_info_list += gutil.define_gripper_grasps_with_rotation(gripper_instance, objcm,
                                                                         jaw_center_pos=np.array([0, 0, height]),
                                                                         approaching_direction=gl_hndz,
                                                                         thumb_opening_direction=, jaw_width=.025)
    for grasp_info in grasp_info_list:
        jaw_width, gl_jaw_center, pos, rotmat = grasp_info
        # gic = grpr.copy()
        gripper_instance.fix_to(pos, rotmat)
        gripper_instance.change_jaw_width(jaw_width)
        gripper_instance.gen_meshmodel().attach_to(base)
    gutil.write_pickle_file(cmodel_name='tubebig', grasp_info_list=grasp_info_list)
    base.run()