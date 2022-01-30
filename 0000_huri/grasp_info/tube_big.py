import math
import numpy as np
import basis.robot_math as rm
import grasping.annotation.utils as gutil

if __name__ == '__main__':

    import robot_sim.end_effectors.gripper.yumi_gripper.yumi_gripper as yg
    import modeling.collision_model as cm
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[.5, .5, .3], lookat_pos=[0, 0, 0])
    gripper_instance = yg.YumiGripper(enable_cc=True, cdmesh_type='aabb')
    objcm = cm.CollisionModel('../objects/tubebig.stl', cdmesh_type='convex_hull')
    objcm.attach_to(base)
    objcm.show_localframe()
    grasp_info_list = []
    for height in [.08, .095]:
        for roll_angle in [math.pi*.1, math.pi*.2]:
            gl_hndz = rm.rotmat_from_axangle(np.array([1,0,0]), roll_angle).dot(np.array([0,0,-1]))
            grasp_info_list += gutil.define_grasp_with_rotation(gripper_instance,
                                                                objcm,
                                                                gl_jaw_center_pos=np.array([0, 0, height]),
                                                                gl_jaw_center_z=gl_hndz,
                                                                gl_hndx=np.array([1,0,0]),
                                                                jaw_width=.025,
                                                                gl_rotation_ax=np.array([0, 0, 1]))
    for grasp_info in grasp_info_list:
        jaw_width, gl_jaw_center, pos, rotmat = grasp_info
        # gic = gripper_s.copy()
        gripper_instance.fix_to(pos, rotmat)
        gripper_instance.jaw_to(jaw_width)
        gripper_instance.gen_meshmodel().attach_to(base)
    gutil.write_pickle_file(objcm_name='tubebig', grasp_info_list=grasp_info_list)
    base.run()