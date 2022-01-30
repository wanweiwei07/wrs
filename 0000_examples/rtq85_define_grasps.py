import grasping.annotation.utils as gau

if __name__ == '__main__':
    import numpy as np
    import robot_sim.end_effectors.gripper.robotiq85.robotiq85 as rtq85
    import modeling.collision_model as cm
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[.5, .5, .3], lookat_pos=[0, 0, 0])
    gripper_s = rtq85.Robotiq85(enable_cc=True)
    objcm = cm.CollisionModel("./objects/bunnysim.stl")
    objcm.set_pos(np.array([.5,-.3,1.2]))
    objcm.attach_to(base)
    objcm.show_localframe()
    grasp_info_list = gau.define_grasp_with_rotation(gripper_s,
                                                     objcm,
                                                     gl_jaw_center_pos=np.array([0, 0, 0]),
                                                     gl_jaw_center_z=np.array([1, 0, 0]),
                                                     gl_hndx=np.array([0, 1, 0]),
                                                     jaw_width=.04,
                                                     gl_rotation_ax=np.array([0, 0, 1]))
    for grasp_info in grasp_info_list:
        jaw_width, gl_jaw_center, pos, rotmat = grasp_info
        gic = gripper_s.copy()
        gic.fix_to(pos, rotmat)
        gic.jaw_to(jaw_width)
        print(pos, rotmat)
        gic.gen_meshmodel().attach_to(base)
    base.run()
