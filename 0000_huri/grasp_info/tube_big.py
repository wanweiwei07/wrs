import grasping.annotated.utils as gutil

if __name__ == '__main__':

    import os
    import basis
    import robotsim.grippers.yumi_gripper.yumi_gripper as yg
    import modeling.collisionmodel as cm
    import visualization.panda.world as wd

    base = wd.World(campos=[.5, .5, .3], lookatpos=[0, 0, 0])
    gripper_instance = yg.YumiGripper(enable_cc=True)
    objpath = os.path.join(file.__path__[0], 'objects', 'tubebig.stl')
    objcm = cm.CollisionModel(objpath)
    objcm.attach_to(base)
    objcm.show_localframe()
    # base.run()
    grasp_info_list = define_grasp_with_rotation(gripper_instance, objcm,
                                                 gl_jaw_center=np.array([0,0,0]),
                                                 gl_hndz=np.array([1,0,0]),
                                                 gl_hndx=np.array([0,1,0]),
                                                 jaw_width=.04)
    for grasp_info in grasp_info_list:
        jaw_width, gl_jaw_center, pos, rotmat = grasp_info
        gic = gripper_instance.copy()
        gic.fix_to(pos, rotmat)
        gic.jaw_to(jaw_width)
        print(pos, rotmat)
        gic.gen_meshmodel().attach_to(base)
    base.run()
