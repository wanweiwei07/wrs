if __name__ == '__main__':
    import math
    import numpy as np
    import random
    from wrs import basis as rm, robot_sim as cbt, motion as rrtc, modeling as gm
    import wrs.robot_con.cobotta.cobotta_x as cbtx
    import wrs.visualization.panda.world as wd
    import wrs._misc.promote_rt as pr

    base = wd.World(cam_pos=[1, 1, .5], lookat_pos=[0, 0, .2])
    gm.gen_frame().attach_to(base)

    pr.set_realtime()
    robot_s = cbt.Cobotta()
    robot_x = cbtx.CobottaX()
    while True:
        start_conf = robot_x.get_jnt_values()
        # tgt_pos = np.array([.25, .2, .15])
        # tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2/ 3)
        x = random.uniform(0, .5)
        y = random.uniform(-.37, .37)
        z = random. uniform(.1, .5)
        tgt_pos = np.array([x,y,z])
        angle = random.uniform(math.pi/3, math.pi)
        # print(math.degrees(angle))
        tgt_rotmat = rm.rotmat_from_axangle([0,1,0], angle)
        # mgm.gen_frame(tgt_pos, tgt_rotmat).attach_to(base)
        # robot_s.gen_meshmodel().attach_to(base)
        # base.run()
        jnt_values = robot_s.ik(tgt_pos=tgt_pos,tgt_rotmat=tgt_rotmat)
        if jnt_values is None:
            continue
        rrtc_planner = rrtc.RRTConnect(robot_s)
        path = rrtc_planner.plan(component_name="arm",
                                 start_conf=start_conf,
                                 goal_conf=jnt_values,
                                 ext_dist=1,
                                 max_time=300)
        if path is None:
            continue
        robot_x.move_jnts_motion(path)
    # robot_x.close_gripper()
    # for pose in path:
    #     robot_s.fk("arm", pose)
    #     robot_meshmodel = robot_s.gen_meshmodel()
    #     robot_meshmodel.attach_to(base)
    base.run()