import math
import numpy as np
import wrs.visualization.panda.world as wd
from wrs import basis as rm, robot_sim as ur3d, motion as rrtc, modeling as gm, modeling as cm

if __name__ == '__main__':
    base = wd.World(cam_pos=[2, 1, 3], lookat_pos=[0, 0, 1.1])
    gm.gen_frame().attach_to(base)

    # 牛乳パックの生成
    object1 = cm.CollisionModel("./objects/milkcarton.stl")
    object1.set_pos(np.array([.45, .48, 1.11]))
    object1.set_rgba([.8, .8, .8, 1])
    object1.attach_to(base)

    # マグカップの生成
    object2 = cm.CollisionModel("./objects/mug.stl")
    object2.set_pos(np.array([.3, .48, 1.11]))
    object2.set_rgba([.5, .7, .3, 1])
    object2.attach_to(base)

    # ロボットの生成
    component_name = 'lft_arm'
    robot_s = ur3d.UR3Dual()
    start_pos = np.array([.4, .3, 1.11])
    start_rotmat = rm.rotmat_from_euler(ai=math.pi/2, aj=math.pi, ak=0, axes='szxz')
    start_conf = robot_s.ik(component_name=component_name, tgt_pos=start_pos, tgt_rotmat=start_rotmat)
    goal_pos = np.array([.4, .7, 1.11])
    goal_rotmat = rm.rotmat_from_euler(ai=math.pi/2, aj=math.pi, ak=0, axes='szxz')
    goal_conf = robot_s.ik(component_name=component_name, tgt_pos=goal_pos, tgt_rotmat=goal_rotmat)

    # 経路計画
    rrtc_planner = rrtc.RRTConnect(robot_s)
    path = rrtc_planner.plan(component_name=component_name,
                             start_conf=start_conf,
                             goal_conf=goal_conf,
                             obstacle_list=[object1, object2],
                             ext_dist=.2,
                             max_time=300)
    print(path)
    for pose in path:
        print(pose)
        robot_s.fk(component_name, pose)
        robot_meshmodel = robot_s.gen_meshmodel()
        robot_meshmodel.attach_to(base)
        robot_s.gen_stickmodel().attach_to(base)

    base.run()