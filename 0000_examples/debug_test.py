import visualization.panda.world as wd
import robot_sim.robots.yumi.yumi as ym
import numpy as np

base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
robot_s = ym.Yumi()                                                          # simulation rbt_s
jnts= np.array([-1.95503661, - 1.50237315,  1.15998945,  1.02486284 , 3.42901897,- 0.30705591,
            1.83156597])
robot_s.fk("rgt_arm",jnts)
robot_s.gen_meshmodel().attach_to(base)
robot_s.show_cdprimit()
print(robot_s.is_collided())
base.run()