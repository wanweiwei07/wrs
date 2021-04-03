import math
import numpy as np
import visualization.panda.world as wd
import modeling.geometricmodel as gm
import modeling.collisionmodel as cm
import robotsim.robots.ur3_dual.ur3_dual as ur3d
import motion.probabilistic.rrt_connect as rrtc

base = wd.World(campos=[2, 1, 3], lookatpos=[0, 0, 1.1])
gm.gen_frame().attach_to(base)
# object
object = cm.CollisionModel("./objects/bunnysim.stl")
object.set_pos(np.array([.55, -.3, 1.3]))
object.set_rgba([.5,.7,.3,1])
object.attach_to(base)
# robot
component_name='rgt_arm'
robot_instance = ur3d.UR3Dual()

# robot_instance.fk(component_name, np.array([0, math.pi/9, 0, math.pi, 0, -math.pi / 6]))
robot_instance.fk(component_name, np.array([math.pi/3, math.pi, 0, math.pi/2, 0, math.pi / 6]))
robot_meshmodel = robot_instance.gen_meshmodel()
robot_meshmodel.attach_to(base)
robot_instance.show_cdprimit()
is_collided, contact_points  = robot_instance.is_collided(toggle_contact_points=True)
for point in contact_points:
    gm.gen_sphere(point, radius=10).attach_to(base)
base.run()
rrtc_planner = rrtc.RRTConnect(robot_instance)
path = rrtc_planner.plan(start_conf=np.array([0, math.pi/9, 0, math.pi, 0, -math.pi / 6]),
                         goal_conf=np.array([math.pi/3, math.pi, 0, math.pi/2, 0, math.pi / 6]),
                         obstacle_list=[object],
                         ext_dist=.1,
                         rand_rate=70,
                         maxtime=300,
                         component_name=component_name)
print(path)
for pose in path:
    print(pose)
    robot_instance.fk(component_name, pose)
    robot_meshmodel = robot_instance.gen_meshmodel()
    robot_meshmodel.attach_to(base)
    robot_instance.gen_stickmodel().attach_to(base)

base.run()
