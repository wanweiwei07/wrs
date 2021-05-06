import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.grippers.robotiq85.robotiq85 as rtq85

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)
# object
# object_box = cm.gen_box(extent=[.02, .06, 1])
# object_box.set_rgba([.7, .5, .3, .7])
# object_box.attach_to(base)
object_bunny = cm.CollisionModel("objects/bunnysim.stl")
object_bunny.set_rgba([.9, .75, .35, .3])
object_bunny.attach_to(base)
# hnd_s
gripper_s = rtq85.Robotiq85()
grasp_info_list = gpa.plan_grasps(gripper_s, object_bunny, max_samples=100, min_dist_between_sampled_contact_points=.01)
for grasp_info in grasp_info_list:
    aw_width, gl_jaw_center, hnd_pos, hnd_rotmat = grasp_info
    gripper_s.fix_to(hnd_pos, hnd_rotmat)
    gripper_s.jaw_to(aw_width)
    gripper_s.gen_meshmodel().attach_to(base)
base.run()