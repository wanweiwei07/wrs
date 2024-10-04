import wrs.visualization.panda.world as wd
import wrs.grasping.planning.antipodal as gpa
from wrs import robot_sim as xag, modeling as gm, modeling as cm

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)
object_box = cm.gen_box(xyz_lengths=[.02, .06, .7])
object_box.set_rgba([.7, .5, .3, 1])
object_box.attach_to(base)
# grippers
gripper_s = xag.XArmGripper()
grasp_info_list = gpa.plan_gripper_grasps(gripper_s, object_box, openning_direction='loc_y', max_samples=10, min_dist_between_sampled_contact_points=.05)
gpa.write_pickle_file('box', grasp_info_list, './', 'xarm_long_box.pickle')
for grasp_info in grasp_info_list:
    jaw_width, gl_jaw_center_pos, gl_jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    gripper_s.fix_to(hnd_pos, hnd_rotmat)
    gripper_s.change_jaw_width(jaw_width)
    gripper_s.gen_meshmodel().attach_to(base)
base.run()