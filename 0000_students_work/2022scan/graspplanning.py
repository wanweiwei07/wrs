import wrs.visualization.panda.world as wd
import wrs.grasping.planning.antipodal as gpa
from wrs import robot_sim as rtq85, modeling as gm, modeling as cm

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0], auto_cam_rotate=True)
gm.gen_frame().attach_to(base)
object_bunny = cm.CollisionModel("objects/pblcm_cropped_8_2_20000_cvt.stl")
object_bunny.set_rgba([.9, .75, .35, .3])
object_bunny.attach_to(base)
# grippers
gripper_s = rtq85.Robotiq85()
# base.run()
# grpr.gen_meshmodel(toggle_jnt_frames=True, toggle_flange_frame=True).attach_to(base)
# base.run()
grasp_info_list = gpa.plan_gripper_grasps(gripper_s, object_bunny, openning_direction ='loc_y', max_samples=100, min_dist_between_sampled_contact_points=.01)
for grasp_info in grasp_info_list:
    aw_width, gl_jaw_center, gl_jaw_rotmat, hnd_pos, hnd_rotmat = grasp_info
    gripper_s.fix_to(hnd_pos, hnd_rotmat)
    gripper_s.change_jaw_width(aw_width)
    gripper_s.gen_meshmodel().attach_to(base)
    # break
base.run()