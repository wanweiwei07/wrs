import robot_sim.end_effectors.suction.mvfln40.mvfln40 as suction
import robot_sim.end_effectors.gripper.robotiq85_gelsight.robotiq85_gelsight_pusher as rtqgel
import robot_sim.end_effectors.gripper.robotiq85.robotiq85 as rtq
import robot_sim.end_effectors.gripper.robotiq85_gelsight.robotiq85_gelsight_pusher as rtqgp
import modeling.geometric_model as gm
import modeling.collision_model as cm
import numpy as np
import visualization.panda.world as wd
import basis.robot_math as rm
import math
import grasping.annotation.utils as gau
import grasping.planning.antipodal as gpa
import grasping.planning.pushing as gpp

base = wd.World(cam_pos=[1.5, 1, 1], lookat_pos=[0, 0, 0])

# ground = gm.gen_box(np.array([2, 2, .001]))
# ground.set_rgba([1, 253 / 255, 219 / 255, 1])
# ground.attach_to(base)

# box_0 = cm.gen_box(np.array([.3, .3, .04]))
# box_0.set_rgba([153 / 255, 183 / 255, 1, .3])
# box_0.set_pos(np.array([0, 0, .02]))
# box_0.attach_to(base)

# suction_s = suction.MVFLN40()
# loc_pos_box = np.array([.1, 0, .02])
# loc_rotmat_box = rm.rotmat_from_euler(math.pi, 0, 0)
# gl_pos_box = box_0.get_pos() + box_0.get_rotmat().dot(loc_pos_box)
# gl_rotmat_box = box_0.get_rotmat().dot(loc_rotmat_box)
# suction_s.suction_to_with_scpose(gl_pos_box, gl_rotmat_box)
# suction_s.gen_meshmodel(rgba=[.55, .55, .55, .3]).attach_to(base)
# gm.gen_stick(
#     suction_s.pos,
#     np.array([0,0,1]),rgba=[1,0,0,.3]).attach_to(base)


rtq_s = rtq.Robotiq85()

box = cm.gen_box(np.array([.3, .3, .04]))
box.set_rgba([153 / 255, 183 / 255, 1, 1])
box.set_pos(np.array([0, 0, .02]))
# box.set_rotmat(rm.rotmat_from_axangle([0, 1, 0], -math.pi / 12))
grasp_info_list = gpa.plan_grasps(rtq_s, box, angle_between_contact_normals=math.radians(175), openning_direction='loc_y')
# for grasp_info in grasp_info_list:
#     jaw_width, gl_jaw_center_pos, gl_jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
#     rtq_s.fix_to(hnd_pos, hnd_rotmat)
#     rtq_s.jaw_to(jaw_width)
#     rtq_s.gen_meshmodel().attach_to(base)
grasp_info=grasp_info_list[11]
jaw_width, gl_jaw_center_pos, gl_jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
rtq_s.fix_to(hnd_pos, hnd_rotmat)
rtq_s.jaw_to(jaw_width)
rtq_s.gen_meshmodel().attach_to(base)

box.attach_to(base)
base.run()

rtq_s = rtqgp.Robotiq85GelsightPusher()

box = cm.gen_box(np.array([.3, .3, .04]))
box.set_rgba([153 / 255, 183 / 255, 1, 1])
box.set_pos(np.array([0, 0, .02]))
# box.set_rotmat(rm.rotmat_from_axangle([0, 1, 0], -math.pi / 12))
push_info_list = gpp.plan_pushing(rtq_s, box, cone_angle=math.radians(60),
                                  icosphere_level=1,
                                  min_dist_between_sampled_contact_points=.02,
                                  max_samples=70,
                                  local_rotation_interval=math.radians(90),
                                  contact_offset=.01,
                                  toggle_debug=False)
for push_info in push_info_list:
    gl_push_pos, gl_push_rotmat, hnd_pos, hnd_rotmat = push_info
    gic = rtq_s.copy()
    gic.fix_to(hnd_pos, hnd_rotmat)
    gic.gen_meshmodel().attach_to(base)

# push_info=push_info_list[70]
# gl_push_pos, gl_push_rotmat, hnd_pos, hnd_rotmat = push_info
# gic = rtq_s.copy()
# gic.fix_to(hnd_pos, hnd_rotmat)
# gic.gen_meshmodel().attach_to(base)

box.attach_to(base)

base.run()
