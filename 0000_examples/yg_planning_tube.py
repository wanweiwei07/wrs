from wrs import wd, rm, gpa, mcm, mgm
import wrs.robot_sim.end_effectors.grippers.yumi_gripper.yumi_gripper as ee

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)
# tube
obj_cmodel = mcm.CollisionModel("objects/tubebig.stl")
obj_cmodel.rgba = rm.np.array([.9, .75, .35, 1])
obj_cmodel.attach_to(base)
# grippers
gripper = ee.YumiGripper()
grasp_collection = gpa.plan_gripper_grasps(gripper, obj_cmodel,
                                           angle_between_contact_normals=rm.radians(177),
                                           max_samples=15, min_dist_between_sampled_contact_points=.005,
                                           contact_offset=.005)
grasp_collection.save_to_disk(file_name='yumi_tube_big.pickle')
for grasp in grasp_collection:
    gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos, jaw_center_rotmat=grasp.ac_rotmat, jaw_width=grasp.ee_values)
    gripper.gen_meshmodel(alpha=.3).attach_to(base)
base.run()
