from wrs import wd, gpa, mcm, rm
import wrs.robot_sim.end_effectors.grippers.or2fg7.or2fg7 as end_effector
import wrs.robot_sim.robots.khi as khi
import os

mesh_name = "bracketR1"
mesh_path = os.path.join(os.getcwd(), "meshes", mesh_name+".stl")
grasp_path = os.path.join(os.getcwd(), "pickles", mesh_name+"_grasp.pickle")

base = wd.World(cam_pos=rm.np.array([.5, .5, .5]), lookat_pos=rm.np.array([0, 0, 0]))
# mgm.gen_frame().attach_to(base)

obj_cmodel = mcm.CollisionModel(os.path.join(mesh_path, mesh_name))
obj_cmodel.attach_to(base)

gripper = end_effector.OR2FG7()
# gripper.gen_meshmodel().attach_to(base)
# base.run()
grasp_collection = gpa.plan_gripper_grasps(gripper,
                                           obj_cmodel,
                                           angle_between_contact_normals=rm.np.radians(180),
                                           rotation_interval=rm.np.radians(30),
                                           max_samples=150,
                                           min_dist_between_sampled_contact_points=.001,
                                           contact_offset=.001,
                                           toggle_dbg=False)
print(grasp_collection)
grasp_collection.save_to_disk(grasp_path)
for grasp in grasp_collection:
    gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos, jaw_center_rotmat=grasp.ac_rotmat, jaw_width=grasp.ee_values)
    gripper.gen_meshmodel(alpha=.1).attach_to(base)
base.run()
