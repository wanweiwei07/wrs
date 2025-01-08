from wrs import wd, rm, mgm, mcm, cbt, gpa, ppp

base = wd.World(cam_pos=rm.vec(2, 2, .5), lookat_pos=rm.vec(0, 0, .2))
mgm.gen_frame().attach_to(base)
# ground
ground = mcm.gen_box(xyz_lengths=rm.vec(5, 5, 1), rgb=rm.vec(.7, .7, .7), alpha=1)
ground.pos = rm.np.array([0, 0, -.5])
ground.attach_to(base)
ground.show_cdprim()
# robot
robot = cbt.Cobotta(enable_cc=True)
# object holder
obj_cmodel = mcm.CollisionModel("objects/holder.stl")
obj_cmodel.rgba = rm.np.array([.5, .5, .5, 1])
obj_cmodel.pos = rm.np.array([.2, 0, .3])
obj_cmodel.rotmat = rm.np.eye(3)
mgm.gen_frame().attach_to(obj_cmodel)
obj_cmodel.attach_to(base)
# planner
ppp = ppp.PickPlacePlanner(robot)

grasp_collection = gpa.GraspCollection.load_from_disk(file_name="cobotta_gripper_grasps.pickle")
pick_jnt_values = None
for grasp in grasp_collection:
    goal_jaw_center_pos = obj_cmodel.pos + obj_cmodel.rotmat.dot(grasp.ac_pos)
    goal_jaw_center_rotmat = obj_cmodel.rotmat.dot(grasp.ac_rotmat)
    jnt_values = robot.ik(tgt_pos=goal_jaw_center_pos, tgt_rotmat=goal_jaw_center_rotmat)
    if jnt_values is not None:
        robot.goto_given_conf(jnt_values=jnt_values, ee_values=grasp.ee_values)
        if not robot.is_collided(obstacle_list=[ground]):
            if not robot.end_effector.is_mesh_collided(cmodel_list=[ground]):
                pick_jnt_values = jnt_values
                break
if pick_jnt_values is not None:
    ppp.gen_pick_and_moveto_given_conf(obj_cmodel,
                                       start_jnt_values=pick_jnt_values,
                                       moveto_jnt_values=robot.home_conf,
                                       moveto_approach_direction=None,
                                       moveto_approach_distance=.07,
                                       grasp_jaw_width=.0,
                                       pick_approach_jaw_width=None,
                                       pick_approach_direction=None,
                                       pick_approach_distance=.07,
                                       pick_depart_direction=None,
                                       pick_depart_distance=.07,
                                       linear_granularity=.02,
                                       obstacle_list=None,
                                       use_rrt=True,
                                       toggle_dbg=False)
