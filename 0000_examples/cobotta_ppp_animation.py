import visualization.panda.world as wd
import modeling.geometric_model as mgm
import modeling.collision_model as mcm
import grasping.planning.antipodal as gpa
import math
import numpy as np
import basis.robot_math as rm
import robot_sim.robots.cobotta.cobotta as cbt
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc

base = wd.World(cam_pos=[1.2, .7, 1], lookat_pos=[.0, 0, .15])
mgm.gen_frame().attach_to(base)
# ground
ground = mcm.gen_box(xyz_lengths=[5, 5, 1], rgba=[.7, .7, .7, .7])
ground.pos = np.array([0, 0, -.51])
ground.attach_to(base)
# object holder
holder_1 = mcm.CollisionModel("objects/holder.stl")
holder_1.rgba= np.array([.5,.5,.5,1])
h1_gl_pos = np.array([-.15, -.3, .0])
h1_gl_rotmat = np.eye(3)
holder_1.pos = h1_gl_pos
holder_1.rotmat = h1_gl_rotmat
mgm.gen_frame().attach_to(holder_1)
h1_copy = holder_1.copy()
h1_copy.attach_to(base)
# object holder goal
holder_2 = mcm.CollisionModel("objects/holder.stl")
h2_gl_pos = np.array([.25, -.05, .05])
h2_gl_rotmat = rm.rotmat_from_euler(0, 0, -math.pi / 2)
holder_2.pos = h2_gl_pos
holder_2.rotmat = h2_gl_rotmat
h2_copy = holder_2.copy()

robot = cbt.Cobotta()
robot.gen_meshmodel(rgb=rm.bc.jet_map(0.0)).attach_to(base)
robot.end_effector.hold(holder_1)
robot.goto_given_conf(jnt_values=np.array([np.pi/3,np.pi/2,np.pi/2,np.pi,np.pi,np.pi]))
robot.gen_meshmodel(rgb=rm.bc.jet_map(.2)).attach_to(base)
robot.goto_given_conf(jnt_values=np.array([0,np.pi/2,np.pi/2,np.pi,np.pi,np.pi]))
robot.end_effector.release(holder_1)
robot.gen_meshmodel(rgb=rm.bc.jet_map(.4)).attach_to(base)
base.run()

rrtc = rrtc.RRTConnect(robot)
ppp = ppp.PickPlacePlanner(robot)

original_grasp_info_list = gpa.load_pickle_file(obj_name='holder', path='./', file_name='cobg_holder_grasps.pickle')
start_conf = robot.get_jnt_values()


conf_list, jawwidth_list, objpose_list = \
    ppp.gen_pick_and_place_motion(hnd_name=hand_name,
                                    objcm=object_holder,
                                    grasp_info_list=original_grasp_info_list,
                                    start_conf=start_conf,
                                    end_conf=start_conf,
                                    goal_homomat_list=[obgl_start_homomat, obgl_goal_homomat],
                                    approach_direction_list=[None, np.array([0, 0, -1])],
                                    approach_distance_list=[.2] * 2,
                                    depart_direction_list=[np.array([0, 0, 1]), None],
                                    depart_distance_list=[.2] * 2)
robot_attached_list = []
object_attached_list = []
counter = [0]
def update(robot_s,
           object_box,
           robot_path,
           jawwidth_path,
           obj_path,
           robot_attached_list,
           object_attached_list,
           counter,
           task):
    if counter[0] >= len(robot_path):
        counter[0] = 0
    if len(robot_attached_list) != 0:
        for robot_attached in robot_attached_list:
            robot_attached.detach()
        for object_attached in object_attached_list:
            object_attached.detach()
        robot_attached_list.clear()
        object_attached_list.clear()
    pose = robot_path[counter[0]]
    robot_s.fk(manipulator_name, pose)
    robot_s.change_jaw_width(hand_name, jawwidth_path[counter[0]])
    robot_meshmodel = robot_s.gen_mesh_model()
    robot_meshmodel.attach_to(base)
    robot_attached_list.append(robot_meshmodel)
    obj_pose = obj_path[counter[0]]
    objb_copy = object_box.copy()
    objb_copy.set_rgba([1,0,0,1])
    objb_copy.set_homomat(obj_pose)
    objb_copy.attach_to(base)
    object_attached_list.append(objb_copy)
    counter[0] += 1
    return task.again

taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[robot_s,
                                 object_holder,
                                 conf_list,
                                 jawwidth_list,
                                 objpose_list,
                                 robot_attached_list,
                                 object_attached_list,
                                 counter],
                      appendTask=True)
base.run()
