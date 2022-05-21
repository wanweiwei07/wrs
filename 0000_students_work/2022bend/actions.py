import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import robot_sim.end_effectors.gripper.robotiq85.robotiq85 as rtq85
import robot_sim.end_effectors.gripper.robotiqhe.robotiqhe as rtqhe

base = wd.World(cam_pos=np.array([1,1,1]))
pos0 = np.array([0,0.07,.3])
rotmat0 = rm.rotmat_from_axangle([1,0,0], np.pi/12)
rotmat0 = rm.rotmat_from_axangle([0,1,0], np.pi/12)
rotmat0 = rm.rotmat_from_axangle([0,0,1], np.pi/9)
hnd0 = rtqhe.RobotiqHE()
hnd0.grip_at_with_jcpose(gl_jaw_center_pos=pos0, gl_jaw_center_rotmat=rotmat0, jaw_width=.005)
hnd0.gen_meshmodel().attach_to(base)
pos1 = np.array([0,-0.07,.3])
rotmat1 = rm.rotmat_from_axangle([-1,0,0], np.pi/12)
rotmat1 = rm.rotmat_from_axangle([0,-1,0], np.pi/12)
rotmat1 = rm.rotmat_from_axangle([0,0,-1], np.pi/9)
hnd1 = rtqhe.RobotiqHE()
hnd1.grip_at_with_jcpose(gl_jaw_center_pos=pos1, gl_jaw_center_rotmat=rotmat1, jaw_width=.005)
hnd1.gen_meshmodel().attach_to(base)
base.run()