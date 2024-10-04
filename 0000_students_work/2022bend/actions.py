import numpy as np
from wrs import basis as rm, robot_sim as rtqhe
import wrs.visualization.panda.world as wd

base = wd.World(cam_pos=np.array([1,1,1]))
pos0 = np.array([0,0.07,.3])
rotmat0 = rm.rotmat_from_axangle([1,0,0], np.pi/12)
rotmat0 = rm.rotmat_from_axangle([0,1,0], np.pi/12)
rotmat0 = rm.rotmat_from_axangle([0,0,1], np.pi/9)
hnd0 = rtqhe.RobotiqHE()
hnd0.grip_at_by_pose(jaw_center_pos=pos0, jaw_center_rotmat=rotmat0, jaw_width=.005)
hnd0.gen_meshmodel().attach_to(base)
pos1 = np.array([0,-0.07,.3])
rotmat1 = rm.rotmat_from_axangle([-1,0,0], np.pi/12)
rotmat1 = rm.rotmat_from_axangle([0,-1,0], np.pi/12)
rotmat1 = rm.rotmat_from_axangle([0,0,-1], np.pi/9)
hnd1 = rtqhe.RobotiqHE()
hnd1.grip_at_by_pose(jaw_center_pos=pos1, jaw_center_rotmat=rotmat1, jaw_width=.005)
hnd1.gen_meshmodel().attach_to(base)
base.run()