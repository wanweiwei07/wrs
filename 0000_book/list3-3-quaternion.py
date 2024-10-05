from wrs import wd, rm, mgm

base = wd.World(cam_pos=rm.np.array([2, .0,.5]), lookat_pos=rm.np.zeros(3))
bunny = mgm.GeometricModel(initor="objects/bunnysim.stl")
bunny.alpha=.3
start_pos = rm.np.array([0, 0.3, 0])
start_rotmat = rm.np.eye(3)
goal_pos = rm.np.array([.5, -.3, .0])
goal_rotmat = rm.rotmat_from_euler(rm.np.pi , rm.np.pi/2, 0)
start_quaternion = rm.rotmat_to_quaternion(start_rotmat)
goal_quaternion = rm.rotmat_to_quaternion(goal_rotmat)
for t in rm.np.linspace(0, 1, 30):
    pos = start_pos * (1 - t) + goal_pos * t
    quat = rm.quaternion_slerp(start_quaternion, goal_quaternion, fraction=t)
    bunny_copy = bunny.copy()
    bunny_copy.rgb=rm.const.cool_map(t)
    mgm.gen_frame(alpha=.3).attach_to(bunny_copy)
    bunny_copy.pose = (pos, rm.quaternion_to_rotmat(quat))
    bunny_copy.attach_to(base)
base.run()