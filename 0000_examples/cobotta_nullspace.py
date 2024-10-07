from wrs import wd, rm, mgm, cbt

base = wd.World(cam_pos=rm.vec(1.2, 1.2, .5), lookat_pos=rm.vec(0, 0, .2))
mgm.gen_frame().attach_to(base)

# robot
component_name = 'arm'
robot = cbt.Cobotta()
tgt_pos = rm.np.array([0.25, .0, .25])
tgt_rotmat = rm.rotmat_from_axangle(rm.const.y_ax,rm.pi/2).dot(rm.rotmat_from_axangle(rm.const.z_ax, 0))
jnt_values = robot.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
if jnt_values is None:
    mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    base.run()
robot.goto_given_conf(jnt_values=jnt_values)
# angle = rm.angle_between_vectors(robot.gl_tcp_rotmat[:, 1], rm.np.array([0,1,0]))
# print(angle)
# robot.gen_meshmodel().attach_to(base)
# base.run()
task_rot = rm.rotmat_from_axangle(rm.const.y_ax,rm.pi/3)
j_rot = rm.np.eye(6)
j_rot[3:,3:] = task_rot
mgm.gen_frame(pos=tgt_pos, rotmat=task_rot.T).attach_to(base)
# null space planning
path = []
ratio = .001
for t in range(0, 5000, 1):
    print("-------- timestep = ", t, " --------")
    xa_jacob = j_rot.dot(robot.jacobian())
    # nullspace rotate
    xa_ns = rm.null_space(xa_jacob[[0,1,2,3,4], :])
    cur_jnt_values = robot.get_jnt_values()
    cur_jnt_values -= rm.np.ravel(xa_ns[:, 0]) * ratio
    # mmgm.gen_frame(pos=gl_tcp[0], rotmat=gl_tcp[1]).attach_to(base)
    print(xa_ns)
    print(robot.gl_tcp_rotmat[:3,2])
    robot.goto_given_conf(jnt_values=cur_jnt_values)
    if t % 200 == 0:
        path.append(cur_jnt_values)
        robot.gen_meshmodel(toggle_tcp_frame=True, rgb=rm.const.cyan, alpha=.3).attach_to(base)
path = path[::-1]
robot.fk(jnt_values=jnt_values)
ratio = -ratio
for t in range(0, 5000, 1):
    print("-------- timestep = ", t, " --------")
    xa_jacob = j_rot.dot(robot.jacobian())
    # xa_ns = rm.null_space(xa_jacob)
    xa_ns = rm.null_space(xa_jacob[[0,1,2,3,4], :])
    cur_jnt_values = robot.get_jnt_values()
    cur_jnt_values -= rm.np.ravel(xa_ns[:, 0]) * ratio
    # mmgm.gen_frame(pos=gl_tcp[0], rotmat=gl_tcp[1]).attach_to(base)
    print(xa_ns)
    print(robot.gl_tcp_rotmat[:3,2])
    robot.goto_given_conf(jnt_values=cur_jnt_values)
    # if status == "succ":
    if t % 200 == 0:
        path.append(cur_jnt_values)
        robot.gen_meshmodel(toggle_tcp_frame=True, rgb=rm.const.cyan, alpha=.3).attach_to(base)

# robot_x = cbtx.CobottaX()
# robot_x.move_jnts_motion(path)

# uncomment the following part for animation
def update(rbtmnp, motioncounter, robot, path, armname, task):
    if motioncounter[0] < len(path):
        if rbtmnp[0] is not None:
            rbtmnp[0].detach()
        jnt_values = path[motioncounter[0]]
        robot.goto_given_conf(jnt_values)
        rbtmnp[0] = robot.gen_meshmodel(toggle_tcp_frame=True)
        rbtmnp[0].attach_to(base)
        motioncounter[0] += 1
    else:
        motioncounter[0] = 0
    return task.again
rbtmnp = [None]
motioncounter = [0]
taskMgr.doMethodLater(0.1, update, "update",
                      extraArgs=[rbtmnp, motioncounter, robot, path, component_name], appendTask=True)
base.setFrameRateMeter(True)
base.run()
