import math
import numpy as np
from wrs import basis as rm, robot_sim as cbt, modeling as gm
import wrs.visualization.panda.world as wd


def genSphere(pos, radius=0.02, rgba=None):
    if rgba is None:
        rgba = [1, 0, 0, 1]
    gm.gen_sphere(pos=pos, radius=radius, rgba=rgba).attach_to(base)


if __name__ == '__main__':

    base = wd.World(cam_pos=[1.2, 1.2, .5], lookat_pos=[0, 0, .2])
    gm.gen_frame().attach_to(base)

    # robot_s
    component_name = 'arm'
    robot_s = cbt.CobottaRIPPS()
    tgt_pos = np.array([0.25, .0, .15])
    tgt_rotmat = rm.rotmat_from_axangle([0,1,0], math.pi/2).dot(rm.rotmat_from_axangle([0,0,1], 0))
    jnt_values = robot_s.ik(component_name=component_name, tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
    if jnt_values is None:
        gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
        base.run()
    robot_s.fk(component_name=component_name,
               jnt_values=jnt_values)
    # pos, rotmat = robot_s.get_gl_tcp(manipulator_name=component_name)
    # angle = rm.angle_between_vectors(rotmat[:, 1], np.array([0,1,0]))
    # print(angle)
    # robot_s.gen_meshmodel().attach_to(base)
    # base.run()

    gl_tcp = robot_s.get_gl_tcp(manipulator_name="arm")
    task_rot = rm.rotmat_from_axangle([0,0,1], math.pi/6)
    j_rot = np.eye(6)
    # j_rot[3:,3:] = task_rot
    # mgm.gen_frame(pos=tgt_pos, rotmat=task_rot.T).attach_to(base)
    # null space planning
    path = []
    ratio = .001
    for t in range(0, 200, 1):
        print("-------- timestep = ", t, " --------")
        xa_jacob = j_rot.dot(robot_s.jacobian())
        # nullspace rotate
        xa_z = rm.null_space(xa_jacob[[0,1,2,3,4], :])
        cur_jnt_values = robot_s.get_jnt_values(component_name=component_name)
        cur_jnt_values -= np.ravel(xa_z[:, 0]) * ratio
        status = robot_s.fk(component_name=component_name, jnt_values=cur_jnt_values)
        for t_ext in range(0, 200, 1):
            xa_jacob = j_rot.dot(robot_s.jacobian())
            xa_x = rm.null_space(xa_jacob[[0,1,2,4,5], :])
            cur_jnt_values_ext = cur_jnt_values-np.ravel(xa_x[:, 0]) * ratio
            # for t_ext in range(0, 200, 20):
            #     cur_jnt_values -= np.ravel(xa_x[:, 0]) * ratio
            #     for t_ext in range(0, 200, 20):
            #         cur_jnt_values -= np.ravel(xa_y[:, 0]) * ratio
            #         # mgm.gen_frame(pos=gl_tcp[0], rotmat=gl_tcp[1]).attach_to(base)
            #         status = robot_s.fk(component_name=component_name, jnt_values=cur_jnt_values)
            #         # if status == "succ":
            status = robot_s.fk(component_name=component_name, jnt_values=cur_jnt_values_ext)
            if t % 20 == 0:
                path.append(cur_jnt_values)
                robot_s.gen_meshmodel(rgba=[0, 1, 1, .1]).attach_to(base)
                pos, rotmat = robot_s.get_gl_tcp(manipulator_name=component_name)
                gm.gen_frame(pos=pos, rotmat=rotmat).attach_to(base)
            robot_s.fk(component_name=component_name, jnt_values=cur_jnt_values)
    # path = path[::-1]
    # robot_s.fk(component_name=component_name,
    #            jnt_values=jnt_values)
    # ratio = -ratio
    # for t in range(0, 5000, 1):
    #     print("-------- timestep = ", t, " --------")
    #     xa_jacob = j_rot.dot(robot_s.jacobian())
    #     # xa_ns = rm.null_space(xa_jacob)
    #     xa_ns = rm.null_space(xa_jacob[[0,1,2], :])
    #     cur_jnt_values = robot_s.get_jnt_values(component_name=component_name)
    #     cur_jnt_values -= np.ravel(xa_ns[:, 1]) * ratio
    #     # if t < 50:
    #     #     cur_jnt_values -= np.ravel(xa_ns[:, 0]) * ratio
    #     #     cur_jnt_values -= np.ravel(xa_ns[:, 1]) * ratio
    #     # mgm.gen_frame(pos=gl_tcp[0], rotmat=gl_tcp[1]).attach_to(base)
    #     print(xa_ns)
    #     print(gl_tcp[1][:3,2])
    #     status = robot_s.fk(component_name=component_name, jnt_values=cur_jnt_values)
    #     # if status == "succ":
    #     if t % 20 == 0:
    #         path.append(cur_jnt_values)
    #         robot_s.gen_meshmodel(rgba=[0, 1, 1, .1]).attach_to(base)
    #         pos, rotmat = robot_s.get_gl_tcp(manipulator_name=component_name)
    #         mgm.gen_frame(pos=pos, rotmat=rotmat).attach_to(base)

    # robot_x = cbtx.CobottaX()
    # robot_x.move_jnts_motion(path)

    # uncomment the following part for animation
    def update(rbtmnp, motioncounter, robot, path, armname, task):
        if motioncounter[0] < len(path):
            if rbtmnp[0] is not None:
                rbtmnp[0].detach()
            pose = path[motioncounter[0]]
            robot.fk(armname, pose)
            rbtmnp[0] = robot.gen_mesh_model(toggle_tcpcs=True)
            rbtmnp[0].attach_to(base)
            # genSphere(robot.get_gl_tcp(component_name)[0], major_radius=0.01, rgba=[1, 1, 0, 1])
            motioncounter[0] += 1
        else:
            motioncounter[0] = 0
        return task.again
    rbtmnp = [None]
    motioncounter = [0]
    taskMgr.doMethodLater(0.1, update, "update",
                          extraArgs=[rbtmnp, motioncounter, robot_s, path, component_name], appendTask=True)
    base.setFrameRateMeter(True)
    base.run()
