import numpy as np
from wrs import basis as rm, robot_sim as ur5e, modeling as gm
import wrs.visualization.panda.world as wd


def genSphere(pos, radius=0.02, rgba=None):
    if rgba is None:
        rgba = [1, 0, 0, 1]
    gm.gen_sphere(pos=pos, radius=radius, rgba=rgba).attach_to(base)

if __name__ == '__main__':

    base = wd.World(cam_pos=[1.5, 0, 3], lookat_pos=[0, 0, .5])
    gm.gen_frame().attach_to(base)

    # robot_s
    component_name = 'arm'
    robot_s = ur5e.UR5EConveyorBelt(enable_cc=True)
    robot_s.gen_meshmodel().attach_to(base)
    # base.run()
    # null space planning
    ratio =.01
    path = []
    for t in range(0, 5000, 1):
        print("-------- timestep = ", t, " --------")
        xa_jacob = robot_s.jacobian()
        # xa_ns = rm.null_space(xa_jacob)
        # cur_jnt_values = robot_s.get_jnt_values()
        # cur_jnt_values += np.ravel(xa_ns)*.01
        # nullspace rotate
        xa_ns = rm.null_space(xa_jacob[:3,:])
        cur_jnt_values = robot_s.get_jnt_values(component_name="arm")
        cur_jnt_values += np.ravel(xa_ns[:,2])*ratio
        status = robot_s.fk(component_name=component_name, jnt_values=cur_jnt_values)
        print(status, cur_jnt_values)
        if status == "succ":
            if t % 20 == 0:
                path.append(cur_jnt_values)
                robot_s.gen_meshmodel(rgba=[0, 1, 1, .1]).attach_to(base)
        else:
            ratio=-ratio

    def update(rbtmnp, motioncounter, robot, path, armname, task):
        if motioncounter[0] < len(path):
            if rbtmnp[0] is not None:
                rbtmnp[0].detach()
            pose = path[motioncounter[0]]
            robot.fk(armname, pose)
            rbtmnp[0] = robot.gen_mesh_model(toggle_tcpcs=True)
            rbtmnp[0].attach_to(base)
            genSphere(robot.get_gl_tcp(component_name)[0], radius=0.01, rgba=[1, 1, 0, 1])
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
    