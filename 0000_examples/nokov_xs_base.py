import numpy as np
from wrs import basis as rm, robot_sim as xa, modeling as gm
import wrs.visualization.panda.world as wd
from wrs.drivers.nokov.nokov_client import NokovClient


def genSphere(pos, radius=0.02, rgba=None):
    if rgba is None:
        rgba = [1, 0, 0, 1]
    gm.gen_sphere(pos=pos, radius=radius, rgba=rgba).attach_to(base)


if __name__ == '__main__':
    # nokov client init
    nokov_client = NokovClient()

    base = wd.World(cam_pos=[1.5, 0, 3], lookat_pos=[0, 0, .5])
    gm.gen_frame().attach_to(base)

    # robot_s
    component_name = 'agv'
    robot_1 = xa.XArmShuidi(enable_cc=True)
    robot_2 = xa.XArmShuidi(enable_cc=True)


    def update(rbtmnp, robot_1, robot_2, nokov_client: NokovClient, armname, task):
        rigidbody_dataframe = nokov_client.get_rigidbody_set_frame()
        if rigidbody_dataframe is not None:
            if rbtmnp[0] is not None:
                rbtmnp[0].detach()
            if rbtmnp[1] is not None:
                rbtmnp[1].detach()
            # robot 1
            rigidbodydata = rigidbody_dataframe.rigidbody_set_dict[1]
            rob_rotmat = rigidbodydata.get_rotmat()
            offset = np.array([0.16, 0.088, -0.461862])
            # mgm.gen_frame(pos=rigidbodydata.coord, rotmat=rob_rotmat, axis_length=1).attach_to(base)
            xyz_pose = rigidbodydata.get_pos() + np.dot(rob_rotmat, offset)
            pose = np.zeros(3)
            pose[:2] = xyz_pose[:2]
            theta = rm.quaternion_to_euler(rigidbodydata.quat)[2] - np.pi
            pose[2] = theta
            robot_1.fk(armname, pose)
            rbtmnp[0] = robot_1.gen_mesh_model(toggle_tcpcs=True)
            rbtmnp[0].attach_to(base)
            rigidbodydata.gen_mesh_model().attach_to(base)
            # robot 2
            rigidbodydata = rigidbody_dataframe.rigidbody_set_dict[2]
            rob_rotmat = rigidbodydata.get_rotmat()
            offset = np.array([0.12, 0.108, -0.461862])
            # mgm.gen_frame(pos=rigidbodydata.coord, rotmat=rob_rotmat, axis_length=1).attach_to(base)
            xyz_pose = rigidbodydata.get_pos() + np.dot(rob_rotmat, offset)
            pose = np.zeros(3)
            pose[:2] = xyz_pose[:2]
            theta = rm.quaternion_to_euler(rigidbodydata.quat)[2] - np.pi
            pose[2] = theta
            robot_2.fk(armname, pose)
            rbtmnp[1] = robot_2.gen_mesh_model(toggle_tcpcs=True)
            rbtmnp[1].attach_to(base)
        return task.again


    rbtmnp = [None, None]
    taskMgr.doMethodLater(0.1, update, "update",
                          extraArgs=[rbtmnp, robot_1, robot_2, nokov_client, component_name], appendTask=True)
    base.setFrameRateMeter(True)

    base.run()
