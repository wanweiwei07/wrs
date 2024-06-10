import mujoco
import basis.trimesh as trm
import modeling.collision_model as mcm
import modeling.geometric_model as mgm
import numpy as np
import basis.robot_math as rm


def cvt_geom(model, geom_id):
    """
    Convert the geom of the MuJoCo model to a geometric model.
    :param model: MjModel: The MuJoCo model.
    :param geom_id: int: The ID of the geom.
    :return: CollisionModel: The geometric model.
    author: weiwei
    date: 20240528
    """
    geom_type = model.geom_type[geom_id]
    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = model.geom_dataid[geom_id]
        print(model.mesh(mesh_id).name)
        vertadr_list = model.mesh(mesh_id).vertadr
        vertnum_list = model.mesh(mesh_id).vertnum
        faceadr_list = model.mesh(mesh_id).faceadr
        facenum_list = model.mesh(mesh_id).facenum
        for vertadr, vertnum in zip(vertadr_list, vertnum_list):
            vertices = model.mesh_vert[vertadr:(vertadr + vertnum)]
        for faceadr, facenum in zip(faceadr_list, facenum_list):
            faces = model.mesh_face[faceadr:(faceadr + facenum)]
        return mcm.CollisionModel(initor=trm.Trimesh(vertices=vertices, faces=faces),
                                  name=model.mesh(mesh_id).name,
                                  rgb=model.geom_rgba[geom_id][:3],
                                  alpha=model.geom_rgba[geom_id][3])
    elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        box_size = model.geom_size[geom_id]
        return mcm.gen_box(xyz_lengths=2 * box_size, rgb=model.geom_rgba[geom_id][:3],
                           alpha=model.geom_rgba[geom_id][3])
    elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        capsule_radius = model.geom_size[geom_id][0]
        capsule_half_length = model.geom_size[geom_id][1]
        return mcm.gen_stick(spos=np.array([0, 0, -capsule_half_length]),
                             epos=np.array([0, 0, capsule_half_length]),
                             radius=capsule_radius,
                             type="round",
                             rgb=model.geom_rgba[geom_id][:3],
                             alpha=model.geom_rgba[geom_id][3],
                             n_sec=12)
    elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        cylinder_radius = model.geom_size[geom_id][0]
        cylinder_half_length = model.geom_size[geom_id][1]
        return mcm.gen_stick(spos=np.array([0, 0, -cylinder_half_length]),
                             epos=np.array([0, 0, cylinder_half_length]),
                             radius=cylinder_radius,
                             rgb=model.geom_rgba[geom_id][:3],
                             alpha=model.geom_rgba[geom_id][3])
    elif geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
        return mcm.gen_box(xyz_lengths=np.array([100, 100, 0.001]),
                           rotmat=np.eye(3),
                           rgb=model.geom_rgba[geom_id][:3],
                           alpha=model.geom_rgba[geom_id][3])
    elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        sphere_radius = model.geom_size[geom_id][0]
        return mcm.gen_sphere(radius=sphere_radius,
                              rgb=model.geom_rgba[geom_id][:3],
                              alpha=model.geom_rgba[geom_id][3])
    # elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
    #     ellipsoid_size = model.geom_size[geom_id]
    #     axes_mat = np.diag([ellipsoid_size[0], ellipsoid_size[1], ellipsoid_size[2]])
    #     return mcm.gen_ellipsoid(axes_mat=axes_mat, rgba=model.geom_rgba[geom_id])
    # elif geom_type == mujoco.mjtGeom.mjGEOM_LINE:
    #     line_size = model.geom_size[geom_id]
    #     return mcm.gen_linesegs(linesegs=[[-line_size, line_size]], rgba=model.geom_rgba[geom_id])
    # elif geom_type == mujoco.mjtGeom.mjGEOM_POINT:
    #     point_position = data.geom_xpos[geom_id]
    #     return mcm.gen_pointcloud(points=[point_position], rgba=model.geom_rgba[geom_id])


def cvt_bodies(model):
    """
    Convert the bodies of the MuJoCo model to a list of dictionaries.
    :param model: MjModel: The MuJoCo model.
    :return: list: The list of dictionaries.
    """
    body_geom_dict = {}
    # Get geoms associated with the mujoco body
    for body_id in range(model.nbody):
        body_geom_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body_id]
        geom_dict = {}
        for geom_id in body_geom_ids:
            result = cvt_geom(model, geom_id)
            if result is not None:
                geom_dict[geom_id] = result
        body_geom_dict[body_id] = geom_dict
    return body_geom_dict


class MJModel(object):

    def __init__(self, input_string):
        """
        Initialize the MuJoCo model.
        :param input_string: file_name or xml_string
        """
        if '<' in input_string and '>' in input_string:
            self.model = self._load_from_string(input_string)
        else:
            self.model = self._load_from_file(input_string)
        self.data = mujoco.MjData(self.model)
        self.body_geom_dict = cvt_bodies(self.model)
        self.control_callback = None
        self.desired_aj_pos = [0]*self.model.nu

    def _load_from_file(self, file_name):
        """
        Load the MuJoCo model from the given file.
        :param file_name: str: The file name of the MuJoCo model.
        :return: MjModel: The MuJoCo model.
        """
        return mujoco.MjModel.from_xml_path(file_name)

    def _load_from_string(self, xml_string):
        """
        Load the MuJoCo model from the given string.
        :param xml_string: str: The XML string of the MuJoCo model.
        :return: MjModel: The MuJoCo model.
        """
        return mujoco.MjModel.from_xml_string(xml_string)

    def attach_to(self, base):
        """
        Attach the MuJoCo model to the given base.
        :param base: ShowBase: The base to attach the MuJoCo model to.
        """
        base.mj_model = self


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[3, 3, 3], lookat_pos=[0, 0, .7])
    gm.gen_frame().attach_to(base)


    class PIDController:
        def __init__(self, kp, ki, kd, setpoint):
            self.kp = kp
            self.ki = ki
            self.kd = kd
            self.setpoint = setpoint
            self.integral = np.zeros_like(setpoint)
            self.previous_error = np.zeros_like(setpoint)

        def compute(self, measurement, dt):
            error = self.setpoint - measurement
            self.integral += error * dt
            derivative = (error - self.previous_error) / dt
            self.previous_error = error
            return self.kp * error + self.ki * self.integral + self.kd * derivative


    kp = 10.0  # Proportional gain
    ki = 0.5  # Integral gain
    kd = 2.0  # Derivative gain



    def control_callback(mj_model):
        kp = 1.5  # Proportional gain
        ki = 0.5  # Integral gain
        kd = 0.3  # Derivative gain

        class PIDController:
            def __init__(self, kp, ki, kd, setpoint):
                self.kp = kp
                self.ki = ki
                self.kd = kd
                self.setpoint = setpoint
                self.integral = np.zeros_like(setpoint)
                self.previous_error = np.zeros_like(setpoint)

            def compute(self, measurement, dt):
                error = self.setpoint - measurement
                self.integral += error * dt
                derivative = (error - self.previous_error) / dt
                self.previous_error = error
                return self.kp * error + self.ki * self.integral + self.kd * derivative

        for aid in range(mj_model.model.nu):
            jid = mj_model.model.actuator(aid).trnid[0]
            print(mj_model.model.joint(jid).name)
            current_position = mj_model.data.qpos[jid]
            position_error = mj_model.desired_aj_pos[aid] - current_position
            velocity_error = -mj_model.data.qvel[jid]
            mj_model.data.ctrl[aid] = kp * position_error + kd * velocity_error

            # Compute control input
            # control_input = PIDController(kp, ki, kd, mj_model.desired_aj_pos[aid]).compute(current_position, mj_model.model.opt.timestep)
            # mj_model.data.ctrl[aid] = control_input
            # mj_model.data.qpos[jid] = mj_model.desired_aj_pos[aid]
        print(mj_model.data.qpos)


    mj_model = MJModel("g1.xml")
    mj_model.control_callback = control_callback
    mujoco.mj_forward(mj_model.model, mj_model.data)

    for aid in range(mj_model.model.nu):
        jid = mj_model.model.actuator(aid).trnid[0]
        mj_model.desired_aj_pos[aid]=mj_model.data.qpos[jid]
    base.run_mj_physics(mj_model, 100)
    # for geom_dict in mj_model.body_geom_dict.values():
    #     for key, geom in geom_dict.items():
    #         pos = mj_model.data.geom_xpos[key]
    #         rotmat = mj_model.data.geom_xmat[key].reshape(3, 3)
    #         geom.pose = [pos, rotmat]
    #         geom.attach_to(base)

    base.run()
