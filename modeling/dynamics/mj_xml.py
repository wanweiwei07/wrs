import mujoco
import basis.trimesh as trm
import modeling.collision_model as mcm
import modeling.geometric_model as mgm
import numpy as np
import basis.robot_math as rm


def cvt_geom(model, data, geom_id):
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
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        mesh = model.mesh[model.geom_dataid[geom_id]]
        vertices = mesh.vert.reshape(-1, 3)
        faces = mesh.face.reshape(-1, 3)
        return mcm.CollisionModel(initor=trm.Trimesh(vertices=vertices, faces=faces),
                                  name=geom_name,
                                  rgb=model.geom_rgba[geom_id][:3],
                                  alpha=model.geom_rgba[geom_id][3])
    elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        box_size = model.geom_size[geom_id]
        return mcm.gen_box(xyz_lengths=2 * box_size, rgb=model.geom_rgba[geom_id][:3],
                           alpha=model.geom_rgba[geom_id][3])
    elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        capsule_radius = model.geom_size[geom_id][0]
        capsule_half_length = model.geom_size[geom_id][1]
        capsule_pos = model.geom_pos[geom_id]
        capsule_quat = model.geom_quat[geom_id]
        # local_from = np.array([0, 0, -capsule_half_length])
        # from_point = np.empty(3, np.float64)
        # mujoco.mju_rotVecQuat(from_point, local_from, capsule_quat)
        # from_point += capsule_pos
        # local_to = np.array([0, 0, capsule_half_length])
        # to_point = np.empty(3, np.float64)
        # mujoco.mju_rotVecQuat(to_point, local_to, capsule_quat)
        # to_point += capsule_pos
        # return mcm.gen_stick(spos=from_point,
        #                      epos=to_point,
        #                      radius=capsule_radius,
        #                      type="round",
        #                      rgb=model.geom_rgba[geom_id][:3],
        #                      alpha=model.geom_rgba[geom_id][3],
        #                      n_sec=12)
        return mcm.gen_stick(spos=np.array([capsule_pos[0], capsule_pos[1], capsule_pos[2] - capsule_half_length]),
                             epos=np.array([capsule_pos[0], capsule_pos[1], capsule_pos[2] + capsule_half_length]),
                             radius=capsule_radius,
                             type="round",
                             rgb=model.geom_rgba[geom_id][:3],
                             alpha=model.geom_rgba[geom_id][3],
                             n_sec=12)
    elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        cylinder_radius = model.geom_size[geom_id][0]
        cylinder_half_length = model.geom_size[geom_id][1]
        capsule_pos = model.geom_pos[geom_id]
        capsule_quat = model.geom_quat[geom_id]
        local_from = np.array([0, 0, cylinder_half_length])
        from_point = np.empty(3, np.float64)
        mujoco.mju_rotVecQuat(from_point, local_from, capsule_quat)
        from_point += capsule_pos
        local_to = np.array([0, 0, -cylinder_half_length])
        to_point = np.empty(3, np.float64)
        mujoco.mju_rotVecQuat(to_point, local_to, capsule_quat)
        to_point += capsule_pos
        return mcm.gen_stick(spos=from_point,
                             epos=to_point,
                             radius=cylinder_radius,
                             rgb=model.geom_rgba[geom_id][:3],
                             alpha=model.geom_rgba[geom_id][3])
    elif geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
        geom_mat = rm.rotmat_from_quaternion(model.geom_quat[geom_id])
        plane_normal = geom_mat[:, 2]
        plane_size = model.geom_size[geom_id]
        angle = np.arccos(np.dot(np.array([0, 0, 1]), plane_normal))
        axis = np.cross(np.array([0, 0, 1]), plane_normal)
        rotmat = rm.rotmat_from_axangle(axis=axis, angle=angle)
        return mcm.gen_box(xyz_lengths=np.array([plane_size[0] * 2, plane_size[1] * 2, 0.001]),
                           rotmat=rotmat,
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


def cvt_bodies(model, data):
    """
    Convert the bodies of the MuJoCo model to a list of dictionaries.
    :param model: MjModel: The MuJoCo model.
    :param data: MjData: The MuJoCo data.
    :return: list: The list of dictionaries.
    """
    body_geom_dict = {}
    # Get geoms associated with the mujoco body
    for body_id in range(model.nbody):
        body_geom_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body_id]
        geom_dict = {}
        for geom_id in body_geom_ids:
            geom_dict[geom_id] = cvt_geom(model, data, geom_id)
            mgm.gen_frame(ax_length=1).attach_to(geom_dict[geom_id])
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
        self.body_geom_dict = cvt_bodies(self.model, self.data)

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
    xml = """
    <mujoco model="denso_cobotta">
        <compiler angle="degree" coordinate="local"/>
        <option timestep="0.01" gravity="0 0 -1" iterations="50" integrator="Euler"/>

        <default>
            <joint limited="true" damping="1"/>
            <geom type="capsule" size="0.05" rgba="0.8 0.6 0.4 1"/>
        </default>

        <worldbody>
            <light diffuse="1 1 1" specular="0.1 0.1 0.1" pos="0 0 3"/>
            <geom name="floor" type="plane" pos="0 0 0" size="10 10 0.1" rgba="0.8 0.9 0.8 1"/>

            <!-- Base -->
            <body name="base" pos="0 0 0">
                <geom type="box" size="0.1 0.1 0.1" rgba="0.5 0.5 0.5 1"/>

                <!-- Link 1 -->
                <body name="link1" pos="0 0 0.1">
                    <joint name="joint1" type="hinge" axis="0 0 1" range="-180 180"/>
                    <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.05"/>

                    <!-- Link 2 -->
                    <body name="link2" pos="0 0 0.2">
                        <joint name="joint2" type="hinge" axis="0 1 0" range="-90 90"/>
                        <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.05"/>

                        <!-- Link 3 -->
                        <body name="link3" pos="0 0 0.2">
                            <joint name="joint3" type="hinge" axis="1 0 0" range="-180 180"/>
                            <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.05"/>

                            <!-- Link 4 -->
                            <body name="link4" pos="0 0 0.2">
                                <joint name="joint4" type="hinge" axis="0 1 0" range="-180 180"/>
                                <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.05"/>

                                <!-- Link 5 -->
                                <body name="link5" pos="0 0 0.2">
                                    <joint name="joint5" type="hinge" axis="1 0 0" range="-180 180"/>
                                    <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.05"/>

                                    <!-- Link 6 -->
                                    <body name="link6" pos="0 0 0.2">
                                        <joint name="joint6" type="hinge" axis="0 1 0" range="-180 180"/>
                                        <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.05"/>

                                        <!-- End Effector -->
                                        <body name="end_effector" pos="0 0 0.2">
                                            <geom type="sphere" size="0.05" rgba="1 0 0 1"/>
                                        </body>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </worldbody>
    </mujoco>
    """

    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[3, 3, 3], lookat_pos=[0, 0, 1])
    gm.gen_frame().attach_to(base)

    mj_model = MJModel(xml)
    base.run_mj_physics(mj_model, 7)

    # for geom_dict in mj_model.body_geom_dict.values():
    #     for values in geom_dict.values():
    #         values.attach_to(base)

    base.run()
