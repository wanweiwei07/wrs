import mujoco
from wrs import basis as trm, modeling as mcm, modeling as gm
import numpy as np


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
        vert_start = model.mesh_vertadr[mesh_id]
        vert_count = model.mesh_vertnum[mesh_id]
        face_start = model.mesh_faceadr[mesh_id]
        face_count = model.mesh_facenum[mesh_id]
        vertices = model.mesh_vert[vert_start:(vert_start + vert_count)]
        faces = model.mesh_face[face_start:(face_start + face_count)]
        # face_normals = model.mesh_facenormal[face_start:(face_start + face_count)]
        if len(faces) == 0:
            return mcm.gen_box()
        if vert_count-1 != max(faces.flatten()):
            print("Warning: the vertices and faces are not consistent!")
            return mcm.gen_box()
        name_start = model.name_meshadr[mesh_id]
        name_end = model.names.find(b'\x00', name_start)
        mesh_name = model.names[name_start:name_end].decode('utf-8')
        return mcm.CollisionModel(initor=trm.Trimesh(vertices=vertices, faces=faces),
                                  name=mesh_name,
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
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[3, 3, 3], lookat_pos=[0, 0, .7])
    gm.gen_frame().attach_to(base)

    mj_model = MJModel("humanoid.xml")
    # print(mj_model.chains)
    base.run_mj_physics(mj_model, 0)
    # mujoco.mj_forward(mj_model.model, mj_model.data)
    # for geom_dict in mj_model.body_geom_dict.values():
    #     for key, geom in geom_dict.items():
    #         pos = mj_model.data.geom_xpos[key]
    #         rotmat = mj_model.data.geom_xmat[key].reshape(3, 3)
    #         geom.pose = [pos, rotmat]
    #         geom.attach_to(base)

    base.run()
