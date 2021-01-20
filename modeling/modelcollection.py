import basis.robot_math as rm
import modeling._ode_cdhelper as mcd

class ModelCollection(object):
    """
    a helper class to further hide pandanodes
    list of collision and geom models can be added to this collection for visualization
    author: weiwei
    date: 201900825, 20201212
    """

    def __init__(self, name='modelcollection'):
        self._name = name
        self._gm_list = []
        self._cm_list = []

    @property
    def name(self):
        return self._name

    @property
    def cm_list(self):
        return self._cm_list

    @property
    def gm_list(self):
        return self._gm_list

    @property
    def cdmesh(self):
        vertices = []
        vertex_normals = []
        faces = []
        for objcm in self._cm_list:
            if objcm.cdmesh_type == 'aabb':
                objtrm = objcm.objtrm.bounding_box
            elif objcm.cdmesh_type == 'obb':
                objtrm = objcm.objtrm.bounding_box_oriented
            elif objcm.cdmesh_type == 'convexhull':
                objtrm = objcm.objtrm.convex_hull
            elif objcm.cdmesh_type == 'triangles':
                objtrm = objcm.objtrm
            homomat = objcm.get_homomat()
            vertices += rm.homomat_transform_points(homomat, objtrm.vertices)
            vertex_normals += rm.homomat_transform_points(homomat, objtrm.vertex_normals)
            faces += (objtrm.faces+len(faces))
        return mcd.gen_cdmesh_vvnf(vertices, vertex_normals, faces)

    @property
    def cdmesh_list(self):
        return [objcm.cdmesh for objcm in self._cm_list]

    def add_cm(self, objcm):
        self._cm_list.append(objcm)

    def remove_cm(self, objcm):
        self._cm_list.remove(objcm)

    def add_gm(self, objcm):
        self._gm_list.append(objcm)

    def remove_gm(self, objcm):
        self._gm_list.remove(objcm)

    def attach_to(self, obj):
        # TODO check if obj is ShowBase
        for cm in self._cm_list:
            cm.attach_to(obj)
        for gm in self._gm_list:
            gm.attach_to(obj)

    def detach(self):
        for cm in self._cm_list:
            cm.detach()
        for gm in self._gm_list:
            gm.detach()

    def show_cdprimit(self): # only work for cm
        for cm in self._cm_list:
            cm.show_cdprimit()

    def unshow_cdprimit(self): # only work for cm
        for cm in self._cm_list:
            cm.unshow_cdprimit()

    def show_cdmesh(self):
        for objcm in self._cm_list:
            objcm.show_cdmesh()

    def unshow_cdmesh(self):
        for objcm in self._cm_list:
            objcm.unshow_cdmesh()