import modeling._mcdhelper as mcd

class ModelCollection(object):
    """
    a helper class to further hide pandanodes
    list of collision and geom models can be added to this collection for visualization
    author: weiwei
    date: 201900825, 20201212
    """

    def __init__(self, name="modelcollection"):
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

    def show_cdmesh(self, type='triangles'):
        if type == 'triangles':
            self._bullnode = mcd.show_triangles_cdmesh(self._cm_list)
        elif type == 'box':
            self._bullnode = mcd.show_box_cdmesh(self._cm_list)
        else:
            raise NotImplementedError('The requested '+type+' type cdmesh is not supported!')

    def unshow_cdmesh(self):
        if hasattr(self, '_bullnode'):
            mcd.unshow(self._bullnode)