import numpy as np
import basis.robot_math as rm
from visualization.panda.world import ShowBase


class ModelCollection(object):
    """
    a helper class to further hide pandanodes
    list of collision and geom models can be added to this collection for visualization
    author: weiwei
    date: 201900825, 20201212
    """

    def __init__(self,
                 name='model_collection'):
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

    def set_rgba(self, rgba):
        for cm in self._cm_list:
            cm.set_rgba(rgba)
        for gm in self._gm_list:
            gm.set_rgba(rgba)

    def set_alpha(self, alpha):
        for cm in self._cm_list:
            cm.set_alpha(alpha)
        for gm in self._gm_list:
            gm.set_alpha(alpha)

    def add_cm(self, objcm):
        self._cm_list.append(objcm)

    def remove_cm(self, objcm):
        self._cm_list.remove(objcm)

    def add_gm(self, objcm):
        self._gm_list.append(objcm)

    def remove_gm(self, objcm):
        self._gm_list.remove(objcm)

    def attach_to(self, target):
        if isinstance(target, ShowBase):
            for cm in self._cm_list:
                cm.attach_to(target)
            for gm in self._gm_list:
                gm.attach_to(target)
        elif isinstance(target, ModelCollection):
            for cm in self._cm_list:
                target.add_cm(cm)
            for gm in self._gm_list:
                target.add_gm(gm)
        else:
            raise ValueError("Acceptable: ShowBase, ModelCollection!")

    def detach(self):
        for cm in self._cm_list:
            cm.detach()
        for gm in self._gm_list:
            gm.detach()

    def show_cdprimitive(self):  # only work for mcm
        for cm in self._cm_list:
            cm.show_cdprimitive()

    def unshow_cdprimitive(self):  # only work for mcm
        for cm in self._cm_list:
            cm.unshow_cdprimitive()

    def show_cdmesh(self):
        for objcm in self._cm_list:
            objcm.show_cdmesh()

    def unshow_cdmesh(self):
        for objcm in self._cm_list:
            objcm.unshow_cdmesh()
