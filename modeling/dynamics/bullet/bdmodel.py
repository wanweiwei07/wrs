import copy
import math
import modeling.geometricmodel as gm
import modeling.dynamics.bullet.bdbody as bdb


class BDModel(object):
    """
    load an object as a bullet dynamics model
    author: weiwei
    date: 20190627
    """

    def __init__(self, objinit, mass=None, restitution=0, allowdeactivation=False, allowccd=True, friction=.2,
                 stationary=False, type="convex", name="bdm"):
        """
        :param objinit: GeometricModel (CollisionModel also work)
        :param mass:
        :param restitution:
        :param allowdeactivation:
        :param allowccd:
        :param friction:
        :param dynamic:
        :param type: "convex", "triangle"
        :param name:
        """
        if isinstance(objinit, BDModel):
            self._gm = copy.deepcopy(objinit.gm)
            self._bdb = objinit.bdb.copy()
        elif isinstance(objinit, gm.GeometricModel):
            if mass is None:
                mass = 0
            self._gm = objinit
            self._bdb = bdb.BDBody(self._gm, type, mass, restitution, allowdeactivation=allowdeactivation,
                                   allowccd=allowccd, friction=friction, stationary=stationary, name=name)
        else:
            if mass is None:
                mass = 0
            self._gm = gm.GeometricModel(objinit)
            self._bdb = bdb.BDBody(self._gm, type, mass, restitution, allowdeactivation=allowdeactivation,
                                   allowccd=allowccd, friction=friction, stationary=stationary, name=name)

    @property
    def gm(self):
        # read-only property
        return self._gm

    @property
    def bdb(self):
        # read-only property
        return self._bdb

    def setcolor(self, rgba):
        self._gm.setcolor(rgba)

    def clearcolor(self):
        self._gm.clearcolor()

    def getcolor(self):
        return self._gm.getcolor()

    def setpos(self, npvec3):
        homomat_bdb = self._bdb.gethomomat()
        homomat_bdb[:3, 3] = npvec3
        self._bdb.sethomomat(homomat_bdb)
        self._gm.sethomomat(homomat_bdb)

    def getpos(self):
        return self._bdb.getpos()

    def sethomomat(self, npmat4):
        self._bdb.sethomomat(npmat4)
        self._gm.sethomomat(npmat4)

    def gethomomat(self):
        return self._bdb.gethomomat()

    def setmass(self, mass):
        self._bdb.setmass(mass)

    def reparent_to(self, obj):
        """
        obj must be base.render
        :param obj:
        :return:
        author: weiwei
        date: 20190627
        """
        if obj is not base.render:
            raise ValueError("This bullet dynamics model doesnt support rendering to non base.render nodes!")
        else:
            self._gm.sethomomat(self.bdb.gethomomat()) # get updated with dynamics
            self._gm.reparent_to(obj)

    def remove(self):
        self._gm.remove()

    def detach(self):
        self._gm.detach()

    def startphysics(self):
        base.physicsworld.attach(self._bdb)

    def endphysics(self):
        base.physicsworld.remove(self._bdb)

    def showlocalframe(self):
        self._gm.showlocalframe()

    def unshowlocalframe(self):
        self._gm.unshowlocalframe()

    def copy(self):
        return BDModel(self)


if __name__ == "__main__":
    import os
    import numpy as np
    import basis.robot_math as rm
    import visualization.panda.world as wd
    import random

    base = wd.World(camp=[1, .3, 1], lookat_pos=[0, 0, 0], toggle_debug=False)
    base.setFrameRateMeter(True)

    this_dir, this_filename = os.path.split(__file__)
    objpath = os.path.join(this_dir, "objects", "block.meshes")
    bunnycm = BDModel(objpath, mass=1, type="convex")

    objpath2 = os.path.join(this_dir, "objects", "bowlblock.meshes")
    bunnycm2 = BDModel(objpath2, mass=0, type="triangle", stationary=True)
    bunnycm2.setcolor(np.array([0, 0.7, 0.7, 1.0]))
    bunnycm2.setpos(np.array([0, 0, 0]))
    base.attach_autoupdate_object(bunnycm2)

    def update(bunnycm, task):
        if base.inputmgr.keymap['space'] is True:
            for i in range(100):
                bunnycm1 = bunnycm.copy()
                bunnycm1.setmass(.1)
                rndcolor = np.random.rand(4)
                rndcolor[-1] = 1
                bunnycm1.setcolor(rndcolor)
                rotmat = rm.rotmat_from_euler(0, 0, math.pi/12)
                z = math.floor(i / 100)
                y = math.floor((i - z * 100) / 10)
                x = i - z * 100 - y * 10
                print(x, y, z, "\n")
                bunnycm1.sethomomat(rm.homomat_from_posrot(np.array([x * 0.015 - 0.07, y * 0.015 - 0.07, 0.15 + z * 0.015]), rotmat))
                base.attach_autoupdate_object(bunnycm1)
                bunnycm1.startphysics()
        base.inputmgr.keymap['space'] = False
        return task.cont

    gm.genframe().reparent_to(base.render)
    taskMgr.add(update, "addobject", extraArgs=[bunnycm], appendTask=True)

    base.run()
