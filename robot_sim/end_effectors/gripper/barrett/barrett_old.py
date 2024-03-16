import os
import warnings
import math
import utiltools.robotmath as rm
import pandaplotutils.pandactrl as pandactrl
from panda3d.core import *
import numpy as np
from environment import collisionmodel as cm
import trimesh.primitives as tp
import copy as cp
import utiltools.thirdparty.p3dhelper as p3dh


class BH828X(object):

    def __init__(self, *args, **kwargs):
        """
        load the robotiq85 model, set ee_values

        :param args:
        :param kwargs:
            'ee_values' 0-85
            'ftsensoroffset' the offset for forcesensor
            'toggleframes' True, False

        author: weiwei
        date: 20160627, 20190518, 20190824osaka, 20200321osaka
        """

        if 'jawwidthopen' in kwargs:
            self.__jawwidthopen = kwargs['jawwidthopen']
        else:
            self.__jawwidthopen = 100
        if 'jawwidthclose' in kwargs:
            self.__jawwidthclose = kwargs['jawwidthclose']
        else:
            self.__jawwidthclose = 0
        if 'ee_values' in kwargs:
            self.__jawwidth = kwargs['ee_values']
        else:
            self.__jawwidth = self.__jawwidthopen
        if 'ftsensoroffset' in kwargs:
            self.__ftsensoroffset = kwargs['ftsensoroffset']
        if 'toggleframes' in kwargs:
            self.__toggleframes = kwargs['toggleframes']
        if 'hndbase' in kwargs:
            self.__hndbase = cp.deepcopy(kwargs['hndbase'])
            # for copy
            self.__hndbase_bk = cp.deepcopy(kwargs['hndbase'])
        if 'hndfingerprox' in kwargs:
            self.__finger1_prox = cp.deepcopy(kwargs['hndfingerprox'])
            self.__finger2_prox = cp.deepcopy(kwargs['hndfingerprox'])
            self.__finger3_prox = cp.deepcopy(kwargs['hndfingerprox'])
            # for copy
            self.__hndfingerprox_bk = cp.deepcopy(kwargs['hndfingerprox'])
        if 'hndfingermed' in kwargs:
            self.__finger1_med = cp.deepcopy(kwargs['hndfingermed'])
            self.__finger2_med = cp.deepcopy(kwargs['hndfingermed'])
            self.__finger3_med = cp.deepcopy(kwargs['hndfingermed'])
            # for copy
            self.__hndfingermed_bk = cp.deepcopy(kwargs['hndfingermed'])
        if 'hndfingerdist' in kwargs:
            self.__finger1_dist = cp.deepcopy(kwargs['hndfingerdist'])
            self.__finger2_dist = cp.deepcopy(kwargs['hndfingerdist'])
            self.__finger3_dist = cp.deepcopy(kwargs['hndfingerdist'])
            # for copy
            self.__hndfingerdist_bk = cp.deepcopy(kwargs['hndfingerdist'])

        self.__name = "barrett-bh8-282"
        self.__hndnp = NodePath(self.__name)

        self.__angle_open = self._compute_jawangle(self.__jawwidthopen)
        self.__angle_close = self._compute_jawangle(self.__jawwidthclose)
        print(self.__angle_open, self.__angle_close)

        baselength = 79.5
        fingerlength = 100

        # eetippos/eetiprot
        self.__eetip = np.array([0.0, 0.0, baselength + fingerlength]) + np.array(
            [0.0, 0.0, self.__ftsensoroffset])  # max axis_length 136

        # base
        # self.__hndbase.setrotmat(rm.rodrigues(np.array([0,0,1]), 180))
        self.__hndbase.setpos(np.array([0.0, 0.0, self.__ftsensoroffset]))

        # ftsensor
        if self.__ftsensoroffset > 0:
            self.__ftsensor = cm.CollisionModel(tp.Cylinder(height=self.__ftsensoroffset, radius=30), name="ftsensor")
            self.__ftsensor.setPos(0, 0, -self.__ftsensoroffset / 2)
            self.__ftsensor.reparentTo(self.__hndbase)
        else:
            self.__ftsensor = None

        # 1,2 are the index and middle fingers, 3 is the thumb
        self.__finger1_prox_pos = np.array([-25, 0, 41.5])
        self.__finger1_prox_rot = rm.rotmat_from_euler(0, 0, -90)
        self.__finger1_med_pos = np.array([50, 0, 33.9])
        self.__finger1_med_rot = rm.rotmat_from_euler(90, 0, 0)
        self.__finger1_dist_pos = np.array([69.94, 3, 0])
        self.__finger1_dist_rot = rm.rotmat_from_euler(0, 0, -3)

        self.__finger2_prox_pos = np.array([25, 0, 41.5])
        self.__finger2_prox_rot = rm.rotmat_from_euler(0, 0, -90)
        self.__finger2_med_pos = np.array([50, 0, 33.9])
        self.__finger2_med_rot = rm.rotmat_from_euler(90, 0, 0)
        self.__finger2_dist_pos = np.array([69.94, 3, 0])
        self.__finger2_dist_rot = rm.rotmat_from_euler(0, 0, -3)

        self.__finger3_med_pos = np.array([0, 50, 75.4])
        self.__finger3_med_rot = rm.rotmat_from_euler(90, 0, 90)
        self.__finger3_dist_pos = np.array([69.94, 3, 0])
        self.__finger3_dist_rot = rm.rotmat_from_euler(0, 0, -3)

        # controllable angles
        self.__angle_main = 0.0
        self.__angle_finger1 = 0.0
        self.__angle_finger2 = 0.0
        self.__angle_finger3 = 0.0

        # angle ranges
        self.__am_range = {"rngmin": 0, "rngmax": 180}
        self.__af1_range = {"rngmin": 0, "rngmax": 140}
        self.__af2_range = {"rngmin": 0, "rngmax": 140}
        self.__af3_range = {"rngmin": 0, "rngmax": 140}

        # update finger positions
        # finger 1
        finger1_prox_pos = self.__finger1_prox_pos
        finger1_prox_rot = np.dot(self.__finger1_prox_rot, rm.rodrigues(np.array([0, 0, 1]), -self.__angle_main))
        self.__finger1_prox.sethomomat(rm.homobuild(finger1_prox_pos, finger1_prox_rot))

        finger1_med_pos = finger1_prox_pos + np.dot(finger1_prox_rot, self.__finger1_med_pos)
        finger1_med_rot = np.dot(finger1_prox_rot, np.dot(self.__finger1_med_rot,
                                                          rm.rodrigues(np.array([0, 0, 1]), self.__angle_finger1)))
        self.__finger1_med.sethomomat(rm.homobuild(finger1_med_pos, finger1_med_rot))

        finger1_dist_pos = finger1_med_pos + np.dot(finger1_med_rot, self.__finger1_dist_pos)
        finger1_dist_rot = np.dot(finger1_med_rot, np.dot(self.__finger1_dist_rot,
                                                          rm.rodrigues(np.array([0, 0, 1]), self.__angle_finger1 / 3)))
        self.__finger1_dist.sethomomat(rm.homobuild(finger1_dist_pos, finger1_dist_rot))

        self.__finger1_prox.reparentTo(self.__hndbase)
        self.__finger1_med.reparentTo(self.__hndbase)
        self.__finger1_dist.reparentTo(self.__hndbase)

        # finger 2
        finger2_prox_pos = self.__finger2_prox_pos
        finger2_prox_rot = np.dot(self.__finger2_prox_rot, rm.rodrigues(np.array([0, 0, 1]), self.__angle_main))
        self.__finger2_prox.sethomomat(rm.homobuild(finger2_prox_pos, finger2_prox_rot))

        finger2_med_pos = finger2_prox_pos + np.dot(finger2_prox_rot, self.__finger2_med_pos)
        finger2_med_rot = np.dot(finger2_prox_rot, np.dot(self.__finger2_med_rot,
                                                          rm.rodrigues(np.array([0, 0, 1]), self.__angle_finger2)))
        self.__finger2_med.sethomomat(rm.homobuild(finger2_med_pos, finger2_med_rot))

        finger2_dist_pos = finger2_med_pos + np.dot(finger2_med_rot, self.__finger2_dist_pos)
        finger2_dist_rot = np.dot(finger2_med_rot, np.dot(self.__finger2_dist_rot,
                                                          rm.rodrigues(np.array([0, 0, 1]), self.__angle_finger2 / 3)))
        self.__finger2_dist.sethomomat(rm.homobuild(finger2_dist_pos, finger2_dist_rot))

        self.__finger2_prox.reparentTo(self.__hndbase)
        self.__finger2_med.reparentTo(self.__hndbase)
        self.__finger2_dist.reparentTo(self.__hndbase)

        # finger 3
        finger3_med_pos = self.__finger3_med_pos
        finger3_med_rot = np.dot(self.__finger3_med_rot, rm.rodrigues(np.array([0, 0, 1]), self.__angle_finger3))
        self.__finger3_med.sethomomat(rm.homobuild(finger3_med_pos, finger3_med_rot))

        finger3_dist_pos = finger3_med_pos + np.dot(finger3_med_rot, self.__finger3_dist_pos)
        finger3_dist_rot = np.dot(finger3_med_rot, np.dot(self.__finger3_dist_rot,
                                                          rm.rodrigues(np.array([0, 0, 1]), self.__angle_finger3 / 3)))
        self.__finger3_dist.sethomomat(rm.homobuild(finger3_dist_pos, finger3_dist_rot))
        #
        self.__finger3_med.reparentTo(self.__hndbase)
        self.__finger3_dist.reparentTo(self.__hndbase)

        self.__hndbase.reparentTo(self.__hndnp)
        # self.setjawwidth(self.__jawwidth)

        self.setDefaultColor()
        if self.__toggleframes:
            if self.__ftsensor is not None:
                self.__ftsensorframe = p3dh.genframe()
                self.__ftsensorframe.reparentTo(self.__hndnp)
            self.__hndframe = p3dh.genframe()
            self.__hndframe.reparentTo(self.__hndnp)
            self.__baseframe = p3dh.genframe()
            self.__baseframe.reparentTo(self.__hndbase.objnp)
            self.__finger1_prox_frame = p3dh.genframe()
            self.__finger1_prox_frame.reparentTo(self.__finger1_prox.objnp)
            self.__finger1_med_frame = p3dh.genframe()
            self.__finger1_med_frame.reparentTo(self.__finger1_med.objnp)
            self.__finger1_dist_frame = p3dh.genframe()
            self.__finger1_dist_frame.reparentTo(self.__finger1_dist.objnp)
            self.__finger2_prox_frame = p3dh.genframe()
            self.__finger2_prox_frame.reparentTo(self.__finger2_prox.objnp)
            self.__finger2_med_frame = p3dh.genframe()
            self.__finger2_med_frame.reparentTo(self.__finger2_med.objnp)
            self.__finger2_dist_frame = p3dh.genframe()
            self.__finger2_dist_frame.reparentTo(self.__finger2_dist.objnp)
            self.__finger3_med_frame = p3dh.genframe()
            self.__finger3_med_frame.reparentTo(self.__finger3_med.objnp)
            self.__finger3_dist_frame = p3dh.genframe()
            self.__finger3_dist_frame.reparentTo(self.__finger3_dist.objnp)

    @property
    def hndnp(self):
        # read-only property
        return self.__hndnp

    @property
    def jawwidthopen(self):
        # read-only property
        return self.__jawwidthopen

    @property
    def jawwidthclose(self):
        # read-only property
        return self.__jawwidthclose

    @property
    def jawwidth(self):
        # read-only property
        return self.__jawwidth

    @property
    def cmlist(self):
        # read-only property
        if self.__ftsensor is not None:
            return [self.__ftsensor, self.__hndbase, self.__rfinger, self.__lfinger, self.__rfingertip,
                    self.__lfingertip]
        else:
            return [self.__hndbase, self.__rfinger, self.__lfinger, self.__rfingertip, self.__lfingertip]

    @property
    def eetip(self):
        # read-only property
        return self.__eetip

    @property
    def name(self):
        # read-only property
        return self.__name

    def _compute_jawangle(self, jawwidth):
        """
        compute the angles for open and close in antipodal grasps
        note the computation is approximate, only prox fingers and angles are considered

        :param open:
        :return:

        author: weiwei
        date: 20200322
        """

        return math.degrees(math.pi / 2 - math.asin((jawwidth / 2 + 5) / 70))

    def setangles(self, angle_main, angle_finger1, angle_finger2, angle_finger3):
        """
        set the 4 motor angles
        angle_main = palm joint
        angle_finger1 = index
        angle_finger2 = middle
        angle_finger3 = thumb

        :param angle_main: degree
        :param angle_finger1:
        :param angle_finger2:
        :param angle_finger3:
        :return:

        author: weiwei
        date: 20200322
        """

        if angle_main < self.__am_range["rngmin"]:
            print("Warning. Range of angle_main must be larger than " + str(self.__am_range["rngmin"]))
            angle_main = self.__am_range["rngmin"]
        elif angle_main > self.__am_range["rngmax"]:
            print("Warning. Range of angle_main must be smaller than " + str(self.__am_range["rngmax"]))
            angle_main = self.__am_range["rngmax"]

        if angle_finger1 < self.__af1_range["rngmin"]:
            print("Warning. Range of angle_finger1 must be larger than " + str(self.__af1_range["rngmin"]))
            angle_finger1 = self.__af1_range["rngmin"]
        elif angle_finger1 > self.__af1_range["rngmax"]:
            print("Warning. Range of angle_finger1 must be smaller than " + str(self.__af1_range["rngmax"]))
            angle_finger1 = self.__af1_range["rngmax"]

        if angle_finger2 < self.__af2_range["rngmin"]:
            print("Warning. Range of angle_finger2 must be larger than " + str(self.__af2_range["rngmin"]))
            angle_finger2 = self.__af2_range["rngmin"]
        elif angle_finger2 > self.__af2_range["rngmax"]:
            print("Warning. Range of angle_finger2 must be smaller than " + str(self.__af2_range["rngmax"]))
            angle_finger2 = self.__af2_range["rngmax"]

        if angle_finger3 < self.__af3_range["rngmin"]:
            print("Warning. Range of angle_finger3 must be larger than " + str(self.__af3_range["rngmin"]))
            angle_finger3 = self.__af3_range["rngmin"]
        elif angle_finger3 > self.__af3_range["rngmax"]:
            print("Warning. Range of angle_finger3 must be smaller than " + str(self.__af3_range["rngmax"]))
            angle_finger3 = self.__af3_range["rngmax"]

        self.__am_range = {"rngmin": 0, "rngmax": 180}
        self.__angle_finger1 = {"rngmin": 0, "rngmax": 180}
        self.__angle_finger2 = {"rngmin": 0, "rngmax": 140}

        self.__angle_main = angle_main
        self.__angle_finger1 = angle_finger1
        self.__angle_finger2 = angle_finger2
        self.__angle_finger3 = angle_finger3

        # update finger positions
        # finger 1
        finger1_prox_pos = self.__finger1_prox_pos
        finger1_prox_rot = np.dot(self.__finger1_prox_rot, rm.rodrigues(np.array([0, 0, 1]), -self.__angle_main))
        self.__finger1_prox.sethomomat(rm.homobuild(finger1_prox_pos, finger1_prox_rot))

        finger1_med_pos = finger1_prox_pos + np.dot(finger1_prox_rot, self.__finger1_med_pos)
        finger1_med_rot = np.dot(finger1_prox_rot, np.dot(self.__finger1_med_rot,
                                                          rm.rodrigues(np.array([0, 0, 1]), self.__angle_finger1)))
        self.__finger1_med.sethomomat(rm.homobuild(finger1_med_pos, finger1_med_rot))

        finger1_dist_pos = finger1_med_pos + np.dot(finger1_med_rot, self.__finger1_dist_pos)
        finger1_dist_rot = np.dot(finger1_med_rot, np.dot(self.__finger1_dist_rot,
                                                          rm.rodrigues(np.array([0, 0, 1]), self.__angle_finger1 / 3)))
        self.__finger1_dist.sethomomat(rm.homobuild(finger1_dist_pos, finger1_dist_rot))

        self.__finger1_prox.reparentTo(self.__hndbase)
        self.__finger1_med.reparentTo(self.__hndbase)
        self.__finger1_dist.reparentTo(self.__hndbase)

        # finger 2
        finger2_prox_pos = self.__finger2_prox_pos
        finger2_prox_rot = np.dot(self.__finger2_prox_rot, rm.rodrigues(np.array([0, 0, 1]), self.__angle_main))
        self.__finger2_prox.sethomomat(rm.homobuild(finger2_prox_pos, finger2_prox_rot))

        finger2_med_pos = finger2_prox_pos + np.dot(finger2_prox_rot, self.__finger2_med_pos)
        finger2_med_rot = np.dot(finger2_prox_rot, np.dot(self.__finger2_med_rot,
                                                          rm.rodrigues(np.array([0, 0, 1]), self.__angle_finger2)))
        self.__finger2_med.sethomomat(rm.homobuild(finger2_med_pos, finger2_med_rot))

        finger2_dist_pos = finger2_med_pos + np.dot(finger2_med_rot, self.__finger2_dist_pos)
        finger2_dist_rot = np.dot(finger2_med_rot, np.dot(self.__finger2_dist_rot,
                                                          rm.rodrigues(np.array([0, 0, 1]), self.__angle_finger2 / 3)))
        self.__finger2_dist.sethomomat(rm.homobuild(finger2_dist_pos, finger2_dist_rot))

        self.__finger2_prox.reparentTo(self.__hndbase)
        self.__finger2_med.reparentTo(self.__hndbase)
        self.__finger2_dist.reparentTo(self.__hndbase)

        # finger 3
        finger3_med_pos = self.__finger3_med_pos
        finger3_med_rot = np.dot(self.__finger3_med_rot, rm.rodrigues(np.array([0, 0, 1]), self.__angle_finger3))
        self.__finger3_med.sethomomat(rm.homobuild(finger3_med_pos, finger3_med_rot))

        finger3_dist_pos = finger3_med_pos + np.dot(finger3_med_rot, self.__finger3_dist_pos)
        finger3_dist_rot = np.dot(finger3_med_rot, np.dot(self.__finger3_dist_rot,
                                                          rm.rodrigues(np.array([0, 0, 1]), self.__angle_finger3 / 3)))
        self.__finger3_dist.sethomomat(rm.homobuild(finger3_dist_pos, finger3_dist_rot))
        #
        self.__finger3_med.reparentTo(self.__hndbase)
        self.__finger3_dist.reparentTo(self.__hndbase)

        self.__hndbase.reparentTo(self.__hndnp)

    def setjawwidth(self, jawwidth=None):
        """
        set the ee_values of the hand

        :param jawwidth: mm
        :return:

        author: weiwei
        date: 20160627, 20190514
        """

        if jawwidth is None:
            jawwidth = self.__jawwidthopen
        else:
            if jawwidth >= self.__jawwidthopen + 1e-6:
                print(jawwidth, self.__jawwidthopen + 1e-6)
                print("Warning too large! Jawwidth must be in (" + str(self.__jawwidthclose) + "," + str(
                    self.__jawwidthopen) + "). The input is " + str(jawwidth) + ".")
                jawwidth = self.__jawwidthopen
            elif jawwidth <= self.__jawwidthclose - 1e-6:
                print("Warning too small! Jawwidth must be in (" + str(self.__jawwidthclose) + "," + str(
                    self.__jawwidthopen) + "). The input is " + str(jawwidth) + ".")
                jawwidth = self.__jawwidthclose

        # compute angles
        tmpangle = self._compute_jawangle(jawwidth)
        self.setangles(0, tmpangle, tmpangle, tmpangle)

        # update the private member variable
        self.__jawwidth = jawwidth

    def setPos(self, pandavec3):
        """
        set the pose of the hand
        changes self.__hndnp

        :param pandavec3 panda3d vec3
        :return:
        """

        warnings.warn("The functions with capitalized letters will be deleted in the future, use lowercase functions!",
                      DeprecationWarning)
        self.__hndnp.setPos(pandavec3)

    def getPos(self):
        """
        set the pose of the hand
        changes self.__hndnp

        :param pandavec3 panda3d vec3
        :return:
        """

        return self.__hndnp.getPos()

    def setMat(self, pandamat4):
        """
        set the translation and rotation of a robotiq hand
        changes self.__hndnp

        :param pandamat4 panda3d Mat4
        :return: null

        date: 20161109
        author: weiwei
        """

        self.__hndnp.setMat(pandamat4)

    def getMat(self):
        """
        get the rotation matrix of the hand

        :return: pandamat4 follows panda3d, a LMatrix4f matrix

        date: 20161109
        author: weiwei
        """

        return self.__hndnp.getMat()

    def sethomomat(self, homomat):
        """

        :param homomat: np.ndarray 4x4
        :return:

        author: weiwei
        date: 20191015, osaka
        """

        self.__hndnp.setMat(p3dh.npmat4_to_pdmat4(homomat))

    def gethomomat(self):
        """

        :return: pos: np.ndarray 4x4

        author: weiwei
        date: 20191015, osaka
        """

        return p3dh.npmat4(self.__hndnp.getMat())

    def reparentTo(self, nodepath):
        """
        add to scene, follows panda3d

        :param nodepath: a panda3d pdndp
        :return: null

        date: 20161109
        author: weiwei
        """
        self.__hndnp.reparentTo(nodepath)

    def removeNode(self):
        """

        :return:
        """

        self.__hndnp.removeNode()

    def __lookAt(self, direct0, direct1, direct2):
        """
        set the Y axis of the hnd
        ** deprecated 20190517

        author: weiwei
        date: 20161212
        """

        self.__hndnp.lookAt(direct0, direct1, direct2)

    def gripat(self, fcx, fcy, fcz, c0nx, c0ny, c0nz, rotangle=0, jawwidth=None):
        """
        set the hand to grip at fcx, fcy, fcz, fc = finger center
        the normal of the sglfgr contact is set to be c0nx, c0ny, c0nz
        the rotation around the normal is set to rotangle
        the ee_values is set to ee_values

        date: 20170322
        author: weiwei
        """

        # x is the opening motion_vec of the hand, _org means the value when rotangle=0
        standardx_org = np.array([1, 0, 0])
        newx_org = np.array([c0nx, c0ny, c0nz])
        rotangle_org = rm.degree_betweenvector(newx_org, standardx_org)
        rotaxis_org = np.array([0, 0, 1])
        if not (np.isclose(rotangle_org, 180.0) or np.isclose(rotangle_org, 0.0)):
            rotaxis_org = rm.unit_vector(np.cross(standardx_org, newx_org))
        newrotmat_org = rm.rodrigues(rotaxis_org, rotangle_org)

        # rotate to the given rotangle
        hnd_rotmat4 = np.eye(4)
        hnd_rotmat4[:3, :3] = np.dot(rm.rodrigues(newx_org, rotangle), newrotmat_org)
        handtipnpvec3 = np.array([fcx, fcy, fcz]) - np.dot(hnd_rotmat4[:3, :3], self.__eetip)
        hnd_rotmat4[:3, 3] = handtipnpvec3
        self.__hndnp.setMat(p3dh.npmat4_to_pdmat4(hnd_rotmat4))
        if jawwidth is None:
            jawwidth = self.__jawwidthopen
        self.setjawwidth(jawwidth)

        return [jawwidth, np.array([fcx, fcy, fcz]), hnd_rotmat4]

    def approachat(self, fcx, fcy, fcz, c0nx, c0ny, c0nz, apx, apy, apz, jawwidth=None):
        """
        set the hand to grip at fcx, fcy, fcz, fc = finger center
        the normal of the sglfgr contact is set to be c0nx, c0ny, c0nz
        the approach vector of the hand is set to apx, apy, apz
        the ee_values is set to ee_values

        date: 20190528
        author: weiwei
        """

        # x is the opening motion_vec of the hand, _org means the value when rotangle=0
        nphndmat3 = np.eye(3)
        nphndmat3[:3, 0] = rm.unit_vector(np.array([c0nx, c0ny, c0nz]))
        nphndmat3[:3, 2] = rm.unit_vector(np.array([apx, apy, apz]))
        nphndmat3[:3, 1] = rm.unit_vector(np.cross(np.array([apx, apy, apz]), np.array([c0nx, c0ny, c0nz])))

        # rotate to the given rotangle
        hnd_rotmat4 = np.eye(4)
        hnd_rotmat4[:3, :3] = nphndmat3
        handtipnpvec3 = np.array([fcx, fcy, fcz]) - np.dot(nphndmat3, self.__eetip)
        hnd_rotmat4[:3, 3] = handtipnpvec3
        self.__hndnp.setMat(p3dh.npmat4_to_pdmat4(hnd_rotmat4))
        if jawwidth is None:
            jawwidth = self.__jawwidthopen
        self.setjawwidth(jawwidth)

        return [jawwidth, np.array([fcx, fcy, fcz]), hnd_rotmat4]

    def genrangecmlist(self, jawwidthstart, jawwidthend=None, discretizedegree=10.0):
        '''
        generate a hand mcm model for collision.
        The finger motion is discretized in range (ee_values, jawopen)
        The generated model is an independent copy, the hand itsself is not modified

        ## input
        ee_values:
            the width of the jaw

        author: weiwei
        date: 20160627
        '''

        if jawwidthend is None:
            jawwidthend = self.__jawwidthopen

        if jawwidthstart >= self.__jawwidthopen + 1e-6 or jawwidthstart <= self.__jawwidthclose - 1e-6:
            print("Wrong value! Jawwidthstart must be in (" + str(self.__jawwidthclose) + "," + str(
                self.__jawwidthopen) + "). The input is " + str(jawwidthstart) + ".")
            raise Exception("Jawwidthstart out of range!")
        if jawwidthend >= self.__jawwidthopen + 1e-6 or jawwidthend <= self.__jawwidthclose - 1e-6:
            print("Wrong value! Jawwidthend must be in (" + str(self.__jawwidthclose) + "," + str(
                self.__jawwidthopen) + "). The input is " + str(jawwidthend) + ".")
            raise Exception("Jawwidthend out of range!")
        if jawwidthend != 0 and jawwidthend <= jawwidthstart:
            print("Wrong value! Jawwidthend must be larger than Jawwidthstart.")
            print(jawwidthstart, jawwidthend)
            raise Exception("Jawwidthend leq Jawwidthstart.")

        nls = np.ceil((jawwidthend - jawwidthstart) / discretizedegree)
        print(nls)
        cmlist = [self.__hndbase]
        for rangejw in np.linspace(jawwidthstart, jawwidthend, nls, endpoint=True):
            self.setjawwidth(rangejw)
            cmlist.append(self.copy().cmlist[-4])
            cmlist.append(self.copy().cmlist[-3])
            cmlist.append(self.copy().cmlist[-2])
            cmlist.append(self.copy().cmlist[-1])
        self.setjawwidth(jawwidthstart)

        return cmlist

    def setColor(self, *args, **kwargs):
        """

        :param rgba
        :return:

        author: weiwei
        date: 20190514
        """

        if self.__ftsensor is not None:
            self.__ftsensor.clearColor()
        self.__hndbase.clearColor()
        self.__rfinger.clearColor()
        self.__lfinger.clearColor()
        if len(args) == 1:
            self.__hndnp.setColor(args[0])
        else:
            # print(args[0], args[1], args[2], args[3])
            self.__hndnp.setColor(args[0], args[1], args[2], args[3])

    def setDefaultColor(self, rgba_base=[.7, .7, .7, 1], rgba_finger=[.37, .37, .37, 1]):
        """

        :param rgba_knucklefingertip:
        :param rgba_baseinnerknucklefinger:
        :return:

        author: weiwei
        date: 20190514
        """

        if self.__ftsensor is not None:
            self.__ftsensor.setColor(rgba_base[0], rgba_base[1], rgba_base[2], rgba_base[3])
        self.__hndbase.setColor(rgba_base[0], rgba_base[1], rgba_base[2], rgba_base[3])
        self.__finger1_prox.setColor(rgba_finger[0], rgba_finger[1], rgba_finger[2], rgba_finger[3])
        self.__finger1_med.setColor(rgba_finger[0], rgba_finger[1], rgba_finger[2], rgba_finger[3])
        self.__finger1_dist.setColor(rgba_finger[0], rgba_finger[1], rgba_finger[2], rgba_finger[3])
        self.__finger2_prox.setColor(rgba_finger[0], rgba_finger[1], rgba_finger[2], rgba_finger[3])
        self.__finger2_med.setColor(rgba_finger[0], rgba_finger[1], rgba_finger[2], rgba_finger[3])
        self.__finger2_dist.setColor(rgba_finger[0], rgba_finger[1], rgba_finger[2], rgba_finger[3])
        self.__finger3_med.setColor(rgba_finger[0], rgba_finger[1], rgba_finger[2], rgba_finger[3])
        self.__finger3_dist.setColor(rgba_finger[0], rgba_finger[1], rgba_finger[2], rgba_finger[3])

    def copy(self):
        """
        make a copy of oneself
        use this one to copy a hand since copy.deepcopy doesnt work well with the collisionnode

        :return:

        author: weiwei
        date: 20190525osaka
        """

        hand = BH828X(jawwidthopen=self.__jawwidthopen, jawwidthclose=self.__jawwidthclose,
                      ftsensoroffset=self.__ftsensoroffset, toggleframes=self.__toggleframes, hndbase=self.__hndbase_bk,
                      hndfingerprox=self.__hndfingerprox_bk, hndfingermed=self.__hndfingermed_bk,
                      hndfingerdist=self.__hndfingerdist_bk)
        hand.setMat(self.getMat())
        hand.setjawwidth(self.__jawwidth)
        return hand

    def showcn(self):
        for cm in self.cmlist:
            cm.showcn()
        # self.__hndbase.showcn()
        # self.__hndrfinger.showcn()
        # self.__hndlfinger.showcn()


class HandFactory(object):

    def __init__(self):
        this_dir, this_filename = os.path.split(__file__)
        self.__name = "barrett"
        # open and close is defined for antipodal grip (thumb vs index+middle)
        self.__jawwidthopen = 100
        self.__jawwidthclose = 0
        self.__base = cm.CollisionModel(objinit=this_dir + "/stl/palm.stl", betransparency=True)
        self.__fingerprox = cm.CollisionModel(objinit=this_dir + "/stl/finger_prox.stl", betransparency=True)
        self.__fingermed = cm.CollisionModel(objinit=this_dir + "/stl/finger_med.stl", betransparency=True)
        self.__fingerdist = cm.CollisionModel(objinit=this_dir + "/stl/finger_dist.stl", betransparency=True)

    @property
    def name(self):
        # read-only property
        return self.__name

    @property
    def jawwidthopen(self):
        # read-only property
        return self.__jawwidthopen

    @property
    def jawwidthclose(self):
        # read-only property
        return self.__jawwidthclose

    def genHand(self, ftsensoroffset=0, toggleframes=False):
        return BH828X(jawwidthopen=self.__jawwidthopen, jawwidthclose=self.__jawwidthclose,
                      ftsensoroffset=ftsensoroffset, toggleframes=toggleframes, hndbase=self.__base,
                      hndfingerprox=self.__fingerprox, hndfingermed=self.__fingermed, hndfingerdist=self.__fingerdist)


if __name__ == '__main__':
    import copy as cp
    import environment.bulletcdhelper as bcd

    base = pandactrl.World(lookatpos=[0, 0, 0])
    p3dh.genframe().reparentTo(base.render)

    yfa = HandFactory()
    yihnd = yfa.genHand(toggleframes=True)
    # yihnd.setangles(180, 140, 140, 140)
    yihnd.setjawwidth(0)
    yihnd.reparentTo(base.render)

    yihnd.gripat(100, 0, 0, 0, 1, 0, 30, 5)
    # yihnd.reparentTo(base.render)
    # yihnd.showcn()
    base.run()

    yihnd2 = yfa.genHand(toggleframes=True)
    # yihnd2.setjawwidth(yihnd2.jawwidthopen)
    # print(yihnd2.jawwidthopen)
    yihnd2.setjawwidth(0)
    yihnd2.gripat(100, 0, 0, 0, 1, 0, 30, 5)
    # yihnd2.setColor(1,0,0,.2)
    yihnd2.reparentTo(base.render)
    yihnd3 = yihnd2.copy()
    yihnd3.gripat(0, 0, 0, 0, 1, 0, 90, 33)
    yihnd3.reparentTo(base.render)
    yihnd3.showcn()
    #
    bmc = bcd.MCMchecker(toggledebug=True)
    print(bmc.isMeshListMeshListCollided(yihnd3.cmlist, yihnd2.genrangecmlist(10)))
    bmc.showMeshList(yihnd3.genrangecmlist(20))
    bmc.showMeshList(yihnd2.cmlist)
    bmc.showMeshList(yihnd3.cmlist)
    bmc.showBoxList(yihnd.cmlist)

    base.run()
