#!/usr/bin/env python3

import sys
sys.path.append("../../")  # noqa # TODO

import os
import math

from panda3d.core import *
from panda3d.bullet import (BulletWorld,
                            BulletDebugNode,
                            BulletRigidBodyNode,
                            BulletTriangleMesh,
                            BulletTriangleMeshShape)

import pyhiro.pandactrl as pandactrl
import pyhiro.pandageom as pandageom
from direct.showbase.ShowBase import ShowBase

import pyhiro.robotmath as rm


class SGB30():
    '''
    use utils.designpattern.singleton() to get a single instance of this class
    '''

    def __init__(self, hndcolor=Vec4(.5, .5, .5, .7)):
        '''
        load the sgb30 model and return a nodepath
        '''

        self.__sgb = NodePath("sgb")

        this_dir, this_filename = os.path.split(__file__)
        self.sgbpath = Filename.fromOsSpecific(
            os.path.join(this_dir, "sgb30mm.obj"))
        # self.sgbpath = Filename.fromOsSpecific(
        #    os.path.join(this_dir, "sgb30mm.egg"))
        self.__sgbl = loader.loadModel(self.sgbpath)
        self.__sgbl.instanceTo(self.__sgb)

        self.__sgb.setColor(
            hndcolor[0],
            hndcolor[1],
            hndcolor[2],
            hndcolor[3])
        self.__sgb.setTransparency(TransparencyAttrib.MAlpha)

    @property
    def handnp(self):
        # read-only property
        return self.__sgb

    def setColor(self, r, g, b, a):
        self.__sgb.setColor(r, g, b, a)

    def setPos(self, pgvec3):
        """
        set the pose of the hand
        changes self.rtq85np

        :param pgvec3 a panda3d vector
        :return:
        """

        self.__sgb.setPos(pgvec3)

    def getPos(self):
        """
        get the pose of the hand

        :return:npvec3
        """

        return self.__sgb.getPos()

    def setMat(self, pgrotmat4):
        """
        set the translation and rotation of a robotiq hand
        changes self.rtq85np

        :param pgrotmat4: follows panda3d, a LMatrix4f matrix
        :return: null
        """

        self.__sgb.setMat(pgrotmat4)

    def getHandName(self):
        return "sgb"

    def getMat(self):
        """
        get the rotation matrix of the hand

        :return: pgrotmat4: follows panda3d, a LMatrix4f matrix
        """

        return self.__sgb.getMat()

    def getHpr(self):
        """
        get the roll, pitch, yaw of the hand
        H angle is how the model rotates around the (0, 0, 1) axis,
        the P angle how much it rotates around the (1, 0, 0) axis,
        the R angle how much it rotates around the (0, 1, 0) axis
        """

        return self.__sgb.getHpr()

    def reparentTo(self, nodepath):
        """
        add to scene, follows panda3d

        :param nodepath: a panda3d nodepath
        :return: null
        """
        self.__sgb.reparentTo(nodepath)

    def removeNode(self):
        """

        :return:
        """

        self.__sgb.removeNode()

    def xAlong(self, x, y, z):
        """
        set the X axis of the hnd along [x,y,z]
        """

        self.__sgb.setMat(Mat4.identMat())
        self.__sgb.lookAt(x, y, z)
        rotmat4z = Mat4.rotateMat(90, Vec3(0, 0, 1))
        self.__sgb.setMat(rotmat4z*self.__sgb.getMat())

    def attachTo(self, pntx, pnty, pntz, nrmlx, nrmly, nrmlz, rotangle=0):
        """
        set the hand to suc pntx,pnty,pntz with x facing nrmalx, nrmaly, nrmalz

        :param: rotangle in degree, around x
        """

        self.__sgb.setMat(Mat4.identMat())
        self.__sgb.lookAt(nrmlx, nrmly, nrmlz)
        rotmat4x = Mat4.rotateMat(rotangle, Vec3(1, 0, 0))
        rotmat4y = Mat4.rotateMat(180, Vec3(0, 1, 0))
        rotmat4z = Mat4.rotateMat(90, Vec3(0, 0, 1))
        self.__sgb.setMat(rotmat4x*rotmat4y*rotmat4z*self.__sgb.getMat())
        rotmat4 = Mat4(self.__sgb.getMat())
        handtipvec3 = rotmat4.getRow3(0)*-5
        rotmat4.setRow(3, Vec3(pntx, pnty, pntz)+handtipvec3)
        self.__sgb.setMat(rotmat4)

    def clearRotmat(self):
        """
        set the X axis of the hnd along [x,y,z]
        """

        self.__sgb.setMat(Mat4.identMat())


def newHandNM(hndid='rgt', hndcolor=[1, 0, 0, .1]):
    return SGB30(Vec4(hndcolor[0], hndcolor[1], hndcolor[2], hndcolor[3]))


def newHand(hndid='rgt', hndcolor=[.5, .5, .5, 1]):
    return SGB30(Vec4(hndcolor[0], hndcolor[1], hndcolor[2], hndcolor[3]))


if __name__ == '__main__':

    import time
    import pyhiro.collisiondetection as cd
    import pyhiro.pandageom as pg

    base = pandactrl.World(w=800, h=600)
    base.taskMgr.step()

    # first hand
    sgb = SGB30()
    sgb.reparentTo(base.render)
    handbullnp = cd.genCollisionMeshNp(sgb.handnp)
    base.taskMgr.step()
    time.sleep(2)

    # second hand
    sgb1 = SGB30()
    sgb1.attachTo(0, 0, 0, 0, 0, 0, 0)
    sgb1.reparentTo(base.render)
    hand1bullnp = cd.genCollisionMeshNp(sgb1.handnp)
    base.taskMgr.step()
    time.sleep(2)

    pg.plotAxisSelf(base.render, Vec3(0, 0, 0))
    base.taskMgr.step()
    time.sleep(2)

    base.run()
