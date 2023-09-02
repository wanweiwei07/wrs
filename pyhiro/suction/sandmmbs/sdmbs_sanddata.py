#!/usr/bin/env python3

import sys
sys.path.append("../../")  # TODO

import math
import os

import robotmath as rm
import pandactrl as pandactrl
import pandageom as pandageom
from direct.showbase.ShowBase import ShowBase
from panda3d.bullet import BulletDebugNode
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletTriangleMesh
from panda3d.bullet import BulletTriangleMeshShape
from panda3d.bullet import BulletWorld
from panda3d.core import *


class SdmbsSd():
    '''
    use utils.designpattern.singleton() to get a single instance of this class
    '''

    def __init__(self, hndcolor = Vec4(.5,.5,.5,.7)):
        '''
        load the sdmbs model and return a nodepath

        author: weiwei
        date: 20170313
        '''

        self.__sdmbs = NodePath("sdmbs")

        this_dir, this_filename = os.path.split(__file__)
        mbspath = Filename.fromOsSpecific(os.path.join(this_dir, "mbsegg", "mbs_sanddata.egg"))
        self.__mbsl = loader.loadModel(mbspath)
        self.__mbsl.instanceTo(self.__sdmbs)

        self.__sdmbs.setColor(hndcolor[0],hndcolor[1],hndcolor[2],hndcolor[3])
        self.__sdmbs.setTransparency(TransparencyAttrib.MAlpha)

    @property
    def handnp(self):
        # read-only property
        return self.__sdmbs

    def setColor(self, r, g, b, a):
        self.__sdmbs.setColor(r,g,b,a)

    def setPose(self, pgvec3):
        """
        set the pose of the hand
        changes self.rtq85np

        :param pgvec3 a panda3d vector
        :return:
        """

        self.__sdmbs.setPos(pgvec3)

    def setMat(self, pgrotmat4):
        """
        set the translation and rotation of a robotiq hand
        changes self.rtq85np

        :param pgrotmat4: follows panda3d, a LMatrix4f matrix
        :return: null

        date: 20170313
        author: weiwei
        """

        self.__sdmbs.setMat(pgrotmat4)

    def getMat(self):
        """
        get the rotation matrix of the hand

        :return: pgrotmat4: follows panda3d, a LMatrix4f matrix

        date: 20170313
        author: weiwei
        """

        return self.__sdmbs.getMat()

    def getHpr(self):
        """
        get the roll, pitch, yaw of the hand
        H angle is how the model rotates around the (0, 0, 1) axis,
        the P angle how much it rotates around the (1, 0, 0) axis,
        the R angle how much it rotates around the (0, 1, 0) axis

        date: 20170315
        author: weiwei
        """

        return self.__sdmbs.getHpr()


    def reparentTo(self, nodepath):
        """
        add to scene, follows panda3d

        :param nodepath: a panda3d nodepath
        :return: null

        date: 20170313
        author: weiwei
        """
        self.__sdmbs.reparentTo(nodepath)

    def removeNode(self):
        """

        :return:
        """

        self.__sdmbs.removeNode()

    def zAlong(self, x, y, z):
        """
        set the X axis of the hnd along [x,y,z]

        author: weiwei
        date: 20170313
        """

        self.__sdmbs.setMat(Mat4.identMat())
        self.__sdmbs.lookAt(x, y, z)
        rotmat4x = Mat4.rotateMat(-90, Vec3(1, 0, 0))
        self.__sdmbs.setMat(rotmat4x*self.__sdmbs.getMat())

    def attachTo(self, pntx, pnty, pntz, nrmlx, nrmly, nrmlz, rotangle = 0):
        """
        set the hand to suc pntx,pnty,pntz with x facing nrmalx, nrmaly, nrmalz

        :param: rotangle in degree, around x

        author: weiwei
        date: 20170315
        """

        self.zAlong(-nrmlx, -nrmly, -nrmlz)
        rotmat4z = Mat4.rotateMat(rotangle, Vec3(0, 0, 1))
        self.__sdmbs.setMat(rotmat4z*self.__sdmbs.getMat())
        self.setPose(Vec3(pntx, pnty, pntz))


    def clearRotmat(self):
        """
        set the X axis of the hnd along [x,y,z]

        author: weiwei
        date: 20170313
        """

        self.__sdmbs.setMat(Mat4.identMat())

def newHandNM(hndcolor = [1,0,0,.1]):
    return SdmbsSd(Vec4(hndcolor[0], hndcolor[1], hndcolor[2], hndcolor[3]))

if __name__=='__main__':

    import collisiondetection as cd
    import pandageom as pg

    def updateworld(world, task):
        world.doPhysics(globalClock.getDt())
        # result = base.world.contactTestPair(bcollidernp.node(), lftcollidernp.node())
        # result1 = base.world.contactTestPair(bcollidernp.node(), ilkcollidernp.node())
        # result2 = base.world.contactTestPair(lftcollidernp.node(), ilkcollidernp.node())
        # print(result)
        # print(result.getContacts())
        # print(result1)
        # print(result1.getContacts())
        # print(result2)
        # print(result2.getContacts())
        # for contact in result.getContacts():
        #     cp = contact.getManifoldPoint()
        #     print(cp.getLocalPointA())
        return task.cont

    base = pandactrl.World()
    # first hand
    sdmbs = SdmbsSd()
    sdmbs.attachTo(0,0,0,0,1,1,15)
    print(sdmbs.getHpr())
    # sdmbs.reparentTo(base.render)
    handbullnp = cd.genCollisionMeshNp(sdmbs.handnp)
    # second hand
    sdmbs1 = SdmbsSd()
    sdmbs1.reparentTo(base.render)
    hand1bullnp = cd.genCollisionMeshNp(sdmbs1.handnp)

    pg.plotAxisSelf(base.render, Vec3(0,0,0))

    bullcldrnp = base.render.attachNewNode("bulletcollider")
    base.world = BulletWorld()

    base.taskMgr.add(updateworld, "updateworld", extraArgs=[base.world], appendTask=True)
    result = base.world.contactTestPair(handbullnp, hand1bullnp)
    for contact in result.getContacts():
        cp = contact.getManifoldPoint()
        pg.plotSphere(base.render, pos=cp.getLocalPointA(), radius=1, rgba=Vec4(1,0,0,1))

    debugNode = BulletDebugNode('Debug')
    debugNode.showWireframe(True)
    debugNode.showConstraints(True)
    debugNode.showBoundingBoxes(False)
    debugNode.showNormals(False)
    debugNP = bullcldrnp.attachNewNode(debugNode)
    # debugNP.show()

    base.world.setDebugNode(debugNP.node())

    base.run()