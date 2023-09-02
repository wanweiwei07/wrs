#!/usr/bin/env python3

import numpy as np
from panda3d.core import *
from direct.showbase.DirectObject import DirectObject

import pyhiro.pandageom as pg


class InputManager(DirectObject):

    def __init__(self, pandabase, lookatp):
        self.pandabase = pandabase
        self.lookatp = Vec3(lookatp[0], lookatp[1], lookatp[2])
        self.initviewdist = self.pandabase.cam.getPos().length()
        self.lastm1pos = None
        self.lastm2pos = None
        self.rotatecenternp = None
        self.keyMap = {
            "mouse1": False,
            "mouse2": False,
            "wheel_up": False,
            "wheel_down": False,
            "space": False}
        self.accept(
            "mouse1",
            self.__setKey,
            ["mouse1", True])
        self.accept(
            "mouse1-up",
            self.__setKey,
            ["mouse1", False])
        self.accept(
            "mouse2",
            self.__setKey,
            ["mouse2", True])
        self.accept(
            "mouse2-up",
            self.__setKey,
            ["mouse2", False])
        self.accept(
            "wheel_up",
            self.__setKey,
            ["wheel_up", True])
        self.accept(
            "wheel_down",
            self.__setKey,
            ["wheel_down", True])
        self.accept(
            "space",
            self.__setKey,
            ["space", True])
        self.accept(
            "space-up",
            self.__setKey,
            ["space", False])
        self.setupMouseAim()

    def __setKey(self, key, value):
        self.keyMap[key] = value
        return

    def setupMouseAim(self):
        """
        set up collision rays, spheres, and planes for mouse manipulation

        :return: None

        author: weiwei
        date: 20161110
        """

        # create a collision ray and set its bitmask to 8
        # the collision ray must be a subnode of cam since we will
        # transform the clicked point (in the view of the cam)
        # to the world coordinate system
        # using the ray
        self.CN = CollisionNode("RayCN")
        self.cRay = CollisionRay()
        self.CN.addSolid(self.cRay)
        self.CN.setFromCollideMask(BitMask32.bit(8))
        self.CN.setIntoCollideMask(BitMask32.allOff())
        self.CN = self.pandabase.cam.attachNewNode(self.CN)

        camdist = self.pandabase.cam.getPos().length()
        # create an inverted collision sphere
        # and puts it into a collision node
        # its bitmask is set to 8,
        # and it will be the only collidable object at bit 8
        # the collision node is attached to the render
        # so that it will NOT move with the camera
        self.aimSphereCN = CollisionNode("aimSphereCN")
        self.aimSphere = CollisionSphere(
            self.lookatp[0], self.lookatp[1], self.lookatp[2], camdist*.6)
        self.aimSphereCN.addSolid(self.aimSphere)
        self.aimSphereCN.setFromCollideMask(BitMask32.allOff())
        self.aimSphereCN.setIntoCollideMask(BitMask32.bit(8))
        self.aimSphereCN = self.pandabase.render.attachNewNode(
            self.aimSphereCN)
        # self.aimSphereCN.show()

        # This creates a collision plane
        self.aimPlaneCN = CollisionNode("aimPlaneCN")
        self.aimPlane = CollisionPlane(Plane(Vec3(0, 0, 1), self.lookatp))
        self.aimPlaneCN.addSolid(self.aimPlane)
        self.aimPlaneCN.setFromCollideMask(BitMask32.allOff())
        self.aimPlaneCN.setIntoCollideMask(BitMask32.bit(8))
        self.aimPlaneCN = self.pandabase.render.attachNewNode(
            self.aimPlaneCN)
        # self.aimPlaneCN.show()

        # creates a traverser to do collision testing
        self.cTrav = CollisionTraverser()

        # creates a queue type handler to receive the collision event info
        self.cHanQ = CollisionHandlerQueue()

        # register the ray as a collider with the traverser,
        # and register the handler queue as
        # the handler to be used for the collisions.
        self.cTrav.addCollider(self.CN, self.cHanQ)

    def changeCollisionSphere(self, center=np.array([0, 0, 0]), radius=500.0):
        self.aimSphere.setCenter(center[0], center[1], center[2])
        self.aimSphere.setRadius(radius)

    def getMouse1Aim(self):
        """
        Get the position of mouse1 (clicked) using collision detection
        between a sphere and a ray

        :return: Vec3 or None

        author: weiwei
        date: 20161110
        """
        if self.pandabase.mouseWatcherNode.hasMouse():
            if self.keyMap['mouse1']:
                # get the mouse position in the window
                mpos = self.pandabase.mouseWatcherNode.getMouse()
                # sets the ray's origin at the camera and
                # directs it to shoot through the mouse cursor
                self.cRay.setFromLens(
                    self.pandabase.cam.node(), mpos.getX(), mpos.getY())
                # performs the collision checking pass
                self.cTrav.traverse(self.aimSphereCN)
                # Sort the handler entries from nearest to farthest
                self.cHanQ.sortEntries()

                if (self.cHanQ.getNumEntries() > 0):
                    entry = self.cHanQ.getEntry(0)
                    colPoint = entry.getSurfacePoint(self.pandabase.render)
                    return (colPoint)
        return None

    def checkMouse1Drag(self):
        curm1pos = self.getMouse1Aim()
        if curm1pos is None:
            if self.lastm1pos is not None:
                self.lastm1pos = None
            return
        if self.lastm1pos is None:
            # first time click
            self.lastm1pos = curm1pos
            return
        curm1vec = Vec3(curm1pos-self.lookatp)
        lastm1vec = Vec3(self.lastm1pos-self.lookatp)
        curm1vec.normalize()
        lastm1vec.normalize()
        rotatevec = curm1vec.cross(lastm1vec)
        rotateangle = curm1vec.signedAngleDeg(lastm1vec, rotatevec)
        if rotateangle > .02 or rotateangle < -.02:
            rotateangle = rotateangle * 15
            rotmat = Mat4(self.pandabase.cam.getMat())
            posvec = Vec3(self.pandabase.cam.getPos())
            rotmat.setRow(3, Vec3(0, 0, 0))
            self.pandabase.cam.setMat(
                rotmat*Mat4.rotateMat(rotateangle, rotatevec))
            self.pandabase.cam.setPos(
                Mat3.rotateMat(rotateangle, rotatevec).
                xform(posvec - self.lookatp) + self.lookatp)
            self.lastm1pos = self.getMouse1Aim()

    def getMouse2Aim(self):
        if self.pandabase.mouseWatcherNode.hasMouse():
            if self.keyMap['mouse2']:
                mpos = self.pandabase.mouseWatcherNode.getMouse()
                self.cRay.setFromLens(
                    self.pandabase.cam.node(), mpos.getX(), mpos.getY())
                self.cTrav.traverse(self.aimPlaneCN)
                self.cHanQ.sortEntries()

                if (self.cHanQ.getNumEntries() > 0):
                    entry = self.cHanQ.getEntry(0)
                    colPoint = entry.getSurfacePoint(self.pandabase.render)
                    return (colPoint)
        return None

    def checkMouse2Drag(self):
        curm2pos = self.getMouse2Aim()
        if curm2pos is None:
            if self.lastm2pos is not None:
                self.lastm2pos = None
            return
        if self.lastm2pos is None:
            # first time click
            self.lastm2pos = curm2pos
            return
        relm2vec = curm2pos - self.lastm2pos
        if relm2vec.length() > 5:
            tmplookatp = self.lookatp-relm2vec
            if (tmplookatp[0] > -1500 and tmplookatp[0] < 1500) and \
               (tmplookatp[1] > -1500 and tmplookatp[1] < 1500) and \
               (tmplookatp[2] > -1500 and tmplookatp[2] < 1500):
                self.lookatp = tmplookatp
                self.pandabase.cam.setPos(self.pandabase.cam.getPos()-relm2vec)
                self.pandabase.cam.lookAt(self.lookatp)
                self.last2mpos = self.getMouse2Aim()
                if self.rotatecenternp is not None:
                    self.rotatecenternp.detachNode()
                self.rotatecenternp = pg.plotDumbbell(
                    self.pandabase.render,
                    np.array([self.lookatp[0],
                              self.lookatp[1],
                              self.lookatp[2]]),
                    np.array([self.lookatp[0],
                              self.lookatp[1],
                              self.lookatp[2]]),
                    thickness=10,
                    rgba=np.array([1, 1, 0, 0]),
                    plotname="transcenter")

    def checkMouseWheel(self):
        if self.keyMap["wheel_up"] is True:
            self.keyMap["wheel_up"] = False
            forward = self.pandabase.cam.getPos() - self.lookatp
            forward.normalize()
            if (self.pandabase.cam.getPos() - self.lookatp).length() < \
               self.initviewdist * 20:
                newpos = self.pandabase.cam.getPos() + forward * 100
                self.pandabase.cam.setPos(newpos[0], newpos[1], newpos[2])
                self.changeCollisionSphere(
                    self.aimSphere.getCenter(),
                    self.aimSphere.getRadius() + 50)
                # self.pandabase.cam.lookAt(
                #    self.lookatp[0], self.lookatp[1], self.lookatp[2])
        if self.keyMap["wheel_down"] is True:
            self.keyMap["wheel_down"] = False
            forward = self.pandabase.cam.getPos()-self.lookatp
            forward.normalize()
            if (self.pandabase.cam.getPos() - self.lookatp).length() > \
               self.initviewdist * .05:
                newpos = self.pandabase.cam.getPos() - forward * 100
                self.pandabase.cam.setPos(newpos[0], newpos[1], newpos[2])
                camdist = self.pandabase.cam.getPos().length()
                self.changeCollisionSphere(
                    self.aimSphere.getCenter(), camdist * .6)
                # self.changeCollisionSphere(
                #    self.aimSphere.getCenter(),
                #    self.aimSphere.getRadius() - 50)
                # self.pandabase.cam.lookAt(
                #    self.lookatp[0], self.lookatp[1], self.lookatp[2])
