#!/usr/bin/env python3

import os
import math
import numpy as np

from panda3d.core import *
from direct.task import Task
from direct.showbase.ShowBase import ShowBase
from direct.filter.CommonFilters import CommonFilters

import pyhiro.pandageom as pg
import pyhiro.inputmanager as im


class World(ShowBase, object):

    def __init__(
            self,
            camp=[2000, 500, 2000],
            lookatp=[0, 0, 250],
            up=[0, 0, 1],
            fov=40,
            w=2000,
            h=1500):
        """
        :param camp:
        :param lookatp:
        :param fov:
        :param w: width of window
        :param h: height of window
        """

        super(self.__class__, self).__init__()
        self.setBackgroundColor(1, 1, 1)

        # set up cartoon effect
        self.__separation = 1
        self.filters = CommonFilters(self.win, self.cam)
        self.filters.setCartoonInk(separation=self.__separation)

        # set up lens
        lens = PerspectiveLens()
        lens.setFov(fov)
        lens.setNearFar(1, 50000)
        self.disableMouse()
        self.cam.setPos(camp[0], camp[1], camp[2])
        self.cam.lookAt(
            Point3(lookatp[0], lookatp[1], lookatp[2]),
            Vec3(up[0], up[1], up[2]))
        self.cam.node().setLens(lens)

        # set up slight
        ablight = AmbientLight("ambientlight")
        ablight.setColor(Vec4(0.2, 0.2, 0.2, 1))
        ablightnode = self.cam.attachNewNode(ablight)
        self.render.setLight(ablightnode)

        ptlight0 = PointLight("pointlight1")
        ptlight0.setColor(VBase4(1, 1, 1, 1))
        ptlightnode0 = self.cam.attachNewNode(ptlight0)
        ptlightnode0.setPos(0, 0, 0)
        self.render.setLight(ptlightnode0)

        ptlight1 = PointLight("pointlight1")
        ptlight1.setColor(VBase4(.4, .4, .4, 1))
        ptlightnode1 = self.cam.attachNewNode(ptlight1)
        ptlightnode1.setPos(
            self.cam.getPos().length(), 0, self.cam.getPos().length())
        self.render.setLight(ptlightnode1)

        ptlight2 = PointLight("pointlight2")
        ptlight2.setColor(VBase4(.3, .3, .3, 1))
        ptlightnode2 = self.cam.attachNewNode(ptlight2)
        ptlightnode2.setPos(
            -self.cam.getPos().length(), 0, self.cam.getPos().length())
        self.render.setLight(ptlightnode2)

        # set up inputmanager
        self.inputmgr = im.InputManager(self, lookatp)
        taskMgr.add(self.cycleUpdate, "cycle update")

        # set up rotational cam
        # self.lookatp = lookatp
        # taskMgr.doMethodLater(.1, self.rotateCam, "rotate cam")

        # set window size
        props = WindowProperties()
        props.setSize(w, h)
        self.win.requestProperties(props)

    def cycleUpdate(self, task):
        # reset aspect ratio
        aspectRatio = self.getAspectRatio()
        self.cam.node().getLens().setAspectRatio(aspectRatio)
        self.inputmgr.checkMouse1Drag()
        self.inputmgr.checkMouse2Drag()
        self.inputmgr.checkMouseWheel()
        return task.cont

    def rotateCam(self, task):
        campos = self.cam.getPos()
        camangle = math.atan2(campos[1], campos[0])
        # print camangle
        if camangle < 0:
            camangle += math.pi*2
        if camangle >= math.pi*2:
            camangle = 0
        else:
            camangle += math.pi/180
        camradius = math.sqrt(campos[0]*campos[0]+campos[1]*campos[1])
        camx = camradius*math.cos(camangle)
        camy = camradius*math.sin(camangle)
        self.cam.setPos(camx, camy, campos[2])
        self.cam.lookAt(self.lookatp[0], self.lookatp[1], self.lookatp[2])
        return task.cont

    # def changeLookAt(self, lookatp):
    #     self.cam.lookAt(lookatp[0], lookatp[1], lookatp[2])
    #     self.inputmgr = im.InputManager(base, lookatp)
