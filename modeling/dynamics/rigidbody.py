import numpy as np
import utiltools.robotmath as rm
import trimesh.transformations as tf
from panda3d.core import *

class Rigidbody(object):
    def __init__(self, name = 'generalrbdname', mass = 1.0, pos = np.array([0,0,0]), com = np.array([0,0,0]),
                 rotmat = np.identity(3), inertiatensor = np.identity(3)):
        # note anglew must be in radian!
        # initialize a rigid body
        self.__name = name
        self.__mass = mass
        # inertiatensor and center of mass are described in local coordinate system
        self.__com = com
        self.__inertiatensor = inertiatensor
        # the following values are in world coordinate system
        self.__pos = pos
        self.__rotmat = rotmat
        self.__linearv = np.array([0,0,0])
        self.__dlinearv = np.array([0,0,0])
        self.__angularw = np.array([0,0,0])
        self.__dangularw = np.array([0,0,0])

    @property
    def mass(self):
        return self.__mass

    @property
    def com(self):
        return self.__com

    @property
    def inertiatensor(self):
        return self.__inertiatensor

    @property
    def pos(self):
        return self.__pos

    @pos.setter
    def pos(self, value):
        self.__pos = value

    @property
    def rotmat(self):
        return self.__rotmat

    @rotmat.setter
    def rotmat(self, value):
        self.__rotmat = value

    @property
    def linearv(self):
        return self.__linearv

    @linearv.setter
    def linearv(self, value):
        self.__linearv = value

    @property
    def angularw(self):
        return self.__angularw

    @angularw.setter
    def angularw(self, value):
        self.__angularw = value

    @property
    def dlinearv(self):
        return self.__dlinearv

    @dlinearv.setter
    def dlinearv(self, value):
        self.__dlinearv = value

    @property
    def dangularw(self):
        return self.__dangularw

    @dangularw.setter
    def dangularw(self, value):
        self.__dangularw = value

def genForce(rbd, dtime):
    gravity = 9800
    Df = 1.0
    Kf = 100.0

    globalcom = rbd.rotmat.dot(rbd.com)+rbd.pos
    force = np.array([0,0,-rbd.mass*gravity])
    torque = np.cross(globalcom, force)

    if rbd.pos[2] < 0.0:
        v = rbd.linearv + np.cross(rbd.angularw, rbd.pos)
        force_re = np.array([-Df*v[0], -Df*v[1], -Kf*rbd.pos[2]-Df*v[2]])
        force = force + force_re
        torque = torque + np.cross(rbd.pos, force_re)

    force = np.array([0.0,0.0,0.0])
    torque = np.array([0.0,0.0,0.0])
    return force, torque

def updateRbdPR(rbd, dtime):
    eps = 1e-6
    angularwvalue = np.linalg.norm(rbd.angularw)
    if angularwvalue < eps:
        rbd.pos = rbd.pos + dtime*rbd.linearv
        rbd.rotmat = rbd.rotmat
    else:
        theta = math.degrees(angularwvalue*dtime)
        waxis = rbd.angularw/angularwvalue
        vnormw = rbd.linearv/angularwvalue
        rotmat = rm.rodrigues(waxis, theta)
        rbd.pos = rotmat.dot(rbd.pos) + \
                  (np.identity(3)-rotmat).dot(np.cross(waxis, vnormw)) + \
                  waxis.dot(waxis.transpose())*vnormw*angularwvalue*dtime
        # print rbd.pos, theta
        rbd.rotmat = rotmat.dot(rbd.rotmat)

def doPhysics(rbd, force, torque, dtime):
    globalcom = rbd.rotmat.dot(rbd.com)+rbd.pos
    globalinertiatensor = rbd.rotmat.dot(rbd.inertiatensor).dot(rbd.rotmat.transpose())
    globalcom_hat = rm.hat(globalcom)
    # si = spatial inertia
    Isi00 = rbd.mass * np.eye(3)
    Isi01 = rbd.mass * globalcom_hat.transpose()
    Isi10 = rbd.mass * globalcom_hat
    Isi11 = rbd.mass * globalcom_hat.dot(globalcom_hat.transpose()) + globalinertiatensor
    Isi = np.bmat([[Isi00, Isi01], [Isi10, Isi11]])
    vw = np.bmat([rbd.linearv, rbd.angularw]).T
    pl = Isi*vw
    # print np.ravel(pl[0:3])
    # print np.ravel(pl[3:6])
    ft = np.bmat([force, torque]).T
    angularw_hat = rm.hat(rbd.angularw)
    linearv_hat = rm.hat(rbd.linearv)
    vwhat_mat = np.bmat([[angularw_hat, np.zeros((3,3))], [linearv_hat, angularw_hat]])
    dvw = Isi.I*(ft-vwhat_mat*Isi*vw)
    # print dvw
    rbd.dlinearv = np.ravel(dvw[0:3])
    rbd.dangularw = np.ravel(dvw[3:6])

    rbd.linearv = rbd.linearv + rbd.dlinearv * dtime
    rbd.angularw = rbd.angularw + rbd.dangularw * dtime

    return [np.ravel(pl[0:3]), np.ravel(pl[3:6])]

if __name__=="__main__":
    import os
    import math
    from panda3d.core import *
    import pandaplotutils.pandactrl as pc
    import environment.collisionmodel as cm

    base = pc.World(camp = [1000,0,0], lookatpos= [0, 0, 0])
    rbd = Rigidbody(mass = 10.0, pos = np.array([0,0,0.0]), com = [0.0,0.0,0.0],
                    rotmat = rm.rodrigues(np.array([1,0,0]), 1.8),
                    inertiatensor = np.array([[1732.0,0.0,0.0],[0.0,1732.0,0.0],[0.0,0.0,3393.0]]))
    rbd.linearv = np.array([0,0,0])
    rbd.angularw = np.array([0,0,50])

    model = cm.CollisionModel('./objects/bunnysim.meshes')
    model.reparentTo(base.render)
    model.setMat(base.pg.np4ToMat4(rm.homobuild(rbd.pos, rbd.rotmat)))

    # base = pc.World(camp=[3000, 0, 3000], lookatp=[0, 0, 0])
    #
    # rbd = Rigidbody(mass = 1.0, pos=np.array([0, 0, 0.0]), rotmat=np.identity(3),
    #                 inertiatensor=np.array([[80833.3, 0.0, 0.0], [0.0, 68333.3, 0.0], [0.0, 0.0, 14166.7]]))
    # rbd.linearv = np.array([0,0,0])
    # rbd.angularw = np.array([1, 1, 1])
    #
    # this_dir, this_filename = os.path.split(__file__)
    # model_filepath = Filename.fromOsSpecific(os.path.join(this_dir, "models", "box.egg"))
    # model = loader.loadModel(model_filepath)
    # model.reparentTo(base.render)

    rbdnp = []
    framenp = []
    def updateshow(rbd, rbdnp, framenp, task):
        for frame in framenp:
            frame.detachNode()

        dtime = 0.001
        skipframe = 15
        for i in range(skipframe):
            force, torque = genForce(rbd, dtime)
            [P, L] = doPhysics(rbd, force, torque, dtime)
            updateRbdPR(rbd, dtime)

        model.setMat(base.pg.np4ToMat4(rm.homobuild(rbd.pos, rbd.rotmat)))
        arrownp = base.pggen.plotArrow(base.render, epos = rbd.angularw*500.0, thickness = 15, rgba=[1,0.5,0.5,1])
        framenp.append(arrownp)

        return task.again

    taskMgr.add(updateshow, 'updateshow', extraArgs=[rbd, rbdnp, framenp], appendTask=True)
    base.run()