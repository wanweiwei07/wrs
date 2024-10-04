from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletTriangleMesh
from panda3d.bullet import BulletTriangleMeshShape
from panda3d.bullet import BulletDebugNode
from panda3d.bullet import BulletRigidBodyNode
from panda3d.core import *
import pandaplotutils.pandactrl as pc
import pandaplotutils.pandageom as pg
import os
import environment.collisionmodel as cm
import environment.bulletcdhelper as bch

base = pc.World(camp = [3000,0,3000], lookatpos= [0, 0, 0])
base.pggen.plotAxis(base.render)

# shape = BulletBoxShape(Vec3(50, 200, 450))
# node = BulletRigidBodyNode('Box')
# node.setMass(1.0)
# node.setAngularVelocity(Vec3(1,1,1))
# node.addShape(shape)

# np = base.render.attachNewNode(node)
# np.setPos(0, 0, 0)

model = cm.CollisionModel("./objects/bunnysim.meshes")
# model.reparentTo(base.render)
# model.setMat(Mat4.rotateMat(10, Vec3(1,0,0)))
# model.setPos(0,0,300)

bulletnode = bch.genBulletCDMesh(model)
bulletnode.setMass(1)
rigidbody = base.render.attachNewNode(bulletnode)
model.reparentTo(rigidbody)

world = BulletWorld()
world.setGravity(Vec3(0, 0, -9.8))
world.attach(bulletnode)

def update(task):
    dt = globalClock.getDt()
    world.doPhysics(dt)
    # vecw= topbullnode.getAngularVelocity()
    # arrownp = pg.plotArrow(base.render, epos = vecw*1000, major_radius = 15)
    # print rotmat
    # model.setMat(base.pg.np4ToMat4(rm.homobuild(rbd.pos, rbd.rotmat)))

    return task.cont

taskMgr.add(update, 'update')
base.run()