import wrs.visualization.panda.world as wd
import numpy as np
from panda3d.core import Vec3
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletBoxShape, BulletTriangleMeshShape
from panda3d.bullet import BulletTriangleMesh
from wrs import modeling as cm
import copy

base = wd.World(cam_pos=np.array([1000, 0, 3000]), lookat_pos=np.array([0, 0, 0]), toggle_debug=True)
# World
# world = BulletWorld()
# world.setGravity(Vec3(0, 0, -9810))
# mgm.gen_frame().attach_to(base)

# Plane
shape = BulletBoxShape(Vec3(500, 500, 100))
# shape = BulletPlaneShape(Vec3(0, 0, 1), 1)
node = BulletRigidBodyNode('Ground')
node.addShape(shape)
np = base.render.attachNewNode(node)
np.setPos(0, 0, 0)
base.physicsworld.attachRigidBody(node)

# Box
# shape = BulletBoxShape(Vec3(0.5, 0.5, 0.5))
# shape = BulletSphereShape(major_radius=.1)
# major_radius = 0.5
# height = 1.4
# shape = BulletCylinderShape(major_radius, height, ZUp)
# major_radius = 0.5
# height = 1.0
# shape = BulletCapsuleShape(major_radius, height, ZUp)
# major_radius = 0.6
# height = 1.0
# shape = BulletConeShape(major_radius, height, ZUp)
# shape = BulletConvexHullShape()
# shape.addPoint(Point3(1, 1, 2))
# shape.addPoint(Point3(0, 0, 0))
# shape.addPoint(Point3(2, 0, 0))
# shape.addPoint(Point3(0, 2, 0))
# shape.addPoint(Point3(2, 2, 0))
# p0 = Point3(-.1, -.1, .1)
# p1 = Point3(-.1, .1, 0)
# p2 = Point3(.1, -.1, .1)
# p3 = Point3(.1, .1, 0)
# mesh = BulletTriangleMesh()
# mesh.addTriangle(p0, p1, p2)
# mesh.addTriangle(p1, p2, p3)
# shape = BulletTriangleMeshShape(mesh, dynamic=False)
# shape1 = BulletBoxShape((1.3, 1.3, 0.2))
# shape2 = BulletBoxShape((0.1, 0.1, 0.5))
# shape3 = BulletBoxShape((0.1, 0.1, 0.5))
# shape4 = BulletBoxShape((0.1, 0.1, 0.5))
# node = BulletRigidBodyNode('Box')
# node.setMass(1.0)
# node.addShape(shape)
# node.addShape(shape1, TransformState.makePos(Point3(0, 0, 0.1)))
# node.addShape(shape2, TransformState.makePos(Point3(-1, -1, -0.5)))
# node.addShape(shape3, TransformState.makePos(Point3(-1, 1, -0.5)))
# node.addShape(shape4, TransformState.makePos(Point3(1, -1, -0.5)))
# node.addShape(shape5, TransformState.makePos(Point3(1, 1, -0.5)))
# np = base.render.attachNewNode(node)
# np.setPos(0, 0, 2)
# world.attachRigidBody(node)

bunny_cm = cm.CollisionModel("./objects/bunnysim_mm.stl")
bunny_geom_nodepath = bunny_cm.pdndp.getChild(0).find("+GeomNode")
geom = copy.deepcopy(bunny_geom_nodepath.node().getGeom(0))
geombmesh = BulletTriangleMesh()
geombmesh.addGeom(geom)
shape = BulletTriangleMeshShape(geombmesh, dynamic = True)
shape.setMargin(1e-6)

for i in range(15):
    node = BulletRigidBodyNode('bunny'+str(i))
    node.setMass(20)
    node.addShape(shape)
    np = base.render.attachNewNode(node)
    np.setPos(0, 0, 1000+i*300)
    base.physicsworld.attachRigidBody(node)

# debugNode = BulletDebugNode('Debug')
# debugNode.showWireframe(True)
# debugNode.showConstraints(True)
# debugNode.showBoundingBoxes(False)
# debugNode.showNormals(True)
# _debugNP = base.render.attachNewNode(debugNode)
# world.setDebugNode(debugNode)
# _debugNP.show()
#
# # Update
# def update(task):
#     dt = globalClock.getDt()
#     world.doPhysics(dt)
#     return task.cont
#
# taskMgr.add(update, 'update')
base.setFrameRateMeter(True)
base.run()