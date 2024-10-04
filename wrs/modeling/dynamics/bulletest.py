import environment.collisionmodel as cm
# import environment.bulletcdhelper as bch
import wrs.modeling.dynamics.bullet.bdbody as bbd
import pandaplotutils.pandactrl as pc
import copy
import trimesh
import utiltools.robotmath as rm
import numpy as np

base = pc.World(camp = [5000,0,5000], lookatpos= [0, 0, 0], toggledebug=True)

# PlaneD
plane = trimesh.primitives.Box(box_extents=[1000, 1000, 100], box_center=[0,0,-50])
planecm = cm.CollisionModel(plane)
# planenode = bch.genBulletCDMesh(planecm)
planenode = bbd.BDTriangleBody(planecm, dynamic=True)
planemat = np.eye(4)
planemat[:3,3] = planemat[:3,3]+np.array([0,0,0])
planenode.set_homomat(planemat)
planenode.setMass(0)
planenode.setRestitution(0)
planenode.setFriction(1)
base.physicsworld.attachRigidBody(planenode)
planecm.reparentTo(base.render)
planecm.setMat(base.pg.np4ToMat4(planenode.get_homomat()))
base.pggen.plotAxis(base.render)

# for i in range(5):
#     print("....")
#     currentmat = planenode.gethomomat()
#     print(currentmat)
#     currentmat[:3, 3] = currentmat[:3, 3] + np.array([0, 0, .0001])
#     planenode.sethomomat(currentmat)
#     print(currentmat)
#     planecm.setMat(base.pg.np4ToMat4(planenode.gethomomat()))
#     print(planenode.gethomomat())

# Boxes
# model = loader.loadModel('models/box.egg')
model = cm.CollisionModel("./objects/bunnysim.meshes")
node = bbd.BDTriangleBody(model, dynamic=True)
bulletnodelist = []
for i in range(3):
    # node = bch.genBulletCDMesh(model)
    # newnode = copy.deepcopy(node)
    newnode = node.copy()
    newnode.setMass(1)
    rot = rm.rodrigues([0,1,0],-45)
    pos = np.array([0,0,100+i*300])
    newnode.set_homomat(rm.homobuild(pos, rot))
    print(newnode.get_homomat())
    newnode.set_homomat(rm.homobuild(pos, rot))
    print(newnode.get_homomat())
    base.physicsworld.attachRigidBody(newnode)
    bulletnodelist.append(newnode)
    # modelcopy = copy.deepcopy(model)
    # modelcopy.setColor(.8, .6, .3, .5)
    # modelcopy.reparentTo(np)

# debugNode = BulletDebugNode('Debug')
# debugNode.showWireframe(True)
# debugNode.showConstraints(True)
# debugNode.showBoundingBoxes(False)
# debugNode.showNormals(False)
# debugNP = base.render.attachNewNode(debugNode)
# # debugNP.show()
# world.setDebugNode(debugNP.node())

# Update
def update(objmodel, bnlist, plotlist, task):

    for plotele in plotlist:
        plotele.removeNode()

    for bn in bnlist:
        # print(bn.gethomomat())
        modelcopy = copy.deepcopy(objmodel)
        modelcopy.setColor(.8, .6, .3, .5)
        modelcopy.setMat(base.pg.np4ToMat4(bn.get_homomat()))
        modelcopy.reparentTo(base.render)
        plotlist.append(modelcopy)

    # for bn in bnlist:
    #     if bn.isActive():
    #         return task.cont
    # currentmat = planenode[0].gethomomat()
    # print(currentmat)
    # currentmat[:3,3] = currentmat[:3,3]+np.array([0,0,.0001])
    # planenode[0].sethomomat(currentmat)
    # print(currentmat)
    # planecm.setMat(base.pg.np4ToMat4(planenode[0].gethomomat()))
    # print(planenode[0].gethomomat())
    # print("....")
    return task.cont

def update_tool(planecm, planebn, bnlist, task):
    for bn in bnlist:
        if bn.isActive():
            return task.cont
    currentmat = planebn[0].get_homomat()
    currentmat[:3,3] = currentmat[:3,3]+np.array([0,0,15])
    planebn[0].set_homomat(currentmat)
    planecm.setMat(base.pg.np4ToMat4(planebn[0].get_homomat()))
    # planebn[0].setLinearVelocity(Vec3(0,0,10))

    return task.cont

plotlist = []
taskMgr.add(update, 'update', extraArgs=[model, bulletnodelist, plotlist], appendTask=True)
taskMgr.add(update_tool, 'update_tool', extraArgs=[planecm, [planenode], bulletnodelist], appendTask=True)
base.run()