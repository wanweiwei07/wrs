# 20201124
# I am running physics for meshes. When using millimeter as the metrics, my frame rate reaches to 60fps,
# but when I change them to meters, the number lowers down to 1fps or less. I am wondering if I did a wrong option.

import modeling.collisionmodel as cm
import modeling.dynamics.bullet.bdbody as bbd
import visualization.panda.world as wd
import math
import basis.robot_math as rm
import numpy as np

base = wd.World(camp=np.array([2, 0, 2]), lookat_pos=np.array([0, 0, 0]), toggle_debug=True)
# PlaneD
plane = basics.objtrm.primitives.Box(box_extents=[1, 1, .1], box_center=[0, 0, -0.05])
planecm = cm.CollisionModel(plane)
# planenode = bch.genBulletCDMesh(planecm)
planenode = bbd.BDBody(planecm, dynamic=False)
planemat = np.eye(4)
planemat[:3, 3] = planemat[:3, 3] + np.array([0, 0, 0])
planenode.sethomomat(planemat)
planenode.setMass(0)
planenode.setRestitution(0)
planenode.setFriction(1)
base.physicsworld.attachRigidBody(planenode)
planecm.reparent_to(base.render)
planecm.sethomomat(planenode.gethomomat())
cm.gm.genframe().reparent_to(base.render)

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
node = bbd.BDBody(model, dynamic=True)
bulletnodelist = []
for i in range(10):
    # node = bch.genBulletCDMesh(model)
    # newnode = copy.deepcopy(node)
    newnode = node.copy()
    newnode.setmass(1)
    rot = rm.rotmat_from_axangle([0, 1, 0], -math.pi / 4)
    pos = np.array([0, 0, .1 + i * .3])
    newnode.sethomomat(rm.homomat_from_posrot(pos, rot))
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
# objmodellist = [model.copy(), model.copy(), model.copy()]
# def update(omlist, bnlist, plotlist, task):
#
#     for plotele in plotlist:
#         plotele.detach()
#
#     for i, bn in enumerate(bnlist):
#         print(i)
#         omlist[i].setcolor(np.array([.8, .6, .3, .5]))
#         omlist[i].sethomomat(bn.gethomomat())
#         omlist[i].reparent_to(base.render)
#         plotlist.append(omlist[i])
#     return task.cont

def update(objmodel, bnlist, plotlist, task):
    for plotele in plotlist:
        plotele.remove()

    for bn in bnlist:
        # print(bn.gethomomat())
        modelcopy = objmodel.copy()
        modelcopy.setcolor(np.array([.8, .6, .3, .5]))
        modelcopy.sethomomat(bn.gethomomat())
        modelcopy.reparent_to(base.render)
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


# def update_tool(planecm, planebn, bnlist, task):
#     for bn in bnlist:
#         if bn.isActive():
#             return task.cont
#     currentmat = planebn[0].gethomomat()
#     currentmat[:3,3] = currentmat[:3,3]+np.array([0,0,.015])
#     planebn[0].sethomomat(currentmat)
#     planecm.sethomomat(planebn[0].gethomomat())
#     # planebn[0].setLinearVelocity(Vec3(0,0,10))
#
#     return task.cont

plotlist = []
# taskMgr.add(update, 'update', extraArgs=[model, bulletnodelist, plotlist], appendTask=True)
# taskMgr.add(update_tool, 'update_tool', extraArgs=[planecm, [planenode], bulletnodelist], appendTask=True)
base.setFrameRateMeter(True)
base.run()
