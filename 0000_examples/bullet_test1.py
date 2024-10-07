# 20201124
# I am running physics for meshes. When using millimeter as the metrics, my frame rate reaches to 60fps,
# but when I change them to meters, the number lowers down to 1fps or less. I am wondering if I did a wrong option.

from wrs import wd, rm, mcm
import wrs.modeling.dynamics.bullet.bdbody as bbd

base = wd.World(cam_pos=rm.vec(2, 0, 2), lookat_pos=rm.vec(0, 0, 0), toggle_debug=False)
# PlaneD
homomat = rm.np.eye(4)
homomat[:3, 3] = rm.np.array([0, 0, -.05])
planecm = mcm.gen_box(xyz_lengths=[1, 1, .1], homomat=homomat)
# planenode = bch.genBulletCDMesh(planecm)
planenode = bbd.BDBody(planecm, cdtype='convex', dynamic=False)
planemat = rm.np.eye(4)
planemat[:3, 3] = planemat[:3, 3] + rm.np.array([0, 0, 0])
planenode.set_homomat(planemat)
planenode.setMass(0)
planenode.setRestitution(0)
planenode.setFriction(1)
base.physicsworld.attachRigidBody(planenode)
planecm.attach_to(base)
planecm.set_homomat(planenode.get_homomat())
cm.gm.gen_frame().attach_to(base)

# for i in range(5):
#     print("....")
#     currentmat = planenode.gethomomat()
#     print(currentmat)
#     currentmat[:3, 3] = currentmat[:3, 3] + rm.np.array([0, 0, .0001])
#     planenode.sethomomat(currentmat)
#     print(currentmat)
#     planecm.setMat(base.pg.np4ToMat4(planenode.gethomomat()))
#     print(planenode.gethomomat())

# Boxes
# model = loader.loadModel('models/box.egg')
model = mcm.CollisionModel("./objects/bunnysim.stl")
node = bbd.BDBody(model, cdtype='box', dynamic=True)
bulletnodelist = []
for i in range(10):
    # state = bch.genBulletCDMesh(model)
    # newnode = copy.deepcopy(state)
    newnode = node.copy()
    newnode.setMass(1)
    rot = rm.rotmat_from_axangle([0, 1, 0], -math.pi / 4)
    pos = rm.np.array([0, 0, .1 + i * .3])
    newnode.set_homomat(rm.homomat_from_posrot(pos, rot))
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
# world.setDebugNode(debugNP.state())

# Update
# objmodellist = [model.copy(), model.copy(), model.copy()]
# def update(omlist, bnlist, plotlist, task):
#
#     for plotele in plotlist:
#         plotele.detach()
#
#     for i, bn in enumerate(bnlist):
#         print(i)
#         omlist[i].setcolor(rm.np.array([.8, .6, .3, .5]))
#         omlist[i].sethomomat(bn.gethomomat())
#         omlist[i].reparent_to(base.render)
#         plotlist.append(omlist[i])
#     return task.cont

def update(objmodel, bnlist, plotlist, task):
    for plotele in plotlist:
        plotele.remove()
    for bn in bnlist:
        # print(bn.get_pos())
        modelcopy = objmodel.copy()
        modelcopy.set_rgba(rm.np.array([.8, .6, .3, .5]))
        modelcopy.set_homomat(bn.get_homomat())
        modelcopy.attach_to(base)
        plotlist.append(modelcopy)
    return task.cont
#
#     # for bn in bnlist:
#     #     if bn.isActive():
#     #         return task.cont
#     # currentmat = planenode[0].gethomomat()
#     # print(currentmat)
#     # currentmat[:3,3] = currentmat[:3,3]+rm.np.array([0,0,.0001])
#     # planenode[0].sethomomat(currentmat)
#     # print(currentmat)
#     # planecm.setMat(base.pg.np4ToMat4(planenode[0].gethomomat()))
#     # print(planenode[0].gethomomat())
#     # print("....")
#     return task.cont


# def update_tool(planecm, planebn, bnlist, task):
#     for bn in bnlist:
#         if bn.isActive():
#             return task.cont
#     currentmat = planebn[0].gethomomat()
#     currentmat[:3,3] = currentmat[:3,3]+rm.np.array([0,0,.015])
#     planebn[0].sethomomat(currentmat)
#     planecm.sethomomat(planebn[0].gethomomat())
#     # planebn[0].setLinearVelocity(Vec3(0,0,10))
#
#     return task.cont

plotlist = []
taskMgr.add(update, 'update', extraArgs=[model, bulletnodelist, plotlist], appendTask=True)
# taskMgr.add(update_tool, 'update_tool', extraArgs=[planecm, [planenode], bulletnodelist], appendTask=True)
base.setFrameRateMeter(True)
base.run()
