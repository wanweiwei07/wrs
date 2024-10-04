from panda3d.ode import OdeWorld, OdeSimpleSpace, OdeJointGroup
from panda3d.ode import OdeBody, OdeMass, OdeBoxGeom, OdePlaneGeom
from panda3d.core import BitMask32, Vec4
import wrs.visualization.panda.world as wd
from random import randint, random
import math
from wrs import basis as rm, basis as da, modeling as cm
import numpy as np


base = wd.World(cam_pos=[15, 15, 15], lookat_pos=[0, 0, 0], toggle_debug=True)

world = OdeWorld()
world.setGravity(0, 0, -9.81)
world.setQuickStepNumIterations(100)
world.setErp(.2)
world.setCfm(1e-3)

# The surface table is needed for autoCollide
world.initSurfaceTable(1)
world.setSurfaceEntry(0, 0, 150, 0.0, 9.1, 0.9, 0.00001, 0.0, 0.002)

# Create a space and add a contactgroup to it to add the contact joints
space = OdeSimpleSpace()
space.setAutoCollideWorld(world)
contactgroup = OdeJointGroup()
space.setAutoCollideJointGroup(contactgroup)

box = cm.gen_box(xyz_lengths=[.3, .3, .3])

# Add a random amount of boxes
boxes = []
for i in range(randint(5, 10)):
    # Setup the geometry
    new_box = box.copy()
    new_box.set_pos(np.array([random()*10-5, random()*10-5, 1 + random()]))
    new_box.set_rgba([random(), random(), random(), 1])
    new_box.set_rotmat(rm.rotmat_from_euler(random()*math.pi/4, random()*math.pi/4, random()*math.pi/4))
    new_box.attach_to(base)
    # Create the body and set the mass
    boxBody = OdeBody(world)
    M = OdeMass()
    M.setBox(3, .3, .3, .3)
    boxBody.setMass(M)
    boxBody.setPosition(da.npvec3_to_pdvec3(new_box.get_pos()))
    boxBody.setQuaternion(da.npmat3_to_pdquat(new_box.get_rotmat()))
    # Create a BoxGeom
    boxGeom = OdeBoxGeom(space, .3, .3, .3)
    # boxGeom = OdeTriMeshGeom(space, OdeTriMeshData(new_box.objpdnp, True))
    boxGeom.setCollideBits(BitMask32(0x00000002))
    boxGeom.setCategoryBits(BitMask32(0x00000001))
    boxGeom.setBody(boxBody)
    boxes.append((new_box, boxBody))

# Add a plane to collide with
ground = cm.gen_box(xyz_lengths=[20, 20, 1], rgba=[.3, .3, .3, 1])
ground.set_pos(np.array([0, 0, -1.5]))
ground.attach_to(base)
# groundGeom = OdeTriMeshGeom(space, OdeTriMeshData(ground.objpdnp, True))
groundGeom = OdePlaneGeom(space, Vec4(0, 0, 1, -1))
groundGeom.setCollideBits(BitMask32(0x00000001))
groundGeom.setCategoryBits(BitMask32(0x00000002))

# The task for our simulation
def simulationTask(task):
    space.autoCollide() # Setup the contact joints
    # Step the simulation and set the new positions
    world.step(globalClock.getDt())
    for cmobj, body in boxes:
        cmobj.set_homomat(rm.homomat_from_posrot(da.npvec3_to_pdvec3(body.getPosition()),
                                                 da.pdmat3_to_npmat3(body.getRotation())))
    contactgroup.empty() # Clear the contact joints
    return task.cont

# Wait a split second, then start the simulation
taskMgr.doMethodLater(0.5, simulationTask, "Physics Simulation")

base.run()