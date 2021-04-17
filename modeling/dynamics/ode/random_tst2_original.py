from direct.directbase import DirectStart
from panda3d.ode import OdeWorld, OdeSimpleSpace, OdeJointGroup
from panda3d.ode import OdeBody, OdeMass, OdeBoxGeom, OdePlaneGeom, OdeTriMeshGeom, OdeTriMeshData
from panda3d.core import BitMask32, CardMaker, Vec4, Quat
from random import randint, random

# Setup our physics world
world = OdeWorld()
world.setGravity(0, 0, -9.81)

# The surface table is needed for autoCollide
world.initSurfaceTable(1)
world.setSurfaceEntry(0, 0, 150, 0.0, 9.1, 0.9, 0.00001, 0.0, 0.002)

# Create a space and add a contactgroup to it to add the contact joints
space = OdeSimpleSpace()
space.setAutoCollideWorld(world)
contactgroup = OdeJointGroup()
space.setAutoCollideJointGroup(contactgroup)

# Load the box
box = loader.loadModel("box")
# Make sure its center is at 0, 0, 0 like OdeBoxGeom
box.setPos(-.5, -.5, -.5)
box.flattenLight() # Apply transform
box.setTextureOff()

# Add a random amount of boxes
boxes = []
for i in range(randint(5, 10)):
    # Setup the geometry
    boxNP = box.copyTo(render)
    # boxNP.setPos(randint(-10, 10), randint(-10, 10), 10 + random())
    boxNP.setPos(random()*10, random()*10, 10 + random())
    boxNP.setColor(random(), random(), random(), 1)
    boxNP.setHpr(randint(-45, 45), randint(-45, 45), randint(-45, 45))
    # Create the body and set the mass
    boxBody = OdeBody(world)
    M = OdeMass()
    M.setBox(5, 1, 1, 1)
    boxBody.setMass(M)
    boxBody.setPosition(boxNP.getPos(render))
    boxBody.setQuaternion(boxNP.getQuat(render))
    # Create a BoxGeom
    boxGeom = OdeBoxGeom(space, 1, 1, 1)
    # boxGeom = OdeTriMeshGeom(space, OdeTriMeshData(boxNP, True))
    boxGeom.setCollideBits(BitMask32(0x00000002))
    boxGeom.setCategoryBits(BitMask32(0x00000001))
    boxGeom.setBody(boxBody)
    boxes.append((boxNP, boxBody))

# Add a plane to collide with
cm = CardMaker("ground")
cm.setFrame(-20, 20, -20, 20)
ground = render.attachNewNode(cm.generate())
ground.setPos(0, 0, 0); ground.lookAt(0, 0, -1)
groundGeom = OdePlaneGeom(space, Vec4(0, 0, 1, 0))
# groundGeom = OdeTriMeshGeom(space, OdeTriMeshData(ground, True))
groundGeom.setCollideBits(BitMask32(0x00000001))
groundGeom.setCategoryBits(BitMask32(0x00000002))

# Set the camera position
base.disableMouse()
base.camera.setPos(40, 40, 20)
base.camera.lookAt(0, 0, 0)

# The task for our simulation
def simulationTask(task):
    space.autoCollide() # Setup the contact joints
    # Step the simulation and set the new positions
    world.quickStep(globalClock.getDt())
    for np, body in boxes:
        np.setPosQuat(render, body.getPosition(), Quat(body.getQuaternion()))
    contactgroup.empty() # Clear the contact joints
    return task.cont

# Wait a split second, then start the simulation
taskMgr.doMethodLater(0.5, simulationTask, "Physics Simulation")

base.run()