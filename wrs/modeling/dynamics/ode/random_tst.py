import wrs.visualization.panda.world as wd
from panda3d.ode import OdeWorld, OdeBody, OdeMass, OdeBallJoint
from wrs import basis as da, modeling as cm, modeling as gm
import numpy as np

base = wd.World(cam_pos=[7, 0, 0], lookat_pos=[0, 0, -.5], toggle_debug=True)
radius = .1

sphere_a = cm.gen_sphere(radius=radius)
sphere_a.set_pos([0, .3, -.3])
sphere_a.set_rgba([1, .2, .3, 1])
sphere_a.attach_to(base)

sphere_b = cm.gen_sphere(radius=radius)
sphere_b.set_pos([0, 1.25, -.7])
sphere_b.set_rgba([.3, .2, 1, 1])
sphere_b.attach_to(base)

gm.gen_linesegs([[np.zeros(3), sphere_a.get_pos()]], thickness=.05, rgba=[0, 1, 0, 1]).attach_to(base)
gm.gen_linesegs([[sphere_a.get_pos(), sphere_b.get_pos()]], thickness=.05, rgba=[0, 0, 1, 1]).attach_to(base)

# Setup our physics world and the body
world = OdeWorld()
world.setGravity(0, 0, -9.81)

body_sphere_a = OdeBody(world)
M = OdeMass()
M.setSphere(7874, radius)
body_sphere_a.setMass(M)
body_sphere_a.setPosition(da.npvec3_to_pdvec3(sphere_a.get_pos()))

body_sphere_b = OdeBody(world)
M = OdeMass()
M.setSphere(7874, radius)
body_sphere_b.setMass(M)
body_sphere_b.setPosition(da.npvec3_to_pdvec3(sphere_b.get_pos()))

# Create the joints
earth_a_jnt = OdeBallJoint(world)
earth_a_jnt.attach(body_sphere_a, None)  # Attach it to the environment
earth_a_jnt.setAnchor(0, 0, 0)
earth_b_jnt = OdeBallJoint(world)
earth_b_jnt.attach(body_sphere_a, body_sphere_b)
earth_b_jnt.setAnchor(0, .3, -.3)

# Create an accumulator to track the time since the sim
# has been running
deltaTimeAccumulator = 0.0
# This stepSize makes the simulation run at 90 frames per second
stepSize = 1.0 / 90.0


# The task for our simulation
def simulationTask(task):
    # Step the simulation and set the new positions
    world.quickStep(globalClock.getDt())
    sphere_a.set_pos(da.pdvec3_to_npvec3(body_sphere_a.getPosition()))
    sphere_b.set_pos(da.pdvec3_to_npvec3(body_sphere_b.getPosition()))
    gm.gen_linesegs([[np.zeros(3), sphere_a.get_pos()]], thickness=.05, rgba=[0, 1, 0, 1]).attach_to(base)
    gm.gen_linesegs([[sphere_a.get_pos(), sphere_b.get_pos()]], thickness=.05, rgba=[0, 0, 1, 1]).attach_to(base)
    return task.cont


taskMgr.doMethodLater(1.0, simulationTask, "Physics Simulation")

base.run()
