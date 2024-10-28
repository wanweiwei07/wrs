import mujoco
import glfw
import numpy as np

# Initialize the GLFW library
if not glfw.init():
    raise Exception("GLFW cannot be initialized!")

width, height = 800, 600

# Create a GLFW window
window = glfw.create_window(width, height, "MuJoCo Free Falling Sphere", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window cannot be created!")

# Make the context current
glfw.make_context_current(window)

# Load the model
try:
    model = mujoco.MjModel.from_xml_path('free_falling_sphere.xml')
except Exception as e:
    print(f"Error loading model: {e}")
    glfw.terminate()
    raise

data = mujoco.MjData(model)

# Create a camera and context
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scn = mujoco.MjvScene(model, maxgeom=1000)
con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

# Set the camera parameters
cam.azimuth = 0.0  # Horizontal angle
cam.elevation = 0.0  # Vertical angle
cam.distance = 5.0  # Distance from the model
cam.lookat = np.array([0.0, 0.0, 0.0])  # Camera lookat point

print(model.geom("sphere_geom").rgba)
print('id of "green_sphere": ', model.geom('sphere_geom').id)
print('name of geom 0: ', model.geom(0).name)
print('name of body 1: ', model.body(1).name)

# Render function
def render():
    viewport = mujoco.MjrRect(0, 0, width, height)
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
    mujoco.mjr_render(viewport, scn, con)

# Run the simulation
while not glfw.window_should_close(window):
    mujoco.mj_step(model, data)
    render()
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
