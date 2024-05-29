import mujoco
import numpy as np
import glfw
import time
import matplotlib.pyplot as plt

# Initialize GLFW
if not glfw.init():
    raise Exception("Failed to initialize GLFW")

xml = """
<mujoco model="denso_cobotta">
    <compiler angle="degree" coordinate="local"/>
    <option timestep="0.01" gravity="0 0 -9.81" iterations="50" integrator="Euler"/>

    <default>
        <joint limited="true" damping="1"/>
        <geom type="capsule" size="0.05" rgba="0.8 0.6 0.4 1"/>
    </default>

    <worldbody>
        <light diffuse="1 1 1" specular="0.1 0.1 0.1" pos="0 0 3"/>
        <geom name="floor" type="plane" pos="0 0 0" size="10 10 0.1" rgba="0.8 0.9 0.8 1"/>

        <!-- Base -->
        <body name="base" pos="0 0 0">
            <geom type="box" size="0.1 0.1 0.1" rgba="0.5 0.5 0.5 1"/>
            
            <!-- Link 1 -->
            <body name="link1" pos="0 0 0.1">
                <joint name="joint1" type="hinge" axis="0 0 1" range="-180 180"/>
                <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.05"/>
                
                <!-- Link 2 -->
                <body name="link2" pos="0 0 0.2">
                    <joint name="joint2" type="hinge" axis="0 1 0" range="-90 90"/>
                    <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.05"/>
                    
                    <!-- Link 3 -->
                    <body name="link3" pos="0 0 0.2">
                        <joint name="joint3" type="hinge" axis="1 0 0" range="-180 180"/>
                        <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.05"/>
                        
                        <!-- Link 4 -->
                        <body name="link4" pos="0 0 0.2">
                            <joint name="joint4" type="hinge" axis="0 1 0" range="-180 180"/>
                            <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.05"/>
                            
                            <!-- Link 5 -->
                            <body name="link5" pos="0 0 0.2">
                                <joint name="joint5" type="hinge" axis="1 0 0" range="-180 180"/>
                                <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.05"/>
                                
                                <!-- Link 6 -->
                                <body name="link6" pos="0 0 0.2">
                                    <joint name="joint6" type="hinge" axis="0 1 0" range="-180 180"/>
                                    <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.05"/>
                                    
                                    <!-- End Effector -->
                                    <body name="end_effector" pos="0 0 0.2">
                                        <geom type="sphere" size="0.05" rgba="1 0 0 1"/>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
</mujoco>
"""

def set_joint_positions(model, data, joint_names, target_positions):
    for joint_name, target_position in zip(joint_names, target_positions):
        joint_id = mujoco.mj_name2id(model,  mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[joint_id] = target_position
#
# Make model and data
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
target_positions = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

# model.opt.timestep = 0.1

# Setup a window for off-screen rendering
width, height = 800, 600
window = glfw.create_window(width, height, "Offscreen window", None, None)
glfw.make_context_current(window)

# Setup renderer and scene
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
scene = mujoco.MjvScene(model, maxgeom=1000)
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

# Set camera properties
cam.distance = model.stat.extent * 1.5


# 渲染函数
def render():
    viewport = mujoco.MjrRect(0, 0, width, height)
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)

duration = 7
framerate = 60
n_frames = int(duration * framerate)
mujoco.mj_resetData(model, data)
data.qvel[3:6] = 2*np.random.randn(3)
# data.joint('root').qvel = 10
# mujoco.mj_resetDataKeyframe(model, data, 0)
print('positions', data.qpos)
print('velocities', data.qvel)
frame = 0
render_time = 0
n_steps = 0
forcetorque = np.zeros(6)
for i in range(n_frames):
    while data.time * framerate < i:
        tic = time.time()
        mujoco.mj_step(model, data)
        n_steps += 1
    tic = time.time()
    # set_joint_positions(model, data, joint_names, target_positions)
    render()
    glfw.swap_buffers(window)
    render_time += time.time() - tic
    for geom_id in range(model.ngeom):
        print("geom ", data.geom_xpos[geom_id])
        print("body ", data.xpos[model.geom_bodyid[geom_id]])

glfw.terminate()
