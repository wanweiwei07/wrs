import mujoco
import glfw
import numpy as np
import os

# 初始化 GLFW
if not glfw.init():
    raise Exception("GLFW can't be initialized")

width=1920
height=1080
# 创建一个窗口
window = glfw.create_window(width,height, "MuJoCo Simulation", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window can't be created")

glfw.make_context_current(window)

# 加载模型和创建仿真器
model = mujoco.MjModel.from_xml_path("humanoid.xml")
data = mujoco.MjData(model)

# 创建场景和上下文
scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
opt = mujoco.MjvOption()
cam = mujoco.MjvCamera()
cam.azimuth = 90.0  # 水平方向的角度
cam.elevation = -20.0  # 垂直方向的角度
cam.distance = 5.0  # 相机与模型的距离
cam.lookat = np.array([0.0, 0.0, 1.0])  # 相机的观察点

# 渲染函数
def render():
    viewport = mujoco.MjrRect(0, 0, width,height)
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)

# 运行仿真
while not glfw.window_should_close(window):
    mujoco.mj_step(model, data)
    render()
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()