import wrs.visualization.panda.world as wd
from wrs import robot_sim as xss, modeling as gm

base = wd.World(cam_pos=[10, 1, 5], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)
# robot_s
component_name='agv'
robot_s = xss.XArmShuidi()
robot_s.gen_meshmodel().attach_to(base)
m_mat = robot_s.manipulability_axmat("arm", type="rotational")
print(m_mat)
gm.gen_ellipsoid(pos=robot_s.get_gl_tcp("arm")[0], axes_mat=m_mat).attach_to(base)
base.run()
