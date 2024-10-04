from wrs import robot_sim as rbt, modeling as gm
import wrs.visualization.panda.world as wd
import numpy as np

base = wd.World(cam_pos=[-7, 0, 7], lookat_pos=[2.5, 0, 0], auto_cam_rotate=False)
rbt_s = rbt.TBMChanger()
rbt_s.fk("arm", np.array([0.1, .1, .1, .1, .1, .1]))
rbt_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
seed0 = np.zeros(6)
m_mat = rbt_s.manipulability_axmat(type="translational")
jac = rbt_s.jacobian(component_name="arm")
print("jac = ", jac)
gm.gen_ellipsoid(pos=rbt_s.get_gl_tcp("arm")[0], axes_mat=m_mat).attach_to(base)
base.run()