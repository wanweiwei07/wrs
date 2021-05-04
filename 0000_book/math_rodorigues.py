import visualization.panda.world as wd
import modeling.geometric_model as gm
import basis.robot_math as rm
import math
import numpy as np

rotmat = rm.rotmat_from_euler(math.pi / 3, -math.pi / 6, math.pi / 3)
ax, angle = rm.axangle_between_rotmat(np.eye(3), rotmat)
rotmat2 = rm.rotmat_from_axangle(ax, math.pi/6)
# cross_vec = rm.unit_vector(np.cross(np.array([0.1,0,1]), rotmat[:3,2]))
cross_vec = rotmat2.dot(rotmat[:3,0])
base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0], toggle_debug=True)
# base = wd.World(cam_pos=ax*2, lookat_pos=[0, 0, 0], toggle_debug=True)

gm.gen_arrow(epos=ax*.3, rgba=[0,0,0,.3]).attach_to(base)
gm.gen_frame(length=.2).attach_to(base)

# gm.gen_dasharrow(epos=cross_vec*.3, rgba=[1,1,0,1], lsolid=.01, lspace=.0077).attach_to(base)
gm.gen_arrow(epos=cross_vec*.3, rgba=[1,1,0,1]).attach_to(base)
gm.gen_sphere(radius=.005, pos=cross_vec*.3, rgba=[0,0,0,1]).attach_to(base)
nxt_vec_uvw = rotmat2.dot(cross_vec)
gm.gen_dasharrow(epos=nxt_vec_uvw*.3, rgba=[1,1,0,1]).attach_to(base)
# gm.gen_arrow(epos=nxt_vec_uvw*.3, rgba=[1,1,0,1]).attach_to(base)
gm.gen_sphere(radius=.005, pos=nxt_vec_uvw*.3, rgba=[0,0,0,1]).attach_to(base)
radius, _ = rm.unit_vector(cross_vec * .3 - cross_vec.dot(ax) * ax * .3, toggle_length=True)
gm.gen_arrow(spos=cross_vec.dot(ax)*ax*.3, epos = cross_vec*.3, rgba=[1,.47,0,.5]).attach_to(base)
gm.gen_dasharrow(spos=cross_vec.dot(ax)*ax*.3, epos = nxt_vec_uvw*.3, rgba=[1,.47,0,.5]).attach_to(base)
gm.gen_arrow(epos=ax*math.sqrt(.3**2-radius**2), rgba=[0,0,0,1]).attach_to(base)

## projections
# gm.gen_dasharrow(spos = ax*math.sqrt(.3**2-radius**2),
#              epos=ax*math.sqrt(.3**2-radius**2)+np.cross(ax, cross_vec*.3)*math.sin(math.pi/6),
#              rgba=[1,0,1,.5]).attach_to(base)
# gm.gen_dasharrow(spos = ax*math.sqrt(.3**2-radius**2),
#              epos=ax*math.sqrt(.3**2-radius**2)+(cross_vec*.3-cross_vec.dot(ax)*ax*.3)*math.cos(math.pi/6),
#              rgba=[0,1,1,.5]).attach_to(base)

# rectangle
epos_vec = rm.unit_vector(cross_vec*.3-cross_vec.dot(ax)*ax*.3)
gm.gen_stick(spos=cross_vec.dot(ax)*ax*.3-ax*.03,
             epos =cross_vec.dot(ax)*ax*.3-ax*.03+epos_vec*.03,
             rgba=[0,0,0,1],
             thickness=.001).attach_to(base)
gm.gen_stick(spos=cross_vec.dot(ax)*ax*.3-ax*.03+epos_vec*.03,
             epos=cross_vec.dot(ax)*ax*.3-ax*.03+epos_vec*.03+ax*.03-ax*.005,
             rgba=[0,0,0,1],
             thickness=.001).attach_to(base)
epos_vec = rm.unit_vector(nxt_vec_uvw*.3-nxt_vec_uvw.dot(ax)*ax*.3)
gm.gen_stick(spos=nxt_vec_uvw.dot(ax)*ax*.3-ax*.03,
             epos =nxt_vec_uvw.dot(ax)*ax*.3-ax*.03+epos_vec*.03,
             rgba=[0,0,0,1],
             thickness=.001).attach_to(base)
gm.gen_stick(spos=nxt_vec_uvw.dot(ax)*ax*.3-ax*.03+epos_vec*.03,
             epos=nxt_vec_uvw.dot(ax)*ax*.3-ax*.03+epos_vec*.03+ax*.03-ax*.0045,
             rgba=[0,0,0,1],
             thickness=.001).attach_to(base)
gm.gen_torus(ax,
             starting_vector=nxt_vec_uvw*.3-nxt_vec_uvw.dot(ax)*ax*.3,
             portion=.9219,
             center=cross_vec.dot(ax)*ax*.3,
             radius=radius,
             discretization=64,
             sections=16,
             thickness=.003,
             rgba=[1,1,1,1]).attach_to(base)
gm.gen_circarrow(ax,
                 starting_vector=cross_vec*.3-cross_vec.dot(ax)*ax*.3,
                 portion=.082,
                 center=cross_vec.dot(ax)*ax*.3,
                 radius=radius,
                 discretization=64,
                 sections=16,
                 thickness=.0037,
                 rgba=[0,0,0,1]).attach_to(base)


base.run()