import numpy as np
import wrs.visualization.panda.world as wd
from wrs import basis as rm, modeling as gm
import math

base = wd.World(cam_pos=np.array([.7, -.7, 1]), lookat_pos=np.array([0.3, 0, 0]))
gm.gen_frame().attach_to(base)
sec1_spos = np.array([0, 0, 0])
sec_len = np.array([.2, 0, 0])
sec1_epos = sec1_spos+sec_len
sec2_spos = sec1_epos

angle=math.pi/5

sec2_rotaxis = np.array([0, 0, 1])
sec2_rotmat = rm.rotmat_from_axangle(sec2_rotaxis, angle)
sec2_epos = sec2_spos + sec2_rotmat.dot(sec_len)
rgba = [1, .4, 0, .3]
gm.gen_stick(spos=sec1_spos, epos=sec1_epos, rgba=rgba, radius=.015).attach_to(base)
gm.gen_frame(pos=sec2_spos, alpha=.2).attach_to(base)
gm.gen_stick(spos=sec2_spos, epos=sec2_spos + sec_len, rgba=rgba, radius=.015).attach_to(base)
gm.gen_dashed_frame(pos=sec2_spos, rotmat=sec2_rotmat).attach_to(base)
gm.gen_stick(spos=sec2_spos, epos=sec2_epos, rgba=rgba, radius=.015).attach_to(base)
gm.gen_circarrow(axis=sec2_rotaxis, center=sec2_rotaxis / 13+sec2_spos, starting_vector=np.array([-0,-1,0]),radius=.025, portion=.5, thickness=.003, rgba=[1,.4,0,1]).attach_to(base)
#
# sec2_rotaxis2 = np.array([1, 0, 0])
# sec2_rotmat2 = rm.rotmat_from_axangle(sec2_rotaxis2, math.pi/3)
# sec2_epos = sec2_spos + sec2_rotmat2.dot(sec2_epos-sec2_spos)
# # sec2_rotmat2 = rm.rotmat_from_axangle([1,0,0], math.pi/6)
# # sec2_epos2 = sec2_spos + sec2_rotmat.dot(np.array([.3, 0, 0]))
# # rgba = [1, .4, 0, .3]
# # mgm.gen_stick(spos=sec1_spos, epos=sec1_epos, rgba=rgba, major_radius=.015).attach_to(base)
# # mgm.gen_dashframe(pos=sec2_spos).attach_to(base)
# mgm.gen_frame(pos=sec2_spos,rotmat=sec2_rotmat2, alpha=.2).attach_to(base)
# mgm.gen_stick(spos=sec2_spos, epos=sec2_epos, rgba=rgba, major_radius=.015).attach_to(base)
# mgm.gen_circarrow(axis=sec2_rotaxis2, center=sec2_rotaxis2 / 13+sec2_spos, starting_vector=np.array([0,0,-1]),major_radius=.025, portion=.6, major_radius=.003, rgba=[1,.4,0,1]).attach_to(base)
# #
# sec2_rotaxis3 = np.array([0, 1, 0])
# sec2_rotmat3 = rm.rotmat_from_axangle(sec2_rotaxis3, math.pi/3)
# sec2_epos = sec2_spos + sec2_rotmat3.dot(sec2_epos-sec2_spos)
# # sec2_rotmat2 = rm.rotmat_from_axangle([1,0,0], math.pi/6)
# # sec2_epos2 = sec2_spos + sec2_rotmat.dot(np.array([.3, 0, 0]))
# rgba = [1, .4, 0, .3]
# # mgm.gen_stick(spos=sec1_spos, epos=sec1_epos, rgba=rgba, major_radius=.015).attach_to(base)
# mgm.gen_dashframe(pos=sec2_spos).attach_to(base)
# mgm.gen_frame(pos=sec2_spos,rotmat=sec2_rotmat3, alpha=.2).attach_to(base)
# mgm.gen_stick(spos=sec2_spos, epos=sec2_epos, rgba=rgba, major_radius=.015).attach_to(base)
# mgm.gen_circarrow(axis=sec2_rotaxis3, center=sec2_rotaxis3 / 13+sec2_spos, starting_vector=np.array([0,0,1]),major_radius=.025, portion=.55, major_radius=.003, rgba=[1,.4,0,1]).attach_to(base)

base.run()
