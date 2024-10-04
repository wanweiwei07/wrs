import numpy as np
from wrs import modeling as mgm
import wrs.visualization.panda.world as wd


def draw_coord(pnt, toggle_pnt=False, toggle_coord=False):
    rgba = np.array([0, 0, 0, 1])
    if toggle_pnt:
        mgm.gen_sphere(pos=pnt, radius=.005, rgb=rgba[:3], alpha=rgba[3]).attach_to(base)
    px, py, pz = pnt[0], pnt[1], pnt[2]
    if toggle_coord:
        mgm.gen_sphere(pos=np.array([px, 0, 0]), radius=.005, rgb=rgba[:3], alpha=rgba[3]).attach_to(base)
        mgm.gen_sphere(pos=np.array([0, py, 0]), radius=.005, rgb=rgba[:3], alpha=rgba[3]).attach_to(base)
        mgm.gen_sphere(pos=np.array([0, 0, pz]), radius=.005, rgb=rgba[:3], alpha=rgba[3]).attach_to(base)
    mgm.gen_stick(spos=pnt, epos=np.array([px, py, 0]), radius=0.001, rgb=rgba[:3], alpha=rgba[3]).attach_to(base)
    mgm.gen_dashed_stick(spos=np.array([px, py, 0]), epos=np.array([px, 0, 0]), radius=0.001, rgb=rgba[:3], alpha=rgba[3], len_solid=.007,
                         len_interval=.005).attach_to(base)
    mgm.gen_dashed_stick(spos=np.array([px, py, 0]), epos=np.array([0, py, 0]), radius=0.001, rgb=rgba[:3], alpha=rgba[3], len_solid=.007,
                         len_interval=.005).attach_to(base)
    mgm.gen_stick(spos=pnt, epos=np.array([px, 0, pz]), radius=0.001, rgb=rgba[:3], alpha=rgba[3]).attach_to(base)
    mgm.gen_dashed_stick(spos=np.array([px, 0, pz]), epos=np.array([px, 0, 0]), radius=0.001, rgb=rgba[:3], alpha=rgba[3], len_solid=.007,
                         len_interval=.005).attach_to(base)
    mgm.gen_dashed_stick(spos=np.array([px, 0, pz]), epos=np.array([0, 0, pz]), radius=0.001, rgb=rgba[:3], alpha=rgba[3], len_solid=.007,
                         len_interval=.005).attach_to(base)
    mgm.gen_stick(spos=pnt, epos=np.array([0, py, pz]), radius=0.001, rgb=rgba[:3], alpha=rgba[3]).attach_to(base)
    mgm.gen_dashed_stick(spos=np.array([0, py, pz]), epos=np.array([0, py, 0]), radius=0.001, rgb=rgba[:3], alpha=rgba[3], len_solid=.007,
                         len_interval=.005).attach_to(base)
    mgm.gen_dashed_stick(spos=np.array([0, py, pz]), epos=np.array([0, 0, pz]), radius=0.001, rgb=rgba[:3], alpha=rgba[3], len_solid=.007,
                         len_interval=.005).attach_to(base)


if __name__ == '__main__':
    base = wd.World(cam_pos=np.array([1, 1, 1]), lookat_pos=np.zeros(3))
    mgm.gen_frame(ax_length=.2).attach_to(base)
    o_r_a = np.array([[0.4330127, -0.64951905, 0.625],
                      [0.75, -0.125, -0.64951905],
                      [0.5, 0.75, 0.4330127]])
    mgm.gen_dashed_frame(rotmat=o_r_a, ax_length=.2).attach_to(base)
    # draw_coord(pnt=o_r_a[:, 0]*.2, toggle_pnt=False, toggle_coord=True)
    # draw_coord(pnt=o_r_a[:, 1]*.2, toggle_pnt=False, toggle_coord=True)
    # mgm.gen_stick(spos=np.zeros(3), epos=-np.array([0,1,0])*.2, rgba=[0, 1, 0, 1]).attach_to(base)
    # mgm.gen_stick(spos=np.zeros(3), epos=-np.array([1,0,0])*.2, rgba=[1, 0, 0, 1]).attach_to(base)
    draw_coord(pnt=o_r_a[:, 2]*.2, toggle_pnt=False, toggle_coord=True)
    mgm.gen_stick(spos=np.zeros(3), epos=-np.array([0,1,0])*.2, rgba=[0, 1, 0, 1]).attach_to(base)
    # a_r = np.array([.05, .07, .15])
    # draw_coord(a_r)
    base.run()
