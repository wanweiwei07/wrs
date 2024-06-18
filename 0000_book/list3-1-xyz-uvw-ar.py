import numpy as np
import modeling.geometric_model as mgm
import visualization.panda.world as wd


def draw_coord(pnt, rotmat=np.eye(3), toggle_pnt=False, toggle_coord=False):
    rgb = np.array([0, 0, 0])
    alpha = 1
    px, py, pz = pnt[0], pnt[1], pnt[2]
    global_pnt = rotmat @ pnt
    if toggle_pnt:
        mgm.gen_sphere(pos=global_pnt, radius=.005, rgb=rgb, alpha=alpha).attach_to(base)
    if toggle_coord:
        mgm.gen_sphere(pos=np.array([px, 0, 0]), radius=.005, rgb=rgb, alpha=alpha).attach_to(base)
        mgm.gen_sphere(pos=np.array([0, py, 0]), radius=.005, rgb=rgb, alpha=alpha).attach_to(base)
        mgm.gen_sphere(pos=np.array([0, 0, pz]), radius=.005, rgb=rgb, alpha=alpha).attach_to(base)
    mgm.gen_stick(spos=global_pnt, epos=rotmat @ np.array([px, py, 0]), radius=0.001, rgb=rgb, alpha=alpha).attach_to(base)
    mgm.gen_dashed_stick(spos=rotmat @ np.array([px, py, 0]), epos=rotmat @ np.array([px, 0, 0]), radius=0.001,
                         rgb=rgb, alpha=alpha, len_solid=.007,
                         len_interval=.005).attach_to(base)
    mgm.gen_dashed_stick(spos=rotmat @ np.array([px, py, 0]), epos=rotmat @ np.array([0, py, 0]), radius=0.001,
                         rgb=rgb, alpha=alpha, len_solid=.007,
                         len_interval=.005).attach_to(base)
    mgm.gen_stick(spos=global_pnt, epos=rotmat @ np.array([px, 0, pz]), radius=0.001, rgb=rgb, alpha=alpha).attach_to(base)
    mgm.gen_dashed_stick(spos=rotmat @ np.array([px, 0, pz]), epos=rotmat @ np.array([px, 0, 0]), radius=0.001,
                         rgb=rgb, alpha=alpha, len_solid=.007,
                         len_interval=.005).attach_to(base)
    mgm.gen_dashed_stick(spos=rotmat @ np.array([px, 0, pz]), epos=rotmat @ np.array([0, 0, pz]), radius=0.001,
                         rgb=rgb, alpha=alpha, len_solid=.007,
                         len_interval=.005).attach_to(base)
    mgm.gen_stick(spos=global_pnt, epos=rotmat @ np.array([0, py, pz]), radius=0.001, rgb=rgb, alpha=alpha).attach_to(base)
    mgm.gen_dashed_stick(spos=rotmat @ np.array([0, py, pz]), epos=rotmat @ np.array([0, py, 0]), radius=0.001,
                         rgb=rgb, alpha=alpha, len_solid=.007,
                         len_interval=.005).attach_to(base)
    mgm.gen_dashed_stick(spos=rotmat @ np.array([0, py, pz]), epos=rotmat @ np.array([0, 0, pz]), radius=0.001,
                         rgb=rgb, alpha=alpha, len_solid=.007,
                         len_interval=.005).attach_to(base)


if __name__ == '__main__':
    base = wd.World(cam_pos=np.array([1, 1, 1]), lookat_pos=np.zeros(3))
    mgm.gen_frame(ax_length=.2).attach_to(base)
    o_r_a = np.array([[0.4330127, -0.64951905, 0.625],
                      [0.75, -0.125, -0.64951905],
                      [0.5, 0.75, 0.4330127]])
    import basis.robot_math as rm
    print(np.degrees(rm.rotmat_to_euler(o_r_a, order='sxyz')))
    print(np.degrees(rm.rotmat_to_euler(o_r_a, order='rzxz')))
    mgm.gen_dashed_frame(rotmat=o_r_a, ax_length=.2).attach_to(base)
    a_r = np.array([.15, .07, .05])
    o_r = o_r_a @ a_r
    # draw_coord(a_r, rotmat=o_r_a, toggle_pnt=True)
    draw_coord(o_r, toggle_pnt=True)
    base.run()
