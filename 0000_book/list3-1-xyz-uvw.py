import numpy as np
import modeling.geometric_model as mgm
import visualization.panda.world as wd

if __name__ == '__main__':
    base = wd.World(cam_pos=np.array([1, 1, 1]), lookat_pos=np.zeros(3))
    frame_model = mgm.gen_frame(ax_length=.2).attach_to(base)
    o_r_a = np.array([[0.4330127, -0.64951905, 0.625],
                      [0.75, -0.125, -0.64951905],
                      [0.5, 0.75, 0.4330127]])
    frame_dashed_model = mgm.gen_dashed_frame(rotmat=o_r_a, ax_length=.2).attach_to(base)
    base.run()