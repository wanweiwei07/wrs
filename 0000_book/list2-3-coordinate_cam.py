import numpy as np
import modeling.geometric_model as mgm
import visualization.panda.world as wd
import visualization.panda.panda3d_utils as pdu

if __name__ == '__main__':
    # cam
    cam_pos = np.array([1, .8, .6])
    lookat_pos = np.zeros(3)
    # cam for rendering
    cam_pos_for_rendering = np.array([1.5, 1.5, 1])
    lookat_pos_for_rendering = np.array([0, 0, 0])
    # setup world
    base = wd.World(cam_pos=cam_pos_for_rendering, lookat_pos=lookat_pos_for_rendering)
    # draw the frame
    frame_model = mgm.gen_frame(ax_length=.2)
    frame_model.attach_to(base)
    # virtual cam
    cam_res = np.array([1920, 1080])
    screen_size = np.array([0.096, 0.054])
    v_cam = pdu.VirtualCamera(cam_pos, lookat_pos, resolution=cam_res,
                              screen_size=screen_size)
    v_cam.gen_meshmodel().attach_to(base)
    display = pdu.Display("virtual_cam_display", v_cam.screen_size)
    display.attach_to(v_cam)
    base.run()
