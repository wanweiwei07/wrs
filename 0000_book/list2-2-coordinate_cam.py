import numpy as np
import basis.robot_math as rm
import basis.data_adapter as da
import modeling.geometric_model as mgm
import visualization.panda.world as wd
from panda3d.core import CardMaker, NodePath, Texture, FrameBufferProperties, AmbientLight, PointLight, LMatrix4f
import visualization.panda.panda3d_utils as pdu

if __name__ == '__main__':
    # cam
    cam_pos = np.array([1, .8, .6])
    lookat_pos = np.zeros(3)
    # cam for rendering
    cam_pos_for_rendering = np.array([1.5, 1.5, 1.0])
    lookat_pos_for_rendering = np.array([.5, 0, 0])
    # setup world
    base = wd.World(cam_pos=cam_pos_for_rendering, lookat_pos=lookat_pos_for_rendering)
    # draw the frame
    frame_model = mgm.gen_frame()
    frame_model.attach_to(base)
    v_cam = pdu.VirtualCamera(cam_pos, lookat_pos)
    v_cam.gen_framemodel().attach_to(base)
    base.run()

    # texture
    tex = Texture()
    tex.setWrapU(Texture.WMClamp)
    tex.setWrapV(Texture.WMClamp)
    tex.setMinfilter(Texture.FTLinear)
    tex.setMagfilter(Texture.FTLinear)
    buffer = base.win.makeTextureBuffer("virtual_cam_buffer", 512, 512, tex, True)
    if buffer is None:
        raise Exception("Failed to create offscreen buffer")
    buffer.setClearColor((1, 1, 1, 1))
    base.virtual_cam = base.makeCamera(buffer, camName="virtual_cam")
    base.virtual_cam.setPos(*cam_pos)
    base.virtual_cam.lookAt(*lookat_pos)
    forward = lookat_pos - cam_pos
    upward = np.array([0, 0, 1])
    right = np.cross(forward, upward)
    upward = np.cross(right, forward)
    right = right / np.linalg.norm(right)
    upward = upward / np.linalg.norm(upward)
    forward = forward / np.linalg.norm(forward)
    cam_rotmat = np.column_stack((right, upward, forward))
    cam_box_half_length = 0.04
    cam_frustrum_half_length = 0.01
    mgm.gen_frame_box(xyz_lengths=np.array([.05, .05, cam_box_half_length]),
                      pos=cam_pos - (cam_box_half_length + cam_frustrum_half_length) * cam_rotmat[:, 2],
                      rotmat=cam_rotmat).attach_to(base)
    # mgm.gen_frame_cylinder(radius=.02, pos=cam_pos, rotmat=cam_rotmat).attach_to(base)
    top_xy_lengths = np.array([.04, .04])
    mgm.gen_frame_frustum(top_xy_lengths=top_xy_lengths,
                          height=cam_frustrum_half_length * 2,
                          pos=cam_pos - cam_frustrum_half_length * cam_rotmat[:, 2],
                          rotmat=cam_rotmat).attach_to(base)
    # mgm.gen_frame(pos=cam_pos, rotmat=cam_rotmat).attach_to(base)
    # # region cardmaker
    # cm = CardMaker("virtual_cam_display")
    # cm.setFrame(-top_xy_lengths[0] / 2, top_xy_lengths[0] / 2, -top_xy_lengths[1] / 2, top_xy_lengths[1] / 2)
    # card = NodePath(cm.generate())
    # card.setTexture(tex)
    # card.setTwoSided(True)
    # card.setPos(*cam_pos)
    # card.setHpr(*np.degrees(rm.rotmat_to_euler(cam_rotmat)))
    # card.setMat(da.npv3mat3_to_pdmat4(cam_pos, rm.rotmat_from_axangle(cam_rotmat[:, 0], np.pi / 2) @ cam_rotmat))
    # card.reparentTo(base.render)

    disp = pdu.Display("virtual_cam_display",
                      top_xy_lengths[0],
                      top_xy_lengths[1],
                      pos=cam_pos,
                      rotmat=cam_rotmat)
    disp.attach_to(base.render)
    base.run()
