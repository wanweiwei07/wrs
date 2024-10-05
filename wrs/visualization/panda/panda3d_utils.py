"""
Panda3d Extension
Author: Hao Chen
"""
import functools

import numpy as np
import cv2
from panda3d.core import (Texture,
                          NodePath,
                          WindowProperties,
                          Vec3,
                          Point3,
                          PerspectiveLens,
                          OrthographicLens,
                          PGTop,
                          CardMaker)
from direct.gui.OnscreenImage import OnscreenImage

import wrs.basis.robot_math as rm
import wrs.basis.data_adapter as da
import wrs.visualization.panda.filter as flt
import wrs.visualization.panda.inputmanager as im
import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as mgm
import wrs.modeling.model_collection as mmc

class VirtualCamera(object):

    def __init__(self, cam_pos, lookat_pos, resolution=np.array((512, 512)), screen_size=np.array((0.16, 0.16))):
        """
        :param cam_pos:
        :param lookat_pos:
        :param resolution:
        :param cam_view_size: (1,2) np array, width x height, mm x mm
        """
        self._cam_pos = cam_pos
        self._lookat_pos = lookat_pos
        self.screen_size = screen_size
        self._tex = Texture()
        self._tex.setWrapU(Texture.WMClamp)
        self._tex.setWrapV(Texture.WMClamp)
        self._tex.setMinfilter(Texture.FTLinear)
        self._tex.setMagfilter(Texture.FTLinear)
        self._buffer = base.win.makeTextureBuffer("virtual_cam_buffer", resolution[0], resolution[1], self._tex, True)
        self._buffer.setClearColor((1, 1, 1, 1))
        self._pdnp = base.makeCamera(self._buffer, camName="virtual_cam")
        self._pdnp.setPos(*cam_pos)
        self._pdnp.lookAt(*lookat_pos)  # cam look at 0,1,0 by default in panda3d

    @property
    def cam_pos(self):
        return np.array([*self._pdnp.getPos()])

    @property
    def cam_rotmat(self):
        return np.array(self._pdnp.getMat().getUpper3()).T

    def gen_framemodel(self, name="virtual_cam_frame_model", alpha=1):
        m_col = mmc.ModelCollection(name=name)
        cam_view_length = max(self.screen_size)
        cam_length = cam_view_length * 1.25
        cam_width = cam_length * 0.5
        cam_lens_length = cam_length / 4
        mgm.gen_frame_box(xyz_lengths=np.array([cam_width, cam_length, cam_width]),
                          pos=self.cam_pos - (cam_length / 2 + cam_lens_length) * self.cam_rotmat[:, 1],
                          rotmat=self.cam_rotmat, alpha=alpha).attach_to(m_col)
        mgm.gen_frame_frustum(
            bottom_xy_lengths=np.array([cam_width, cam_width]),
            top_xy_lengths=np.array([cam_view_length, cam_view_length]),
            height=cam_lens_length,
            pos=self.cam_pos - (cam_lens_length) * self.cam_rotmat[:, 1],
            rotmat=self.cam_rotmat @ rm.rotmat_from_euler(-np.pi / 2, 0, 0), alpha=alpha).attach_to(m_col)
        return m_col

    def gen_meshmodel(self, name="virtual_cam_frame_model", rgb=np.array([.3, .3, .3]), alpha=.3):
        m_col = mmc.ModelCollection(name=name)
        cam_view_length = max(self.screen_size)
        cam_length = cam_view_length * 1.25
        cam_width = cam_length * 0.4
        cam_lens_length = cam_length / 4
        mgm.gen_box(xyz_lengths=np.array([cam_width, cam_length, cam_width]),
                    pos=self.cam_pos - (cam_length / 2 + cam_lens_length) * self.cam_rotmat[:, 1],
                    rotmat=self.cam_rotmat, rgb=rgb, alpha=alpha).attach_to(m_col)
        mgm.gen_frustrum(
            bottom_xy_lengths=np.array([cam_width, cam_width]),
            top_xy_lengths=np.array([cam_view_length, cam_view_length]),
            height=cam_lens_length,
            pos=self.cam_pos - (cam_lens_length) * self.cam_rotmat[:, 1],
            rotmat=self.cam_rotmat @ rm.rotmat_from_euler(-np.pi / 2, 0, 0),
            rgb=rgb, alpha=alpha).attach_to(m_col)
        return m_col


def attach_to(self, parent):
    self._pdnp.reparentTo(parent)


class Display(object):

    def __init__(self, name, size=np.array([.1, .1]), pos=np.zeros(3), rotmat=np.eye(3)):
        """
        :param name:
        :param size: (1,2) np array, width x height, mm x mm
        :param pos:
        :param rotmat:
        """
        self._pdcm = CardMaker(name)
        self._pdcm.setFrame(-size[0] / 2, size[0] / 2, -size[1] / 2, size[1] / 2)
        self._pdnp = NodePath(self._pdcm.generate())
        self._pdnp.setTwoSided(True)
        # self._pdnp.setMat(*rm.homomat_from_posrot(pos,
        #                                           rm.rotmat_from_axangle(rotmat[:, 0], np.pi / 2) @ rotmat).T.flatten())
        # self._pdnp.setPos(*pos)
        # self._pdnp.setMat(da.npv3mat3_to_pdmat4(pos, rm.rotmat_from_axangle(rotmat[:, 0], np.pi / 2) @ rotmat))
        self._pdnp.setMat(da.npv3mat3_to_pdmat4(pos, rotmat))

    def attach_to(self, parent):
        if isinstance(parent, VirtualCamera):
            self._pdnp.reparentTo(parent._pdnp)
            self._pdnp.setTexture(parent._tex)


def img_to_n_channel(img, channel=3):
    """
    Repeat a channel n times and stack
    :param img:
    :param channel:
    :return:
    """
    return np.stack((img,) * channel, axis=-1)


def letter_box(img, new_shape=(640, 640), color=(.45, .45, .45), auto=True, scale_fill=False, scale_up=True, stride=32):
    """
    This function is copied from YOLOv5 (https://github.com/ultralytics/yolov5)
    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:  # only scale down, do not scale up (for better value mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                             value=(color[0] * 255, color[0] * 255, color[0] * 255))  # add border
    return img


class ImgOnscreen(object):
    """
    Add a on screen image in the render 2d scene of Showbase
    """

    def __init__(self, size, parent_np=None):
        """
        :param size: (width, height)
        :param parent_np: Should be ShowBase or ExtraWindow
        author: chenhao
        """
        self._size = size
        self.tx = Texture("video")
        self.tx.setup2dTexture(size[0], size[1], Texture.TUnsignedByte, Texture.FRgb8)
        # this makes some important setup call
        # self.tx.load(PNMImage(card_size[0], card_size[1]))
        self.onscreen_image = OnscreenImage(self.tx,
                                            pos=(0, 0, 0),
                                            parent=parent_np.render2d)

    def update_img(self, img: np.ndarray):
        """
        Update the onscreen image
        :param img:
        :return:
        """
        if img.shape[2] == 1:
            img = img_to_n_channel(img)
        resized_img = letter_box(img, new_shape=[self._size[1], self._size[0]], auto=False)
        self.tx.setRamImage(resized_img.tostring())

    def remove(self):
        """
        Release the memory
        :return:
        """
        if self.onscreen_image is not None:
            self.onscreen_image.destroy()

    def __del__(self):
        self.remove()


class ExtraWindow(object):
    """
    Create a extra window on the scene
    TODO: small bug to fix: win.requestProperties does not change the properties of window immediately
    :return:
    """

    def __init__(self, base: wd.World,
                 window_title: str = "WRS Robot Planning and Control System",
                 cam_pos: np.ndarray = np.array([2.0, 0, 2.0]),
                 lookat_pos: np.ndarray = np.array([0, 0, 0.25]),
                 up: np.ndarray = np.array([0, 0, 1]),
                 fov: int = 40,
                 w: int = 1920,
                 h: int = 1080,
                 lens_type: str = "perspective"):
        self._base = base
        # setup render scene
        self.render = NodePath("extra_win_render")
        # setup render 2d
        self.render2d = NodePath("extra_win_render2d")
        self.render2d.setDepthTest(0)
        self.render2d.setDepthWrite(0)
        # setup window
        self.win = base.openWindow(props=WindowProperties(base.win.getProperties()),
                                   makeCamera=False,
                                   scene=self.render,
                                   requireWindow=True, )
        # set window background to white
        base.setBackgroundColor(r=1, g=1, b=1, win=self.win)
        # set window title and window's dimension
        self.set_win_props(title=window_title,
                           size=(w, h))
        # set len for the camera and set the camera for the new window
        lens = PerspectiveLens()
        lens.setFov(fov)
        lens.setNearFar(0.001, 5000.0)
        if lens_type == "orthographic":
            lens = OrthographicLens()
            lens.setFilmSize(1, 1)
        # make aspect ratio looks same as base window
        aspect_ratio = base.getAspectRatio()
        lens.setAspectRatio(aspect_ratio)
        self.cam = base.makeCamera(self.win, scene=self.render, )  # can also be found in base.camList
        self.cam.reparentTo(self.render)
        self.cam.setPos(Point3(cam_pos[0], cam_pos[1], cam_pos[2]))
        self.cam.lookAt(Point3(lookat_pos[0], lookat_pos[1], lookat_pos[2]), Vec3(up[0], up[1], up[2]))
        self.cam.node().setLens(lens)  # use same len as sys
        # set up cartoon effect
        self._separation = 1
        self.filter = flt.Filter(self.win, self.cam)
        self.filter.setCartoonInk(separation=self._separation)
        # camera in camera 2d
        self.cam2d = base.makeCamera2d(self.win, )
        self.cam2d.reparentTo(self.render2d)
        # attach GPTop to the render2d to make sure the DirectGui can be used
        self.aspect2d = self.render2d.attachNewNode(PGTop("aspect2d"))
        # self.aspect2d.setScale(1.0 / aspect_ratio, 1.0, 1.0)
        # setup mouse for the new window
        # name of mouse watcher is to adapt to the name in the input manager
        self.mouse_thrower = base.setupMouse(self.win, fMultiWin=True)
        self.mouseWatcher = self.mouse_thrower.getParent()
        self.mouseWatcherNode = self.mouseWatcher.node()
        self.aspect2d.node().setMouseWatcher(self.mouseWatcherNode)
        # self.mouseWatcherNode.addRegion(PGMouseWatcherBackground())
        # setup input manager
        self.inputmgr = im.InputManager(self, lookat_pos=lookat_pos)
        # copy attributes and functions from base
        ## change the bound function to a function, and bind to `self` to become a unbound function
        self._interaction_update = functools.partial(base._interaction_update.__func__, self)
        self.p3dh = base.p3dh
        base.taskMgr.add(self._interaction_update, "interaction_extra_window", appendTask=True)

    @property
    def size(self):
        size = self.win.getProperties().size
        return np.array([size[0], size[1]])

    def getAspectRatio(self):
        return self._base.getAspectRatio(self.win)

    def set_win_props(self,
                      title: str,
                      size: tuple):
        """
        set properties of extra window
        :param title: the title of the window
        :param size: 1x2 tuple describe width and height
        :return:
        """
        win_props = WindowProperties()
        win_props.setSize(size[0], size[1])
        win_props.setTitle(title)
        self.win.requestProperties(win_props)

    def set_origin(self, origin: np.ndarray):
        """
        :param origin: 1x2 np array describe the left top corner of the window
        """
        win_props = WindowProperties()
        win_props.setOrigin(origin[0], origin[1])
        self.win.requestProperties(win_props)


if __name__ == "__main__":
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    gm.gen_frame(ax_length=.2).attach_to(base)

    # extra window 1
    ew = ExtraWindow(base, cam_pos=np.array([2, 0, 1.5]), lookat_pos=np.array([0, 0, .2]))
    ew.set_origin((np.array([0, 40])))
    # ImgOnscreen()
    img = cv2.imread("img.png")
    on_screen_img = ImgOnscreen(img.shape[:2][::-1], parent_np=ew)
    on_screen_img.update_img(img)

    # extra window 2
    ew2 = ExtraWindow(base, cam_pos=np.array([2, 0, 1.5]), lookat_pos=np.array([0, 0, .2]))
    ew2.set_origin(np.array([0, ew.size[1]]))
    gm.gen_frame(ax_length=.2).pdndp.reparentTo(ew2.render)

    base.run()
