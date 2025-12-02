"""
Panda3d Extension
Author: Hao Chen
"""
import functools
from typing import Optional

import numpy as np
import cv2
from panda3d.core import (
    Texture, NodePath, WindowProperties,
    Vec3, Point3, PerspectiveLens,
    OrthographicLens, PGTop, CardMaker,
    FrameBufferProperties, GraphicsPipe, GraphicsOutput,
    Shader, ShaderAttrib, RenderState,
    TextureAttrib, LightAttrib, ColorAttrib,
)
from direct.gui.OnscreenImage import OnscreenImage
from pyparsing import Literal

import wrs.basis.robot_math as rm
import wrs.basis.data_adapter as da
import wrs.visualization.panda.filter as flt
import wrs.visualization.panda.inputmanager as im
import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as mgm
import wrs.modeling.model_collection as mmc


class VirtualCamera(object):

    def __init__(self,
                 cam_pos=np.array([2.0, 0, 2.0]),
                 lookat_pos=np.array([0, 0, 0.25]),
                 resolution=(512, 512),
                 screen_size=np.array((0.16, 0.16)),
                 fov=40,
                 lens_type="perspective",
                 w_base=None):
        """
        :param resolution: (width, height)
        :param cam_pos: np.array [x, y, z]
        :param lookat_pos: np.array [x, y, z]
        :param up: np.array [x, y, z]
        :param fov: field of view in degrees
        :param lens_type: "perspective" or "orthographic"
        :param w_base: The ShowBase instance (defaults to global 'base' if None)
        """
        self._base = base if w_base is None else w_base
        self.resolution = resolution
        self.screen_size = screen_size  # Default ref size for visualizer
        self._tex = Texture()
        self._tex.setWrapU(Texture.WMClamp)
        self._tex.setWrapV(Texture.WMClamp)
        # self._tex.setMinfilter(Texture.FTLinear)
        # self._tex.setMagfilter(Texture.FTLinear)
        self._buffer = self._base.win.makeTextureBuffer("virtual_cam_buffer",
                                                        resolution[0],
                                                        resolution[1],
                                                        self._tex,
                                                        True)
        self._buffer.setClearColor((1, 1, 1, 1))
        self._buffer.setSort(-3)  # Render before main window
        # Setup Lens
        if lens_type == "orthographic":
            lens = OrthographicLens()
            lens.setFilmSize(resolution[0] / 1000.0, resolution[1] / 1000.0)  # Approx scale
        else:
            lens = PerspectiveLens()
            lens.setFov(fov)
            lens.setNearFar(0.001, 5000.0)
        # Match aspect ratio to resolution
        lens.setAspectRatio(float(resolution[0]) / float(resolution[1]))
        # Setup Camera Node
        self._pdnp = self._base.makeCamera(self._buffer,
                                           camName="virtual_cam",
                                           lens=lens)
        # Get Display Region (Crucial for getting the image later)
        self._dr = self._pdnp.node().getDisplayRegion(0)
        # Initial Pose
        self.cam_pos = cam_pos
        self.look_at(lookat_pos, )  # cam look at 0,1,0 by default in panda3d

    @property
    def cam_pos(self):
        return np.array([*self._pdnp.getPos()])

    @property
    def cam_rotmat(self):
        return np.array(self._pdnp.getMat().getUpper3()).T

    @property
    def cam_homomat(self):
        return da.pdmat4_to_npmat4(self._pdnp.getMat())

    @cam_pos.setter
    def cam_pos(self, pos):
        """
        :param pos: np.array [x, y, z]
        """
        self._pdnp.setPos(Point3(pos[0], pos[1], pos[2]))

    @cam_rotmat.setter
    def cam_rotmat(self, rotmat):
        """
        :param rotmat: 3x3 rotation matrix
        """
        homomat = rm.homomat_from_posrot(self.cam_pos, rotmat)
        self._set_homomat(homomat)

    @cam_homomat.setter
    def cam_homomat(self, np_mat):
        self._set_homomat(np_mat)

    @property
    def intrinsics(self):
        """
        Calculates and returns the 3x3 Intrinsic Matrix (K) for a Perspective Lens.
        :return: 3x3 numpy array
        """
        lens = self._pdnp.node().getLens()
        if not isinstance(lens, PerspectiveLens):
            print("Warning: Intrinsic matrix calculation assumes PerspectiveLens.")
            return np.eye(3)
        # Get Field of View (degrees)
        # getFov() returns a LVecBase2(horizontal_fov, vertical_fov)
        fov = lens.getFov()
        h_fov = fov[0]
        v_fov = fov[1]
        # Get Resolution (Pixels)
        w, h = self.resolution
        # Calculate Focal Lengths
        # Formula: f = (dim / 2) / tan(fov / 2)
        # We convert degrees to radians for numpy
        f_x = (w / 2.0) / np.tan(np.deg2rad(h_fov) / 2.0)
        f_y = (h / 2.0) / np.tan(np.deg2rad(v_fov) / 2.0)
        # Calculate Principal Point (Center of the image)
        c_x = w / 2.0
        c_y = h / 2.0
        # Construct Matrix
        K = np.array([[f_x, 0.0, c_x],
                      [0.0, f_y, c_y],
                      [0.0, 0.0, 1.0]])
        return K

    def _set_homomat(self, np_mat):
        """
        Sets the camera pose using a 4x4 homogenous matrix (numpy).
        :param np_mat: 4x4 numpy array [[R, t], [0, 1]]
        """
        # Convert to Panda3D matrix (handles transpose/row-major conversion)
        p3d_mat = da.npmat4_to_pdmat4(np_mat)
        self._pdnp.setMat(p3d_mat)

    def look_at(self, target_pos, up=np.array([0, 0, 1])):
        """
        Sets the camera to look at a target position.
        :param target_pos: np.array [x, y, z]
        :param up: np.array [x, y, z]
        """
        self._pdnp.look_at(Point3(target_pos[0], target_pos[1], target_pos[2]),
                           Vec3(up[0], up[1], up[2]))

    def get_image(self, requested_format=None):
        """
        Returns the camera's image as a numpy array (uint8, 0-255).
        :param requested_format: e.g., "RGBA", "BGRA". Default is Panda internal (BGRA).
        :return: numpy array of shape (height, width, channels)
        """
        # Force a render frame to ensure buffer is updated
        self._base.graphicsEngine.renderFrame()

        tex = self._dr.getScreenshot()
        if requested_format is None:
            data = tex.get_ram_image()
        else:
            data = tex.get_ram_image_as(requested_format)

        image = np.frombuffer(data, np.uint8)
        image.shape = (tex.getYSize(), tex.getXSize(), tex.getNumComponents())
        # Panda3D images are upside down in memory relative to standard CV2/Matplotlib
        image = np.flipud(image)
        return image.copy()

    def gen_framemodel(self, name="virtual_cam_frame_model", alpha=1):
        """
        Generates a wireframe visualization of the camera.
        """
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
        """
        Generates a transparent mesh visualization of the camera.
        """
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

    def remove(self):
        self._base.graphicsEngine.removeWindow(self._buffer)
        self._pdnp.removeNode()


class VirtualDepthCamera(VirtualCamera):
    DEPTH_SHADER_VERT = """
    #version 150
    uniform mat4 p3d_ProjectionMatrix;
    uniform mat4 p3d_ModelViewMatrix;
    in vec4 p3d_Vertex;
    out float depthCam;
    void main() {
      vec4 cs_position = p3d_ModelViewMatrix * p3d_Vertex;
      depthCam = -cs_position.z;
      gl_Position = p3d_ProjectionMatrix * cs_position;
    }
    """
    DEPTH_SHADER_FRAG = """
    #version 150
    in float depthCam;
    out vec4 fragColor;
    void main() {
      fragColor = vec4(depthCam, 0, 0, 1);
    }
    """

    def __init__(self,
                 cam_pos=np.array([2.0, 0, 2.0]),
                 lookat_pos=np.array([0, 0, 0.25]),
                 resolution=(512, 512),
                 screen_size=np.array((0.16, 0.16)),
                 fov=40,
                 lens_type="perspective",
                 w_base=None,
                 depth_far=10.0):  # Max distance for reliable depth
        """
        An RGB-D Camera that inherits from VirtualCamera.
        The Depth camera is attached as a child to the RGB camera, so they share extrinsics.
        """
        # 1. Initialize the Base RGB Camera
        super().__init__(cam_pos=cam_pos,
                         lookat_pos=lookat_pos,
                         resolution=resolution,
                         screen_size=screen_size,
                         fov=fov,
                         lens_type=lens_type,
                         w_base=w_base)

        self.depth_far = depth_far

        # 2. Setup Depth Buffer & Camera
        self._depth_tex = Texture()
        self._depth_buffer = self._create_depth_buffer(resolution, self._depth_tex)

        # 3. Setup Depth Camera Node
        # We share the same Lens settings as the RGB camera
        depth_lens = PerspectiveLens()
        depth_lens.setFov(self._pdnp.node().getLens().getFov())
        depth_lens.setAspectRatio(self._pdnp.node().getLens().getAspectRatio())
        depth_lens.setNearFar(0.001, self.depth_far + 10.0)

        self._depth_cam_np = self._base.makeCamera(self._depth_buffer,
                                                   camName="virtual_depth_cam",
                                                   lens=depth_lens)

        # CRITICAL: Reparent depth cam to the main RGB cam.
        # Now, moving 'self' (the RGB cam) automatically moves the depth cam.
        self._depth_cam_np.reparentTo(self._pdnp)

        # 4. Apply Depth Shader
        # We create a RenderState that applies the shader to everything this camera sees.
        self._depth_scene = NodePath("DepthCanvas")
        shader = Shader.make(
            Shader.SL_GLSL,
            vertex=self.DEPTH_SHADER_VERT,
            fragment=self.DEPTH_SHADER_FRAG
        )
        # state = RenderState.makeEmpty()
        state = RenderState.make(ShaderAttrib.make(shader))
        # state = state.addAttrib(ShaderAttrib.make(shader), 1)
        # state = state.addAttrib(TextureAttrib.makeOff(), 1)
        # state = state.addAttrib(LightAttrib.makeAllOff(), 1)
        # state = state.addAttrib(ColorAttrib.makeFlat((1, 1, 1, 1)), 1)
        self._depth_cam_np.node().setInitialState(state)

        # 5. Ensure the depth camera sees the same scene as the RGB camera
        # Usually 'base.render'. Assuming the RGB cam looks at base.render via makeCamera default.
        self._depth_cam_np.node().setScene(self._base.render)

    def _create_depth_buffer(self, resolution, texture):
        """
        Creates an offscreen buffer specifically for high-precision depth.
        """
        window_props = WindowProperties(size=(resolution[0], resolution[1]))
        fb_props = FrameBufferProperties()
        fb_props.setFloatColor(True)
        fb_props.setRgbaBits(32, 0, 0, 0)  # We only need one channel (Red) for depth
        fb_props.setDepthBits(24)
        buffer = self._base.graphicsEngine.makeOutput(
            self._base.pipe,
            "depth_buffer",
            -2,  # Sort order
            fb_props,
            window_props,
            GraphicsPipe.BFRefuseWindow,
            self._base.win.getGsg(),
            self._base.win
        )
        buffer.addRenderTexture(texture, GraphicsOutput.RTMCopyRam)
        buffer.setClearColorActive(True)
        buffer.setClearColor((self.depth_far, 0, 0, 0))  # Clear to max depth
        buffer.setClearDepthActive(True)
        buffer.setClearDepth(1.0)
        return buffer

    def get_rgb_image(self, requested_format=None):
        """ Alias for super().get_image to be explicit """
        return self.get_image(requested_format)

    def get_depth_image(self):
        self._base.graphicsEngine.renderFrame()
        data = self._depth_tex.getRamImage()
        depth_image = np.frombuffer(data, np.float32).copy()
        depth_image.shape = (self._depth_tex.getYSize(), self._depth_tex.getXSize(), self._depth_tex.getNumComponents())
        depth_image = np.flipud(depth_image)
        return depth_image[:, :, 0]

    def _get_xyz_world(self, depth_img=None):
        """ Internal helper to generate XYZ in world coordinates without filtering """
        if depth_img is None:
            depth_img = self.get_depth_image()

        H, W = depth_img.shape
        K = self.intrinsics

        # 1. Meshgrid
        u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)

        # 2. Back-project to Camera Space
        Z = depth_img
        X = (u - K[0, 2]) * Z / K[0, 0]
        Y = (v - K[1, 2]) * Z / K[1, 1]

        # Stack (N, 3) - Camera Coordinates (CV Standard: Y-down, Z-forward)
        points_cam = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

        # 3. Transform to World Space (Panda3D)
        # Convert Intrinsic Space -> Panda Node Space
        # P3D_X=X, P3D_Y=Z, P3D_Z=-Y
        points_p3d_local = np.zeros_like(points_cam)
        points_p3d_local[:, 0] = points_cam[:, 0]
        points_p3d_local[:, 1] = points_cam[:, 2]
        points_p3d_local[:, 2] = -points_cam[:, 1]

        # Transform Local -> World
        cam_to_world = self.cam_homomat
        R = cam_to_world[:3, :3]
        t = cam_to_world[:3, 3]

        points_world = points_p3d_local @ R.T + t
        return points_world

    def get_point_cloud(self, depth_img=None, filter_zeros=True):
        points_world = self._get_xyz_world(depth_img)

        if filter_zeros:
            if depth_img is None: depth_img = self.get_depth_image()
            z_flat = depth_img.flatten()
            valid_mask = (z_flat > 0) & (z_flat < self.depth_far)
            points_world = points_world[valid_mask]

        return points_world

    def get_colored_point_cloud(self):
        """
        Returns a colored point cloud.
        :return: np.array of shape (N, 6).
                 Columns are [x, y, z, r, g, b].
                 RGB values are normalized 0..1.
        """
        # 1. Get Images
        rgb_img = self.get_rgb_image()  # H, W, 3 (BGR or RGB)
        depth_img = self.get_depth_image()  # H, W

        # 2. Get Geometry (N, 3)
        xyz = self._get_xyz_world(depth_img)

        # 3. Flatten RGB to match Geometry (N, 3)
        # Ensure we are working with standard RGB (Panda usually returns BGRA/BGR)
        # Assuming get_image returns BGR (standard OpenCV style in Panda)
        # Convert BGR -> RGB
        rgb_flat = rgb_img.reshape(-1, 3)
        rgb_flat = rgb_flat[:, [2, 1, 0]]  # BGR to RGB

        # Normalize colors to 0..1 for standard point cloud viewers (Open3D etc)
        rgb_norm = rgb_flat.astype(np.float32) / 255.0

        # 4. Filter Invalid Depth
        z_flat = depth_img.flatten()
        valid_mask = (z_flat > 0.05) & (z_flat < self.depth_far)

        xyz_valid = xyz[valid_mask]
        rgb_valid = rgb_norm[valid_mask]

        # 5. Stack to (N, 6) -> [x, y, z, r, g, b]
        pcd_colored = np.hstack((xyz_valid, rgb_valid))

        # If you strictly need 6xN, return pcd_colored.T
        return pcd_colored

    def remove(self):
        """ Clean up depth specific buffers then call parent clean up """
        self._base.graphicsEngine.removeWindow(self._depth_buffer)
        self._depth_cam_np.removeNode()
        super().remove()


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
    mgm.gen_frame(ax_length=.2).attach_to(base)

    # extra window 1
    ew = ExtraWindow(base, cam_pos=np.array([2, 0, 1.5]), lookat_pos=np.array([0, 0, .2]))
    ew.set_origin((np.array([0, 40])))

    # ImgOnscreen()
    img = cv2.imread("../../../wrs_logo_2022.jpg")
    on_screen_img = ImgOnscreen(img.shape[:2][::-1], parent_np=ew)
    on_screen_img.update_img(img)

    # extra window 2
    ew2 = ExtraWindow(base, cam_pos=np.array([2, 0, 1.5]), lookat_pos=np.array([0, 0, .2]))
    ew2.set_origin(np.array([0, ew.size[1]]))
    mgm.gen_frame(ax_length=.2).pdndp.reparentTo(ew2.render)

    base.run()
