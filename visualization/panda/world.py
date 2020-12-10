from panda3d.core import PerspectiveLens, OrthographicLens, AmbientLight, PointLight, Vec4, Vec3, Point3, \
    WindowProperties, Filename, NodePath, Shader, GraphicsPipe, FrameBufferProperties, GraphicsOutput
from direct.showbase.ShowBase import ShowBase
import visualization.panda.inputmanager as im
# import visualization.panda.filter as flt
from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletDebugNode
import os
import math
from basis import dataadapter as p3dh
# from vision.pointcloud import o3dhelper as o3dh
import basis.robotmath as rm
import numpy as np


class World(ShowBase, object):

    def __init__(self, campos=np.array([2.0, 0.5, 2.0]), lookatpos=np.array([0, 0, 0.25]), up=np.array([0, 0, 1]),
                 fov=40, w=1024, h=768, lenstype="perspective", toggledebug=False, autocamrotate=False):
        """
        :param campos:
        :param lookatpos:
        :param fov:
        :param w: width of window
        :param h: height of window
        author: weiwei
        date: 2015?, 20201115
        """
        # the taskMgr, loader, render2d, etc. are added to builtin after initializing the showbase parental class
        super().__init__()
        self.disableAllAudio()
        self.setBackgroundColor(1, 1, 1)
        # set up lens
        lens = PerspectiveLens()
        lens.setFov(fov)
        lens.setNearFar(0.001, 50.0)
        if lenstype == "orthographic":
            lens = OrthographicLens()
            lens.setFilmSize(1024, 768)
        # disable the default mouse control
        self.disableMouse()
        self.cam.setPos(campos[0], campos[1], campos[2])
        self.cam.lookAt(Point3(lookatpos[0], lookatpos[1], lookatpos[2]), Vec3(up[0], up[1], up[2]))
        self.cam.node().setLens(lens)
        # set up slight
        ## ambient light
        ablight = AmbientLight("ambientlight")
        ablight.setColor(Vec4(0.2, 0.2, 0.2, 1))
        self.ablightnode = self.cam.attachNewNode(ablight)
        self.render.setLight(self.ablightnode)
        ## point light 1
        ptlight0 = PointLight("pointlight0")
        ptlight0.setColor(Vec4(1, 1, 1, 1))
        self._ptlightnode0 = self.cam.attachNewNode(ptlight0)
        self._ptlightnode0.setPos(0, 0, 0)
        self.render.setLight(self._ptlightnode0)
        ## point light 2
        ptlight1 = PointLight("pointlight1")
        ptlight1.setColor(Vec4(.4, .4, .4, 1))
        self._ptlightnode1 = self.cam.attachNewNode(ptlight1)
        self._ptlightnode1.setPos(self.cam.getPos().length(), 0, self.cam.getPos().length())
        self.render.setLight(self._ptlightnode1)
        ## point light 3
        ptlight2 = PointLight("pointlight2")
        ptlight2.setColor(Vec4(.3, .3, .3, 1))
        self._ptlightnode2 = self.cam.attachNewNode(ptlight2)
        self._ptlightnode2.setPos(-self.cam.getPos().length(), 0, self.cam.getPos().length())
        self.render.setLight(self._ptlightnode2)
        # helpers
        self.p3dh = p3dh
        # self.o3dh = o3dh
        self.rbtmath = rm
        # set up inputmanager
        self.lookatpos = lookatpos
        self.inputmgr = im.InputManager(self, self.lookatpos)
        taskMgr.add(self._interaction_update, "interaction", appendTask=True)
        # set up rotational cam
        if autocamrotate:
            taskMgr.doMethodLater(.1, self._rotatecam_update, "rotate cam")
        # set window size
        props = WindowProperties()
        props.setSize(w, h)
        self.win.requestProperties(props)
        # outline edge shader
        self.set_outlineshader()
        # set up cartoon effect
        # self._separation = .7
        # self.filter = flt.Filter(self.win, self.cam)
        # self.filter.setCartoonInk(separation=self._separation)
        # set up physics world
        self.physicsworld = BulletWorld()
        self.physicsworld.setGravity(Vec3(0, 0, -9.81))
        taskMgr.add(self._physics_update, "physics", appendTask=True)
        globalbprrender = base.render.attachNewNode("globalbpcollider")
        debugNode = BulletDebugNode('Debug')
        debugNode.showWireframe(True)
        debugNode.showConstraints(True)
        debugNode.showBoundingBoxes(False)
        debugNode.showNormals(True)
        self._debugNP = globalbprrender.attachNewNode(debugNode)
        self._debugNP.show()
        self.toggledebug = toggledebug
        if toggledebug:
            self.physicsworld.setDebugNode(self._debugNP.node())
        self.physicsbodylist = []
        # set up render update
        self._objtodraw = []  # the nodepath, collision model, or bullet dynamics model to be drawn
        taskMgr.add(self._render_update, "render", appendTask=True)

    def _interaction_update(self, task):
        # reset aspect ratio
        aspectRatio = self.getAspectRatio()
        self.cam.node().getLens().setAspectRatio(aspectRatio)
        self.inputmgr.check_mouse1drag()
        self.inputmgr.check_mouse2drag()
        self.inputmgr.check_mouse3click()
        self.inputmgr.check_mousewheel()
        self.inputmgr.check_resetcamera()
        return task.cont

    def _physics_update(self, task):
        dt = globalClock.getDt()
        self.physicsworld.doPhysics(dt, 20, 1 / 1200)
        return task.cont

    def _render_update(self, task):
        for otdele in self._objtodraw:
            otdele.detach()
            otdele.reparent_to(self.render)
        return task.cont

    def _rotatecam_update(self, task):
        campos = self.cam.getPos()
        camangle = math.atan2(campos[1] - self.lookatpos[1], campos[0] - self.lookatpos[0])
        # print camangle
        if camangle < 0:
            camangle += math.pi * 2
        if camangle >= math.pi * 2:
            camangle = 0
        else:
            camangle += math.pi / 360
        camradius = math.sqrt((campos[0] - self.lookatpos[0]) ** 2 + (campos[1] - self.lookatpos[1]) ** 2)
        camx = camradius * math.cos(camangle)
        camy = camradius * math.sin(camangle)
        self.cam.setPos(self.lookatpos[0] + camx, self.lookatpos[1] + camy, campos[2])
        self.cam.lookAt(self.lookatpos[0], self.lookatpos[1], self.lookatpos[2])
        return task.cont

    def change_debugstatus(self, toggledebug):
        if self.toggledebug == toggledebug:
            return
        elif toggledebug:
            self.physicsworld.setDebugNode(self._debugNP.node())
        else:
            self.physicsworld.clearDebugNode()
        self.toggledebug = toggledebug

    def attach_autoupdate_object(self, *args):
        """
        add to the render update list
        *args,**kwargs
        :param obj: nodepath, collision model, or bullet dynamics model
        :return:
        author: weiwei
        date: 20190627
        """
        for obj in args:
            self._objtodraw.append(obj)

    def detach_autoupdate_object(self, *args):
        """
        remove from the render update list
        :param obj: nodepath, collision model, or bullet dynamics model
        :return:
        author: weiwei
        date: 20190627
        """
        for obj in args:
            self.__objtodraw.remove(obj)

    def change_lookat(self, lookatpos):
        """
        This function is questionable
        as lookat changes the rotation of the camera
        :param lookatpos:
        :return:

        author: weiwei
        date: 20180606
        """
        self.cam.lookAt(lookatpos[0], lookatpos[1], lookatpos[2])
        self.inputmgr = im.InputManager(self, lookatpos, self.pggen)

    def set_cartoonshader(self, switchtoon=False):
        """
        set cartoon shader, the following program is a reference
        https://github.com/panda3d/panda3d/blob/master/samples/cartoon-shader/advanced.py
        :return:
        author: weiwei
        date: 20180601
        """
        this_dir, this_filename = os.path.split(__file__)
        if switchtoon:
            lightinggen = Filename.fromOsSpecific(os.path.join(this_dir, "shaders", "lighting_gen.sha"))
            tempnode = NodePath("temp")
            tempnode.setShader(loader.loadShader(lightinggen))
            self.cam.node().setInitialState(tempnode.getState())
            # self.render.setShaderInput("light", self.cam)
            self.render.setShaderInput("light", self._ablightnode)
        normalsBuffer = self.win.makeTextureBuffer("normalsBuffer", 0, 0)
        normalsBuffer.setClearColor(Vec4(0.5, 0.5, 0.5, 1))
        normalsCamera = self.makeCamera(normalsBuffer, lens=self.cam.node().getLens(), scene=self.render)
        normalsCamera.reparentTo(self.cam)
        normalgen = Filename.fromOsSpecific(os.path.join(this_dir, "shaders", "normal_gen.sha"))
        tempnode = NodePath("temp")
        tempnode.setShader(loader.loadShader(normalgen))
        normalsCamera.node().setInitialState(tempnode.getState())
        drawnScene = normalsBuffer.getTextureCard()
        drawnScene.setTransparency(1)
        drawnScene.setColor(1, 1, 1, 0)
        drawnScene.reparentTo(render2d)
        self.drawnScene = drawnScene
        self.separation = 0.001
        self.cutoff = 0.05
        inkGen = Filename.fromOsSpecific(os.path.join(this_dir, "shaders", "ink_gen.sha"))
        drawnScene.setShader(loader.loadShader(inkGen))
        drawnScene.setShaderInput("separation", Vec4(0, 0, self.separation, 0))
        drawnScene.setShaderInput("cutoff", Vec4(self.cutoff))

    def set_outlineshader(self):
        """
        document 1: https://qiita.com/nmxi/items/bfd10a3b3f519878e74e
        document 2: https://docs.panda3d.org/1.10/python/programming/shaders/list-of-cg-inputs
        :return:
        author: weiwei
        date: 20180601, 20201210osaka
        """
        depth_sha = """
        void vshader(float4 vtx_position : POSITION,
                     float4 vtx_normal : NORMAL,
                     uniform float4x4 mat_modelproj,
                     uniform float4x4 mat_modelview,
                     out float4 l_position : POSITION,
                     out float4 l_color0: COLOR0) {
            l_position = mul(mat_modelproj, vtx_position);
            float depth = l_position.z*.15;
            l_color0 = float4(depth, depth, depth, 1);
        }
        void fshader(float4 l_color0: COLOR0,
                     uniform sampler2D tex_0 : TEXUNIT0,
                     out float4 o_color : COLOR) {
            o_color = l_color0;
        }"""
        outline_sha = """
        void vshader(float4 vtx_position : POSITION,
             float2 vtx_texcoord0 : TEXCOORD0,
             uniform float4x4 mat_modelproj,
             out float4 l_position : POSITION,
             out float2 l_texcoord0 : TEXCOORD0)
        {
          l_position = mul(mat_modelproj, vtx_position);
          l_texcoord0 = vtx_texcoord0;
        }
        void fshader(float2 l_texcoord0 : TEXCOORD0,
                     uniform sampler2D tex_0 : TEXUNIT0,
                     uniform float2 sys_windowsize,
                     out float4 o_color : COLOR)
        {
          float sepx = 1/sys_windowsize.x;
          float sepy = 1/sys_windowsize.y;
          float4 color0 = tex2D(tex_0, l_texcoord0);
          float2 texcoord1 = l_texcoord0+float2(sepx, 0);
          float4 color1 = tex2D(tex_0, texcoord1);
          float2 texcoord2 = l_texcoord0+float2(0, sepy);
          float4 color2 = tex2D(tex_0, texcoord2);
          float2 texcoord3 = l_texcoord0+float2(-sepx, 0);
          float4 color3 = tex2D(tex_0, texcoord3);
          float2 texcoord4 = l_texcoord0+float2(0, -sepy);
          float4 color4 = tex2D(tex_0, texcoord4);
          float2 texcoord5 = l_texcoord0+float2(sepx, sepy);
          float4 color5 = tex2D(tex_0, texcoord5);
          float2 texcoord6 = l_texcoord0+float2(-sepx, -sepy);
          float4 color6 = tex2D(tex_0, texcoord6);
          float2 texcoord7 = l_texcoord0+float2(-sepx, sepy);
          float4 color7 = tex2D(tex_0, texcoord7);
          float2 texcoord8 = l_texcoord0+float2(sepx, -sepy);
          float4 color8 = tex2D(tex_0, texcoord8);
          float2 texcoord9 = l_texcoord0+float2(2*sepx, 0);
          float4 color9 = tex2D(tex_0, texcoord9);
          float2 texcoord10 = l_texcoord0+float2(-2*sepx, 0);
          float4 color10 = tex2D(tex_0, texcoord10);
          float2 texcoord11 = l_texcoord0+float2(0, 2*sepy);
          float4 color11 = tex2D(tex_0, texcoord11);
          float2 texcoord12 = l_texcoord0+float2(0, -2*sepy);
          float4 color12 = tex2D(tex_0, texcoord12);
          o_color = (color0-color1).x > .005 || (color0-color2).x > .005 || (color0-color3).x > .005 ||
                    (color0-color4).x > .005 || (color0-color5).x > .005 || (color0-color6).x > .005 ||
                    (color0-color7).x > .005 || (color0-color8).x > .005 || (color0-color9).x > .005 ||
                    (color0-color10).x > .005 || (color0-color11).x > .005 || (color0-color12).x > .005 ?
                    float4(0, 0, 0, 1) : float4(0, 0, 0, 0);
        }"""
        depthBuffer = self.win.makeTextureBuffer("depthBuffer", 0, 0)
        depthBuffer.setClearColor(Vec4(1, 1, 1, 1))
        depthCamera = self.makeCamera(depthBuffer, lens=self.cam.node().getLens(), scene=self.render)
        depthCamera.reparentTo(self.cam)
        tempnode = NodePath("depth")
        tempnode.setShader(Shader.make(depth_sha, Shader.SL_Cg))
        depthCamera.node().setInitialState(tempnode.getState())
        drawnScene = depthBuffer.getTextureCard()
        drawnScene.setTransparency(1)
        drawnScene.setColor(1, 1, 1, 0)
        drawnScene.reparentTo(render2d)
        drawnScene.setShader(Shader.make(outline_sha, Shader.SL_Cg))
