from panda3d.core import PerspectiveLens, OrthographicLens, AmbientLight, PointLight, Vec4, Vec3, Point3, \
    WindowProperties, Filename, NodePath, Shader
from direct.showbase.ShowBase import ShowBase
import wrs.visualization.panda.inputmanager as im
import wrs.visualization.panda.filter as flt
import os
import math
from wrs.basis import data_adapter as p3dh
# from vision.pointcloud import o3dhelper as o3dh
from wrs import basis as rm
import numpy as np
from enum import Enum
try:
    import mujoco
except:
    mujoco = None


class LensType(Enum):
    PERSPECTIVE = 1
    ORTHOGRAPHIC = 2


class World(ShowBase, object):

    def __init__(self,
                 cam_pos=np.array([2.0, 0.5, 2.0]),
                 lookat_pos=np.array([0, 0, 0.25]),
                 up=np.array([0, 0, 1]),
                 fov=40,
                 w=1920,
                 h=1080,
                 lens_type=LensType.PERSPECTIVE,
                 toggle_debug=False,
                 auto_cam_rotate=False):
        """
        :param cam_pos:
        :param lookat_pos:
        :param fov:
        :param w: width of window
        :param h: height of window
        author: weiwei
        date: 20150520, 20201115, 20240527
        """
        # the taskMgr, loader, render2d, etc. are added to builtin after initializing the showbase parental class
        super().__init__()
        # set up window
        winprops = WindowProperties(base.win.getProperties())
        winprops.setTitle("WRS Robot Planning and Control System")
        base.win.requestProperties(winprops)
        self.disableAllAudio()
        self.setBackgroundColor(1, 1, 1)
        # set up lens
        lens = PerspectiveLens()
        lens.setFov(fov)
        lens.setNearFar(0.001, 5000.0)
        if lens_type == LensType.ORTHOGRAPHIC:
            lens = OrthographicLens()
            lens.setFilmSize(640, 480)
        # disable the default mouse control
        self.disableMouse()
        self.cam.setPos(cam_pos[0], cam_pos[1], cam_pos[2])
        self.cam.lookAt(Point3(lookat_pos[0], lookat_pos[1], lookat_pos[2]), Vec3(up[0], up[1], up[2]))
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
        self.lookat_pos = lookat_pos
        self.inputmgr = im.InputManager(self, self.lookat_pos)
        taskMgr.add(self._interaction_update, "interaction", appendTask=True)
        # set up rotational cam
        if auto_cam_rotate:
            taskMgr.doMethodLater(.1, self._rotatecam_update, "rotate cam")
        # set window size
        props = WindowProperties()
        props.setSize(w, h)
        self.win.requestProperties(props)
        # # outline edge shader
        # self.set_outlineshader()
        # set up cartoon effect
        self._separation = 1
        self.filter = flt.Filter(self.win, self.cam)
        self.filter.setCartoonInk(separation=self._separation)
        # self.filter.setViewGlow()
        # # set up physics world
        # self.physics_scale=1e3
        # self.physicsworld = BulletWorld()
        # self.physicsworld.setGravity(Vec3(0, 0, -9.81*self.physics_scale))
        # taskMgr.add(self._physics_update, "physics", appendTask=True)
        # globalbprrender = base.render.attachNewNode("globalbpcollider")
        # debugNode = BulletDebugNode('Debug')
        # debugNode.showWireframe(True)
        # debugNode.showConstraints(True)
        # debugNode.showBoundingBoxes(False)
        # debugNode.showNormals(True)
        # self._debugNP = globalbprrender.attachNewNode(debugNode)
        # self._debugNP.show()
        # self.toggledebug = toggle_debug
        # if toggle_debug:
        #     self.physicsworld.setDebugNode(self._debugNP.node())
        # self.physicsbodylist = []
        # # set up render update (TODO, only for dynamics?)
        # self._internal_update_obj_list = []  # the pdndp, collision model, or bullet dynamics model to be drawn
        # self._internal_update_robot_list = []
        # taskMgr.add(self._internal_update, "internal_update", appendTask=True)

        # for remote visualization
        self._external_update_objinfo_list = []  # see anime_info.py
        self._external_update_robotinfo_list = []
        taskMgr.add(self._external_update, "external_update", appendTask=True)
        # for stationary models
        self._noupdate_model_list = []

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

    def _mj_physics_update(self, mj_model, duration, task):
        elapsed_time = task.time
        if elapsed_time > duration:
            # for geom_dict in mj_model.body_geom_dict.values():
            #   for geom in geom_dict.values():
            #       geom.attach_to(self)
            return task.done
        # time_since_last_update = globalClock.getDt()
        # mj_start_time = mj_model.data.time
        # while mj_model.data.time - mj_start_time < time_since_last_update:
        #     mujoco.mj_step(mj_model.model, mj_model.data)
        # update
        mujoco.mj_step(mj_model.model, mj_model.data)
        if mj_model.control_callback is not None:
            mj_model.control_callback(mj_model)
        for geom_dict in mj_model.body_geom_dict.values():
            for key, geom in geom_dict.items():
                geom.detach()
                pos = mj_model.data.geom_xpos[key]
                rotmat = mj_model.data.geom_xmat[key].reshape(3, 3)
                geom.pose = [pos, rotmat]
                # print(pos)
                geom.attach_to(self)
        return task.cont

    # def _internal_update(self, task):
    #     for robot in self._internal_update_robot_list:
    #         robot.detach()  # TODO gen mesh model?
    #         robot.attach_to(self)
    #     for obj in self._internal_update_obj_list:
    #         obj.detach()
    #         obj.attach_to(self)
    #     return task.cont

    def _rotatecam_update(self, task):
        campos = self.cam.getPos()
        camangle = math.atan2(campos[1] - self.lookat_pos[1], campos[0] - self.lookat_pos[0])
        # print camangle
        if camangle < 0:
            camangle += math.pi * 2
        if camangle >= math.pi * 2:
            camangle = 0
        else:
            camangle += math.pi / 360
        camradius = math.sqrt((campos[0] - self.lookat_pos[0]) ** 2 + (campos[1] - self.lookat_pos[1]) ** 2)
        camx = camradius * math.cos(camangle)
        camy = camradius * math.sin(camangle)
        self.cam.setPos(self.lookat_pos[0] + camx, self.lookat_pos[1] + camy, campos[2])
        self.cam.lookAt(self.lookat_pos[0], self.lookat_pos[1], self.lookat_pos[2])
        return task.cont

    def _external_update(self, task):
        for _external_update_robotinfo in self._external_update_robotinfo_list:
            robot_s = _external_update_robotinfo.robot_s
            robot_component_name = _external_update_robotinfo.robot_component_name
            robot_meshmodel = _external_update_robotinfo.robot_meshmodel
            robot_meshmodel_parameter = _external_update_robotinfo.robot_meshmodel_parameters
            robot_path = _external_update_robotinfo.robot_path
            robot_path_counter = _external_update_robotinfo.robot_path_counter
            robot_meshmodel.detach()
            robot_s.fk(component_name=robot_component_name, joint_values=robot_path[robot_path_counter])
            _external_update_robotinfo.robot_meshmodel = robot_s.gen_mesh_model(
                tcp_jntid=robot_meshmodel_parameter[0],
                tcp_loc_pos=robot_meshmodel_parameter[1],
                tcp_loc_rotmat=robot_meshmodel_parameter[2],
                toggle_tcpcs=robot_meshmodel_parameter[3],
                toggle_jntscs=robot_meshmodel_parameter[4],
                rgba=robot_meshmodel_parameter[5],
                name=robot_meshmodel_parameter[6])
            _external_update_robotinfo.robot_meshmodel.attach_to(self)
            _external_update_robotinfo.robot_path_counter += 1
            if _external_update_robotinfo.robot_path_counter >= len(robot_path):
                _external_update_robotinfo.robot_path_counter = 0
        for _external_update_objinfo in self._external_update_objinfo_list:
            obj = _external_update_objinfo.obj
            obj_parameters = _external_update_objinfo.obj_parameters
            obj_path = _external_update_objinfo.obj_path
            obj_path_counter = _external_update_objinfo.obj_path_counter
            obj.detach()
            obj.set_pos(obj_path[obj_path_counter][0])
            obj.set_rotmat(obj_path[obj_path_counter][1])
            obj.set_rgba(obj_parameters[0])
            obj.attach_to(self)
            _external_update_objinfo.obj_path_counter += 1
            if _external_update_objinfo.obj_path_counter >= len(obj_path):
                _external_update_objinfo.obj_path_counter = 0
        return task.cont

    # def change_debug_status(self, toggledebug):
    #     if self.toggledebug == toggledebug:
    #         return
    #     elif toggledebug:
    #         self.physicsworld.setDebugNode(self._debugNP.node())
    #     else:
    #         self.physicsworld.clearDebugNode()
    #     self.toggledebug = toggledebug

    # def attach_internal_update_obj(self, obj):
    #     """
    #     :param obj: CollisionModel or (Static)GeometricModel
    #     :return:
    #     """
    #     self._internal_update_obj_list.append(obj)

    # def detach_internal_update_obj(self, obj):
    #     self._internal_update_obj_list.remove(obj)
    #     obj.detach()

    # def clear_internal_update_obj(self):
    #     tmp_internal_update_obj_list = self._internal_update_obj_list.copy()
    #     self._internal_update_obj_list = []
    #     for obj in tmp_internal_update_obj_list:
    #         obj.detach()

    # def attach_internal_update_robot(self, robot_meshmodel):  # TODO robot_meshmodel or robot_s?
    #     self._internal_update_robot_list.append(robot_meshmodel)
    #
    # def detach_internal_update_robot(self, robot_meshmodel):
    #     tmp_internal_update_robot_list = self._internal_update_robot_list.copy()
    #     self._internal_update_robot_list = []
    #     for robot in tmp_internal_update_robot_list:
    #         robot.detach()
    #
    # def clear_internal_update_robot(self):
    #     for robot in self._internal_update_robot_list:
    #         self.detach_internal_update_robot(robot)

    def run_mj_physics(self, mj_model, duration):
        for geom_dict in mj_model.body_geom_dict.values():
            for geom in geom_dict.values():
                geom.attach_to(self)
        self.taskMgr.add(self._mj_physics_update, extraArgs=[mj_model, duration], name="mj_physics", appendTask=True)
        # for geom_dict in mj_model.body_geom_dict.values():
        #     for geom in geom_dict.values():
        #         geom.detach()

    def attach_external_update_obj(self, objinfo):
        """
        :param objinfo: anime_info.ObjInfo
        :return:
        """
        self._external_update_objinfo_list.append(objinfo)

    def detach_external_update_obj(self, obj_info):
        self._external_update_objinfo_list.remove(obj_info)
        obj_info.obj.detach()

    def clear_external_update_obj(self):
        for obj in self._external_update_objinfo_list:
            self.detach_external_update_obj(obj)

    def attach_external_update_robot(self, robotinfo):
        """
        :param robotinfo: anime_info.RobotInfo
        :return:
        """
        self._external_update_robotinfo_list.append(robotinfo)

    def detach_external_update_robot(self, robot_info):
        self._external_update_robotinfo_list.remove(robot_info)
        robot_info.robot_meshmodel.detach()

    def clear_external_update_robot(self):
        for robot in self._external_update_robotinfo_list:
            self.detach_external_update_robot(robot)

    def attach_noupdate_model(self, model):
        model.attach_to(self)
        self._noupdate_model_list.append(model)

    def detach_noupdate_model(self, model):
        model.detach()
        self._noupdate_model_list.remove(model)

    def clear_noupdate_model(self):
        for model in self._noupdate_model_list:
            model.detach()
        self._noupdate_model_list = []

    def change_campos(self, campos):
        self.cam.setPos(campos[0], campos[1], campos[2])
        self.inputmgr = im.InputManager(self, self.lookat_pos)

    def change_lookatpos(self, lookatpos):
        """
        This function is questionable
        as lookat changes the rotation of the camera
        :param lookatpos:
        :return:
        author: weiwei
        date: 20180606
        """
        self.cam.lookAt(lookatpos[0], lookatpos[1], lookatpos[2])
        self.lookat_pos = lookatpos
        self.inputmgr = im.InputManager(self, self.lookat_pos)

    def change_campos_and_lookat_pos(self, cam_pos, lookat_pos):
        self.cam.setPos(cam_pos[0], cam_pos[1], cam_pos[2])
        self.cam.lookAt(lookat_pos[0], lookat_pos[1], lookat_pos[2])
        self.lookat_pos = lookat_pos
        self.inputmgr = im.InputManager(self, self.lookat_pos)

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
            float depth = l_position.a*.1;
            //l_color0 = vtx_position + float4(depth, depth, depth, 1);
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
        drawnScene.reparentTo(render2d)
        drawnScene.setTransparency(1)
        drawnScene.setColor(1, 1, 1, 0)
        drawnScene.setShader(Shader.make(outline_sha, Shader.SL_Cg))
