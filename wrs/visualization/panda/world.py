from panda3d.core import PerspectiveLens, OrthographicLens, AmbientLight, PointLight, Vec4, Vec3, Point3, WindowProperties
from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
import wrs.visualization.panda.inputmanager as im
import wrs.visualization.panda.filter as flt
import wrs.basis.robot_math as rm
import wrs.basis.data_adapter as da
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
                 cam_pos=rm.np.array([2.0, 0.5, 2.0]),
                 lookat_pos=rm.np.array([0, 0, 0.25]),
                 up=rm.np.array([0, 0, 1]),
                 fov=40,
                 w=1920,
                 h=1080,
                 lens_type=LensType.PERSPECTIVE,
                 auto_rotate=False):
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
        # set up inputmanager
        self.lookat_pos = lookat_pos
        self.inputmgr = im.InputManager(self, self.lookat_pos, toggle_rotcenter=True)
        taskMgr.add(self._interaction_update, "interaction", appendTask=True)
        # set up rotational cam
        if auto_rotate:
            taskMgr.doMethodLater(.1, self._rotatecam_update, "rotate cam")
        # set window size
        props = WindowProperties()
        props.setSize(w, h)
        self.win.requestProperties(props)
        # set up cartoon filter
        self._separation = 1
        self.filter = flt.Filter(self.win, self.cam)
        self.filter.setCartoonInk(separation=self._separation)
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
        camangle = rm.atan2(campos[1] - self.lookat_pos[1], campos[0] - self.lookat_pos[0])
        # print camangle
        if camangle < 0:
            camangle += rm.pi * 2
        if camangle >= rm.pi * 2:
            camangle = 0
        else:
            camangle += rm.pi / 360
        camradius = rm.sqrt((campos[0] - self.lookat_pos[0]) ** 2 + (campos[1] - self.lookat_pos[1]) ** 2)
        camx = camradius * rm.cos(camangle)
        camy = camradius * rm.sin(camangle)
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

    def show_text(self, text, pos=(0, 0), scale=0.05, color=(0, 0, 0, 1)):
        return OnscreenText(text=text, pos=(pos[0], pos[1]), scale=scale, fg=color)

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