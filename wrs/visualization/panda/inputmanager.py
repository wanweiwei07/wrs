from direct.showbase.DirectObject import DirectObject
from panda3d.core import Vec3, Mat3, Mat4, CollisionNode, CollisionRay, BitMask32, CollisionSphere, Plane, \
    CollisionPlane, CollisionBox, Point3, CollisionTraverser, CollisionHandlerQueue, GeomNode
import numpy as np


class InputManager(DirectObject):

    def __init__(self, base, lookat_pos, togglerotcenter=False):
        self.base = base
        self.originallookatpos = lookat_pos  # for backup
        self.lookatpos_pdv3 = Vec3(lookat_pos[0], lookat_pos[1], lookat_pos[2])
        self.cam2lookatpos_dist = (self.base.cam.getPos() - self.lookatpos_pdv3).length()
        self.initviewdist = (self.base.cam.getPos() - self.lookatpos_pdv3).length()
        self.last_m1_pos = None
        self.last_m2_pos = None
        # toggle on the following part to explicitly show the rotation center
        self.togglerotcenter = togglerotcenter
        if self.togglerotcenter:
            self.rotatecenternp = self.base.p3dh.gensphere(pos=self.originallookatpos, radius=5,
                                                           rgba=np.array([1, 1, 0, 1]))
            self.rotatecenternp.reparentTo(self.base.render)
        # for resetting
        self.original_cam_pdmat4 = Mat4(self.base.cam.getMat())
        self.keymap = {"mouse1": False,
                       "mouse2": False,
                       "mouse3": False,
                       "wheel_up": False,
                       "wheel_down": False,
                       "space": False,
                       "w": False,
                       "s": False,
                       "a": False,
                       "d": False,
                       "g": False,
                       "r": False}
        self.accept("mouse1", self.__setkeys, ["mouse1", True])
        self.accept("mouse1-up", self.__setkeys, ["mouse1", False])
        self.accept("mouse2", self.__setkeys, ["mouse2", True])
        self.accept("mouse2-up", self.__setkeys, ["mouse2", False])
        self.accept("mouse3", self.__setkeys, ["mouse3", True])
        self.accept("mouse3-up", self.__setkeys, ["mouse3", False])
        self.accept("wheel_up", self.__setkeys, ["wheel_up", True])
        self.accept("wheel_down", self.__setkeys, ["wheel_down", True])
        self.accept("space", self.__setkeys, ["space", True])
        self.accept("space-up", self.__setkeys, ["space", False])
        self.accept("w", self.__setkeys, ["w", True])
        self.accept("w-up", self.__setkeys, ["w", False])
        self.accept("s", self.__setkeys, ["s", True])
        self.accept("s-up", self.__setkeys, ["s", False])
        self.accept("a", self.__setkeys, ["a", True])
        self.accept("a-up", self.__setkeys, ["a", False])
        self.accept("d", self.__setkeys, ["d", True])
        self.accept("d-up", self.__setkeys, ["d", False])
        self.accept("g", self.__setkeys, ["g", True])
        self.accept("g-up", self.__setkeys, ["g", False])
        self.accept("r", self.__setkeys, ["r", True])
        self.accept("r-up", self.__setkeys, ["r", False])
        self.setup_interactiongeometries()

    def __setkeys(self, key, value):
        self.keymap[key] = value
        return

    def setup_interactiongeometries(self):
        """
        set up collision rays, spheres, and planes for mouse manipulation
        :return: None
        author: weiwei
        date: 20161110
        """
        # create a trackball ray and set its bitmask to 8
        # the trackball ray must be a subnode of cam since we will
        # transform the clicked point (in the view of the cam) to the world coordinate system
        # using the ray
        self.tracker_cn = CollisionNode("tracker")
        self.tracker_ray = CollisionRay()
        self.tracker_cn.addSolid(self.tracker_ray)
        self.tracker_cn.setFromCollideMask(BitMask32.bit(8))
        self.tracker_cn.setIntoCollideMask(BitMask32.allOff())
        self.tracker_np = self.base.cam.attachNewNode(self.tracker_cn)
        # create an inverted collision sphere and puts it into a collision node
        # its bitmask is set to 8, and it will be the only collidable object at bit 8
        self.trackball_cn = CollisionNode("trackball")
        self.trackball_cn.addSolid(
            CollisionSphere(self.lookatpos_pdv3[0], self.lookatpos_pdv3[1], self.lookatpos_pdv3[2], self.cam2lookatpos_dist))
        self.trackball_cn.setFromCollideMask(BitMask32.allOff())
        self.trackball_cn.setIntoCollideMask(BitMask32.bit(8))
        self.trackball_np = self.base.render.attachNewNode(self.trackball_cn)
        # self.trackball_np.show()
        # This creates a collision plane for mouse track
        self.trackplane_cn = CollisionNode("trackplane")
        self.trackplane_cn.addSolid(CollisionPlane(
            Plane(Point3(-self.base.cam.getMat().getRow3(1)),
                  Point3(self.lookatpos_pdv3[0], self.lookatpos_pdv3[1], 0.0))))
        self.trackplane_cn.setFromCollideMask(BitMask32.allOff())
        self.trackplane_cn.setIntoCollideMask(BitMask32.bit(8))
        self.trackplane_np = self.base.render.attachNewNode(self.trackplane_cn)
        # self.trackplane_np.show()
        # creates a traverser to do collision testing
        self.ctrav = CollisionTraverser()
        # creates a queue end_type handler to receive the collision event info
        self.chandler = CollisionHandlerQueue()
        # register the ray as a collider with the traverser,
        # and register the handler queue as the handler to be used for the collisions.
        self.ctrav.addCollider(self.tracker_np, self.chandler)
        # create a pickerray
        self.picker_cn = CollisionNode('picker')
        self.picker_ray = CollisionRay()
        self.picker_cn.addSolid(self.picker_ray)
        self.picker_cn.setFromCollideMask(BitMask32.bit(7))
        self.picker_cn.setIntoCollideMask(BitMask32.allOff())
        self.picker_np = self.base.cam.attachNewNode(self.picker_cn)
        self.ctrav.addCollider(self.picker_np, self.chandler)

    def update_trackballsphere(self, center=np.array([0, 0, 0])):
        self.cam2lookatpos_dist = (self.base.cam.getPos() - self.lookatpos_pdv3).length()
        self.trackball_cn.setSolid(0, CollisionSphere(center[0], center[1], center[2], self.cam2lookatpos_dist))

    def update_trackplane(self):
        self.trackplane_cn.setSolid(0, CollisionPlane(
            Plane(Point3(-self.base.cam.getMat().getRow3(1)),
                  Point3(self.lookatpos_pdv3[0], self.lookatpos_pdv3[1], 0.0))))

    def get_world_mouse1(self):
        """
        get the position of mouse1 (clicked) using collision detection between a sphere and a ray
        :return: Vec3 or None
        author: weiwei
        date: 20161110
        """
        if self.base.mouseWatcherNode.hasMouse():
            if self.keymap['mouse1']:
                # get the mouse position in the window
                mouse_pos = self.base.mouseWatcherNode.getMouse()
                # sets the ray's origin at the camera and directs it to shoot through the mouse cursor
                self.tracker_ray.setFromLens(self.base.cam.node(), mouse_pos.getX(), mouse_pos.getY())
                # performs the collision checking pass
                self.ctrav.traverse(self.trackball_np)
                if (self.chandler.getNumEntries() > 0):
                    # Sort the handler entries from nearest to farthest
                    self.chandler.sortEntries()
                    entry = self.chandler.getEntry(0)
                    colPoint = entry.getSurfacePoint(self.base.render)
                    return colPoint
        return None

    def check_mouse1drag(self):
        """
        this function uses a collision sphere to track the rotational mouse motion
        :return:
        author: weiwei
        date: 20200315
        """
        current_m1_pos = self.get_world_mouse1()
        if current_m1_pos is None:
            if self.last_m1_pos is not None:
                self.last_m1_pos = None
            return
        if self.last_m1_pos is None:
            # first time click
            self.last_m1_pos = current_m1_pos
            return
        cur_m1_vec = Vec3(current_m1_pos - self.lookatpos_pdv3)
        last_m1_vec = Vec3(self.last_m1_pos - self.lookatpos_pdv3)
        cur_m1_vec.normalize()
        last_m1_vec.normalize()
        rotate_ax = cur_m1_vec.cross(last_m1_vec)
        if rotate_ax.length() > 1e-9:  # avoid zero axis_length
            rotate_angle = cur_m1_vec.signedAngleDeg(last_m1_vec, rotate_ax)
            rotate_angle = rotate_angle * self.cam2lookatpos_dist * 5000
            if rotate_angle > .02 or rotate_angle < -.02:
                rotmat = Mat4(self.base.cam.getMat())
                posvec = Vec3(self.base.cam.getPos())
                rotmat.setRow(3, Vec3(0, 0, 0))
                self.base.cam.setMat(rotmat * Mat4.rotateMat(rotate_angle, rotate_ax))
                self.base.cam.setPos(Mat3.rotateMat(rotate_angle, rotate_ax). \
                                     xform(posvec - self.lookatpos_pdv3) + self.lookatpos_pdv3)
                self.last_m1_pos = self.get_world_mouse1()
                self.update_trackplane()

    def get_world_mouse2(self):
        if self.base.mouseWatcherNode.hasMouse():
            if self.keymap['mouse2']:
                mouse_pos = self.base.mouseWatcherNode.getMouse()
                self.tracker_ray.setFromLens(self.base.cam.node(), mouse_pos.getX(), mouse_pos.getY())
                self.ctrav.traverse(self.trackplane_np)
                self.chandler.sortEntries()
                if (self.chandler.getNumEntries() > 0):
                    entry = self.chandler.getEntry(0)
                    colPoint = entry.getSurfacePoint(self.base.render)
                    return colPoint
        return None

    def check_mouse2drag(self):
        """
        :return:
        author: weiwei
        date: 20200313
        """
        current_m2_pos = self.get_world_mouse2()
        if current_m2_pos is None:
            if self.last_m2_pos is not None:
                self.last_m2_pos = None
            return
        if self.last_m2_pos is None:
            # first time click
            self.last_m2_pos = current_m2_pos
            return
        rel_m2_vec = current_m2_pos - self.last_m2_pos
        if rel_m2_vec.length() > 0.001:
            self.base.cam.setPos(self.base.cam.getPos() - rel_m2_vec)
            self.lookatpos_pdv3 = Vec3(self.lookatpos_pdv3 - rel_m2_vec)
            newlookatpos = self.base.p3dh.pdvec3_to_npvec3(self.lookatpos_pdv3)
            if self.togglerotcenter:
                self.rotatecenternp.detachNode()
                self.rotatecenternp = self.base.p3dh.gensphere(pos=newlookatpos, radius=0.005, rgba=np.array([1, 1, 0, 1]))
                self.rotatecenternp.reparentTo(self.base.render)
            self.update_trackballsphere(self.lookatpos_pdv3)
            self.last2mpos = current_m2_pos

    def get_world_mouse3(self):
        """
        picker ray
        :return:
        author: weiwei
        date: 20200316
        """
        if self.base.mouseWatcherNode.hasMouse():
            if self.keymap['mouse3']:
                mouse_pos = self.base.mouseWatcherNode.getMouse()
                self.picker_ray.setFromLens(self.base.cam.node(), mouse_pos.getX(), mouse_pos.getY())
                self.ctrav.traverse(self.base.render)
                if (self.chandler.getNumEntries() > 0):
                    self.chandler.sortEntries()
                    entry = self.chandler.getEntry(0)
                    colPoint = entry.getSurfacePoint(self.base.render)
                    return colPoint
        return None

    def check_mouse3click(self):
        """
        :return:
        author: weiwei
        date: 20200316
        """
        current_m3_pos = self.get_world_mouse3()
        return None if current_m3_pos is None else print(current_m3_pos)

    def check_mousewheel(self):
        """
        zoom up or down the 3d view considering mouse action
        author: weiwei
        date: 2015?, 20200313
        :return:
        """
        self.cam2lookatpos_dist = (self.base.cam.getPos() - self.lookatpos_pdv3).length()
        if self.keymap["wheel_up"] is True:
            self.keymap["wheel_up"] = False
            backward = self.base.cam.getPos() - self.lookatpos_pdv3
            newpos = self.base.cam.getPos() + backward * 0.05
            if newpos.length() < self.initviewdist * 100:
                self.base.cam.setPos(newpos[0], newpos[1], newpos[2])
                self.update_trackballsphere(self.trackball_cn.getSolid(0).getCenter())
        if self.keymap["wheel_down"] is True:
            self.keymap["wheel_down"] = False
            forward = self.lookatpos_pdv3 - self.base.cam.getPos()
            wheelscale_distance = 0.05
            if forward.length() < 0.05:
                wheelscale_distance = 0.0025
            elif forward.length() < 0.0025:
                return
            newpos = self.base.cam.getPos() + forward * wheelscale_distance
            if newpos.length() > self.initviewdist * .01:
                self.base.cam.setPos(newpos[0], newpos[1], newpos[2])
                self.update_trackballsphere(self.trackball_cn.getSolid(0).getCenter())

    def check_resetcamera(self):
        """
        reset the rendering window to its initial viewpoint
        :return:
        author: weiwei
        date: 20200316
        """
        if self.keymap["r"] is True:
            self.keymap["r"] = False
            self.base.cam.setMat(self.original_cam_pdmat4)
            self.lookatpos_pdv3 = self.base.p3dh.npvec3_to_pdvec3(self.originallookatpos)
            self.update_trackplane()
            self.update_trackballsphere(self.lookatpos_pdv3)
            # toggle on the following part to explicitly show the rotation center
            if self.togglerotcenter:
                self.rotatecenternp.detachNode()
                self.rotatecenternp = self.base.p3dh.gensphere(pos=self.originallookatpos, radius=0.005,
                                                               rgba=np.array([1, 1, 0, 1]))
                self.rotatecenternp.reparentTo(self.base.render)