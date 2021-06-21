from direct.showbase.DirectObject import DirectObject
from panda3d.core import Vec3, Mat3, Mat4, CollisionNode, CollisionRay, BitMask32, CollisionSphere, Plane, \
    CollisionPlane, CollisionBox, Point3, CollisionTraverser, CollisionHandlerQueue, GeomNode
import numpy as np
import math


class InputManager(DirectObject):

    def __init__(self, base, lookatpos, togglerotcenter=False):
        self.base = base
        self.originallookatpos = lookatpos  # for backup
        self.lookatpos_pdv3 = Vec3(lookatpos[0], lookatpos[1], lookatpos[2])
        self.camdist = (self.base.cam.getPos() - self.lookatpos_pdv3).length()
        self.initviewdist = (self.base.cam.getPos() - self.lookatpos_pdv3).length()
        self.lastm1pos = None
        self.lastm2pos = None
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
            CollisionSphere(self.lookatpos_pdv3[0], self.lookatpos_pdv3[1], self.lookatpos_pdv3[2], self.camdist))
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
        # creates a queue type handler to receive the collision event info
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
        self.camdist = (self.base.cam.getPos() - self.lookatpos_pdv3).length()
        self.trackball_cn.setSolid(0, CollisionSphere(center[0], center[1], center[2], self.camdist))

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
                mpos = self.base.mouseWatcherNode.getMouse()
                # sets the ray's origin at the camera and directs it to shoot through the mouse cursor
                self.tracker_ray.setFromLens(self.base.cam.node(), mpos.getX(), mpos.getY())
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
        curm1pos = self.get_world_mouse1()
        if curm1pos is None:
            if self.lastm1pos is not None:
                self.lastm1pos = None
            return
        if self.lastm1pos is None:
            # first time click
            self.lastm1pos = curm1pos
            return
        curm1vec = Vec3(curm1pos - self.lookatpos_pdv3)
        lastm1vec = Vec3(self.lastm1pos - self.lookatpos_pdv3)
        curm1vec.normalize()
        lastm1vec.normalize()
        rotatevec = curm1vec.cross(lastm1vec)
        if rotatevec.length() > 1e-9:  # avoid zero length
            rotateangle = curm1vec.signedAngleDeg(lastm1vec, rotatevec)
            rotateangle = rotateangle * self.camdist * 5000
            if rotateangle > .02 or rotateangle < -.02:
                rotmat = Mat4(self.base.cam.getMat())
                posvec = Vec3(self.base.cam.getPos())
                rotmat.setRow(3, Vec3(0, 0, 0))
                self.base.cam.setMat(rotmat * Mat4.rotateMat(rotateangle, rotatevec))
                self.base.cam.setPos(Mat3.rotateMat(rotateangle, rotatevec). \
                                     xform(posvec - self.lookatpos_pdv3) + self.lookatpos_pdv3)
                self.lastm1pos = self.get_world_mouse1()
                self.update_trackplane()

    def get_world_mouse2(self):
        if self.base.mouseWatcherNode.hasMouse():
            if self.keymap['mouse2']:
                mpos = self.base.mouseWatcherNode.getMouse()
                self.tracker_ray.setFromLens(self.base.cam.node(), mpos.getX(), mpos.getY())
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
        curm2pos = self.get_world_mouse2()
        if curm2pos is None:
            if self.lastm2pos is not None:
                self.lastm2pos = None
            return
        if self.lastm2pos is None:
            # first time click
            self.lastm2pos = curm2pos
            return
        relm2vec = curm2pos - self.lastm2pos
        if relm2vec.length() > 0.001:
            self.base.cam.setPos(self.base.cam.getPos() - relm2vec)
            self.lookatpos_pdv3 = Vec3(self.lookatpos_pdv3 - relm2vec)
            newlookatpos = self.base.p3dh.pdv3_to_npv3(self.lookatpos_pdv3)
            if self.togglerotcenter:
                self.rotatecenternp.detachNode()
                self.rotatecenternp = self.base.p3dh.gensphere(pos=newlookatpos, radius=0.005, rgba=np.array([1, 1, 0, 1]))
                self.rotatecenternp.reparentTo(self.base.render)
            self.update_trackballsphere(self.lookatpos_pdv3)
            self.last2mpos = curm2pos

    def get_world_mouse3(self):
        """
        picker ray
        :return:
        author: weiwei
        date: 20200316
        """
        if self.base.mouseWatcherNode.hasMouse():
            if self.keymap['mouse3']:
                mpos = self.base.mouseWatcherNode.getMouse()
                self.picker_ray.setFromLens(self.base.cam.node(), mpos.getX(), mpos.getY())
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
        curm3pos = self.get_world_mouse3()
        return None if curm3pos is None else print(curm3pos)

    def check_mousewheel(self):
        """
        zoom up or down the 3d view considering mouse action
        author: weiwei
        date: 2015?, 20200313
        :return:
        """
        self.camdist = (self.base.cam.getPos() - self.lookatpos_pdv3).length()
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
            self.lookatpos_pdv3 = self.base.p3dh.npv3_to_pdv3(self.originallookatpos)
            self.update_trackplane()
            self.update_trackballsphere(self.lookatpos_pdv3)
            # toggle on the following part to explicitly show the rotation center
            if self.togglerotcenter:
                self.rotatecenternp.detachNode()
                self.rotatecenternp = self.base.p3dh.gensphere(pos=self.originallookatpos, radius=0.005,
                                                               rgba=np.array([1, 1, 0, 1]))
                self.rotatecenternp.reparentTo(self.base.render)