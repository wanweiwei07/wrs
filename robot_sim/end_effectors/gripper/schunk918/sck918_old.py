# TODO: reduce the dependency on panda3d

import os

import utiltools.robotmath as rm
import pandaplotutils.pandactrl as pandactrl
import pandaplotutils.pandageom as pandageom
from panda3d.bullet import BulletDebugNode
from panda3d.bullet import BulletWorld
from panda3d.core import *

class Sck918(object):
    '''
    use utiltools.designpattern.singleton() to get a single instance of this class
    '''

    def __init__(self, jawwidth=50, ftsensoroffset = 52):
        """
        load the robotiq85 model, set jaw_width and return a nodepath

        ## input
        pandabase:
            the showbase() object
        jaw_width:
            the distance between fingertips
        ftsensoroffset:
            the offset for ftsensor

        ## output
        sck918np:
            the nodepath of this rtq85 hand

        author: wangyan
        date: 20190413
        """

        self.sck918np = NodePath("sck918hnd")
        self.handnp = self.sck918np
        self.__ftsensoroffset = ftsensoroffset
        self.jawwidth = jawwidth

        this_dir, this_filename = os.path.split(__file__)
        sck918basepath = Filename.fromOsSpecific(os.path.join(this_dir, "schunk918egg", "sck918_base.egg"))
        sck918sliderpath = Filename.fromOsSpecific(os.path.join(this_dir, "schunk918egg", "sck918_slider.egg"))
        sck918gripperpath = Filename.fromOsSpecific(os.path.join(this_dir, "schunk918egg", "sck918_gripper.egg"))

        sck918base = NodePath("sck918base")
        sck918lslider = NodePath("sck918lslider")
        sck918rslider = NodePath("sck918rslider")
        sck918lgripper = NodePath("sck918lgripper")
        sck918rgripper = NodePath("sck918rgripper")

        # loader is a global variable defined by panda3d
        sck918_basel = loader.loadModel(sck918basepath)
        sck918_sliderl = loader.loadModel(sck918sliderpath)
        sck918_gripperl = loader.loadModel(sck918gripperpath)

        # base
        sck918_basel.instanceTo(sck918base)
        sck918base.setColor(.2, .2, .2, 1.0)
        sck918base.setHpr(0, -90, 0)
        # left and right outer knuckle
        sck918_sliderl.instanceTo(sck918lslider)
        sck918lslider.setColor(.1, .1, .1, 1.0)
        sck918lslider.setPos(-10, 40, 73)
        sck918lslider.setHpr(-90, 0, 0)
        sck918lslider.reparentTo(sck918base)
        sck918_sliderl.instanceTo(sck918rslider)
        sck918rslider.setColor(.1, .1, .1, 1.0)
        sck918rslider.setPos(10, -40, 73)
        sck918rslider.setHpr(90, 0, 0)
        sck918rslider.reparentTo(sck918base)

        # left and right finger
        sck918_gripperl.instanceTo(sck918lgripper)
        sck918lgripper.setColor(.7, .7, .7, 1.0)
        sck918lgripper.setPos(-8, 20, 0)
        sck918lgripper.setHpr(0, 180, 0)
        sck918lgripper.reparentTo(sck918lslider)
        sck918_gripperl.instanceTo(sck918rgripper)
        sck918rgripper.setColor(.7, .7, .7, 1.0)
        sck918rgripper.setPos(-8, 20, 0)
        sck918rgripper.setHpr(0, 180, 0)
        sck918rgripper.reparentTo(sck918rslider)

        # rotate to x, y, z coordinates (this one rotates the base, not the self.rtq85np)
        # the default x direction is facing the ee, the default z direction is facing downward
        # execute this file to see the default pose
        sck918base.setTransparency(TransparencyAttrib.MAlpha)
        sck918base.setMat(pandageom.npToMat4(rm.rodrigues([0,0,1], 90))*
                          pandageom.npToMat4(rm.rodrigues([1,0,0], 90))*sck918base.getMat())
        sck918base.setPos(0,0,self.__ftsensoroffset)
        sck918base.reparentTo(self.sck918np)
        self.setJawwidth(jawwidth)

        self.__jawwidthopen = 50.0
        self.__jawwidthclosed = 0.0

    @property
    def jawwidthopen(self):
        # read-only property
        return self.__jawwidthopen

    @property
    def jawwidthclosed(self):
        # read-only property
        return self.__jawwidthclosed

    def setJawwidth(self, jawwidth):
        """
        set the jaw_width of sck918hnd

        ## input
        sck918hnd:
            nodepath of a schunk918hand
        jaw_width:
            the width of the jaw

        author: wangyan
        date: 20190413
        """

        if jawwidth > 50.0 or jawwidth < 0.0:
            print ("Wrong value! Jawwidth must be in (0.0,50.0). The input is "+str(jawwidth)+".")
            raise Exception("Jawwidth out of range!")

        self.jawwidth = jawwidth

        translider = 25 - jawwidth/2
        print("translider is %s" % translider)

        # right gripper
        sck918rslider = self.sck918np.find("**/sck918rslider")
        sck918rsliderpos = sck918rslider.getPos()
        sck918rslider.setPos(sck918rsliderpos[0], translider-40, sck918rsliderpos[2])

        # left gripper
        sck918lslider = self.sck918np.find("**/sck918lslider")
        sck918lsliderpos = sck918lslider.getPos()
        sck918lslider.setPos(sck918lsliderpos[0], -translider+40, sck918lsliderpos[2])

    def setPos(self, npvec3):
        """
        set the pose of the hand
        changes self.sck918np

        :param npvec3
        :return:
        """

        self.sck918np.setPos(npvec3)

    def getPos(self):
        """
        set the pose of the hand
        changes self.sck918np

        :param npvec3
        :return:
        """

        return self.sck918np.getPos()

    def setMat(self, npmat4):
        """
        set the translation and rotation of a schunk hand
        changes self.sck918np

        :param npmat4: follows panda3d, a LMatrix4f matrix
        :return: null

        date: 20161109
        author: weiwei
        """

        self.sck918np.setMat(npmat4)

    def getMat(self):
        """
        get the rotation matrix of the hand

        :return: npmat4: follows panda3d, a LMatrix4f matrix

        date: 20161109
        author: weiwei
        """

        return self.sck918np.getMat()

    def reparentTo(self, nodepath):
        """
        add to scene, follows panda3d

        :param nodepath: a panda3d nodepath
        :return: null

        date: 20161109
        author: weiwei
        """
        self.sck918np.reparentTo(nodepath)

    def removeNode(self):
        """

        :return:
        """

        self.sck918np.removeNode()

    def lookAt(self, direct0, direct1, direct2):
        """
        set the Y axis of the hnd

        author: weiwei
        date: 20161212
        """

        self.sck918np.lookAt(direct0, direct1, direct2)

    def gripAt(self, fcx, fcy, fcz, c0nx, c0ny, c0nz, rotangle = 0, jawwidth = 50):
        '''
        set the hand to grip at fcx, fcy, fcz, fc = finger center
        the normal of the sglfgr contact is set to be c0nx, c0ny, c0nz
        the rotation around the normal is set to rotangle
        the jaw_width is set to jaw_width

        date: 20170322
        author: weiwei
        '''

        self.sck918np.setMat(Mat4.identMat())
        self.setJawwidth(jawwidth)
        self.sck918np.lookAt(c0nx, c0ny, c0nz)
        handmat4 = self.sck918np.getMat()
        self.sck918np.setMat(handmat4*Mat4.rotateMat(90.0, handmat4.getRow3(2)))
        rotmat4x = Mat4.rotateMat(rotangle, Vec3(c0nx, c0ny, c0nz))
        self.sck918np.setMat(self.sck918np.getMat()*rotmat4x)
        rotmat4 = Mat4(self.sck918np.getMat())
        handtipvec3 = -rotmat4.getRow3(2)*(145.0+self.__ftsensoroffset)
        rotmat4.setRow(3, Vec3(fcx, fcy, fcz)+handtipvec3)
        self.sck918np.setMat(rotmat4)

    def plot(self, nodepath, pos=None, ydirect=None, zdirect=None, rgba=None):
        '''
        plot the hand under the given nodepath

        ## input
        nodepath:
            the parent node this hand is going to be attached to
        pos:
            the position of the hand
        ydirect:
            the y direction of the hand
        zdirect:
            the z direction of the hand
        rgba:
            the rgba color

        ## note:
            dot(ydirect, zdirect) must be 0

        date: 20160628
        author: weiwei
        '''

        if pos is None:
            pos = Vec3(0,0,0)
        if ydirect is None:
            ydirect = Vec3(0,1,0)
        if zdirect is None:
            zdirect = Vec3(0,0,1)
        if rgba is None:
            rgba = Vec4(1,1,1,0.5)

        # assert(ydirect.dot(zdirect)==0)

        placeholder = nodepath.attachNewNode("sck918holder")
        self.sck918np.instanceTo(placeholder)
        xdirect = ydirect.cross(zdirect)
        transmat4 = Mat4()
        transmat4.setCol(0, xdirect)
        transmat4.setCol(1, ydirect)
        transmat4.setCol(2, zdirect)
        transmat4.setCol(3, pos)
        self.sck918np.setMat(transmat4)
        placeholder.setColor(rgba)

def getHandName():
    return "sck918"

def newHand(jawwidth=50, ftsensoroffset = 52):
    return Sck918(jawwidth, ftsensoroffset)


if __name__=='__main__':
    def updateworld(world, task):
        world.doPhysics(globalClock.getDt())
        # result = base.world.contactTestPair(bcollidernp.node(), lftcollidernp.node())
        # result1 = base.world.contactTestPair(bcollidernp.node(), ilkcollidernp.node())
        # result2 = base.world.contactTestPair(lftcollidernp.node(), ilkcollidernp.node())
        # print result
        # print result.getContacts()
        # print result1
        # print result1.getContacts()
        # print result2
        # print result2.getContacts()
        # for contact in result.getContacts():
        #     cp = contact.getManifoldPoint()
        #     print cp.getLocalPointA()
        return task.cont

    base = pandactrl.World(lookatpos=[0, 0, 0])
    base.world = BulletWorld()
    base.taskMgr.add(updateworld, "updateworld", extraArgs=[base.world], appendTask=True)
    sck918hnd = Sck918(jawwidth=10, ftsensoroffset=0)
    # sck918hnd.setJawwidth(25)
    sck918hnd.gripAt(0,0,0,0,0,1,30)
    sck918hnd.reparentTo(base.render)
    base.pggen.plotAxis(base.render)
    rmathnd = sck918hnd.getMat()
    # base.pggen.plotAxis(base.render, spos = rmathnd.getRow3(3), pandamat3=rmathnd.getUpper3())
    # base.run()

    # base = pandactrl.World()
    # sck918hnd = designpattern.singleton(Sck918)
    # sck918hnd.setJawwidth(50)
    # hndpos = Vec3(0,0,0)
    # ydirect = Vec3(0,1,0)
    # zdirect = Vec3(0,0,1)
    # sck918hnd.gripAt(0,0,0, 1,0,0, 0)
    # sck918hnd.reparentTo(base.render)
    #
    # # axis = loader.loadModel('zup-axis.egg')
    # # axis.reparentTo(base.render)
    # # axis.setPos(hndpos)
    # # axis.setScale(50)
    # # axis.lookAt(hndpos+ydirect)
    # import pandaplotutils.pandageom as pandageom
    # pgg = pandageom.PandaGeomGen()
    # pgg.plotAxis(base.render)
    #
    bullcldrnp = base.render.attachNewNode("bulletcollider")
    base.world = BulletWorld()
    import pandaplotutils.collisiondetection as cd
    obj1bullnode = cd.genCollisionMeshMultiNp(sck918hnd.handnp)
    base.world.attachRigidBody(obj1bullnode)
    #
    # # hand base
    # # rtq85hnd.rtq85np.find("**/rtq85base").showTightBounds()
    # gbnp = sck918hnd.rtq85np.find("**/sck918base").find("**/+GeomNode")
    # gb = gbnp.node().getGeom(0)
    # gbts = gbnp.getTransform(base.render)
    # gbmesh = BulletTriangleMesh()
    # gbmesh.addGeom(gb)
    # bbullnode = BulletRigidBodyNode('gb')
    # bbullnode.addShape(BulletTriangleMeshShape(gbmesh, dynamic=True), gbts)
    # bcollidernp=bullcldrnp.attachNewNode(bbullnode)
    # base.world.attachRigidBody(bbullnode)
    # bcollidernp.setCollideMask(BitMask32.allOn())
    #
    # # rtq85hnd.rtq85np.find("**/rtq85lfgrtip").showTightBounds()
    # glftnp = sck918hnd.rtq85np.find("**/sck918lgipper").find("**/+GeomNode")
    # glft = glftnp.node().getGeom(0)
    # glftts = glftnp.getTransform(base.render)
    # glftmesh = BulletTriangleMesh()
    # glftmesh.addGeom(glft)
    # # lftbullnode = BulletRigidBodyNode('glft')
    # # lftbullnode.addShape(BulletTriangleMeshShape(glftmesh, dynamic=True), glftts)
    # # lftcollidernp=bullcldrnp.attachNewNode(lftbullnode)
    # # base.world.attachRigidBody(lftbullnode)
    # # lftcollidernp.setCollideMask(BitMask32.allOn())
    # # base.world.attachRigidBody(glftbullnode)
    #
    # # rtq85hnd.rtq85np.find("**/rtq85ilknuckle").showTightBounds()
    # gilknp = sck918hnd.sck918np.find("**/sck918lslider").find("**/+GeomNode")
    # gilk = gilknp.node().getGeom(0)
    # gilkts = gilknp.getTransform(base.render)
    # gilkmesh = BulletTriangleMesh()
    # gilkmesh.addGeom(gilk)
    # ilkbullnode = BulletRigidBodyNode('gilk')
    # ilkbullnode.addShape(BulletTriangleMeshShape(gilkmesh, dynamic=True), gilkts)
    # ilkbullnode.addShape(BulletTriangleMeshShape(glftmesh, dynamic=True), glftts)
    # ilkcollidernp=bullcldrnp.attachNewNode(ilkbullnode)
    # base.world.attachRigidBody(ilkbullnode)
    # ilkcollidernp.setCollideMask(BitMask32.allOn())
    # # rtq85hnd.rtq85np.find("**/rtq85ilknuckle").showTightBounds()
    # # rtq85hnd.rtq85np.showTightBounds()
    #
    # base.taskMgr.add(updateworld, "updateworld", extraArgs=[base.world], appendTask=True)
    # result = base.world.contactTestPair(bbullnode, ilkbullnode)
    # print(result)
    # print(result.getContacts())
    # for contact in result.getContacts():
    #     cp = contact.getManifoldPoint()
    #     print(cp.getLocalPointA())
    #     pgg.plotSphere(base.render, pos=cp.getLocalPointA(), radius=1, rgba=Vec4(1,0,0,1))
    #
    debugNode = BulletDebugNode('Debug')
    debugNode.showWireframe(True)
    debugNode.showConstraints(True)
    debugNode.showBoundingBoxes(False)
    debugNode.showNormals(False)
    debugNP = bullcldrnp.attachNewNode(debugNode)
    # debugNP.show()
    #
    base.world.setDebugNode(debugNP.node())
    base.taskMgr.add(updateworld, "updateworld", extraArgs=[base.world], appendTask=True)
    #
    base.run()