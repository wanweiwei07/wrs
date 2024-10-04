from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletTriangleMesh
from panda3d.bullet import BulletTriangleMeshShape
from panda3d.bullet import BulletConvexHullShape
from panda3d.core import TransformState, Vec3, GeomVertexRewriter
import copy
from wrs import basis as dh, basis as rm, modeling as gm


class BDBody(BulletRigidBodyNode):
    def __init__(self, objinit, cdtype="triangle", mass=.3, restitution=0, allowdeactivation=False, allowccd=True,
                 friction=.2, dynamic=True, name="rbd"):
        """
        :param objinit: could be itself (copy), or an instance of collision model
        :param end_type: triangle or convex
        :param mass:
        :param restitution: bounce parameter
        :param friction:
        :param dynamic: only applicable to triangle end_type, if an object does not move with force, it is not dynamic
        :param name:
        author: weiwei
        date: 20190626, 20201119
        """
        super().__init__(name)
        if isinstance(objinit, gm.GeometricModel):
            if objinit._trm_mesh is None:
                raise ValueError("Only applicable to models with a trimesh!")
            self.com = objinit.trm_mesh.center_mass
            self.setMass(mass)
            self.setRestitution(restitution)
            self.setFriction(friction)
            self.setLinearDamping(.3)
            self.setAngularDamping(.3)
            if allowdeactivation:
                self.setDeactivationEnabled(True)
                self.setLinearSleepThreshold(0.001)
                self.setAngularSleepThreshold(0.001)
            else:
                self.setDeactivationEnabled(False)
            if allowccd:  # continuous collision detection
                self.setCcdMotionThreshold(1e-6)
                self.setCcdSweptSphereRadius(0.0005)
            gnd = objinit.pdndp.getChild(0).find("+GeomNode")
            geom = copy.deepcopy(gnd.node().getGeom(0))
            vdata = geom.modifyVertexData()
            vertrewritter = GeomVertexRewriter(vdata, 'vertex')
            while not vertrewritter.isAtEnd():  # shift local coordinate to geom to correctly update dynamic changes
                v = vertrewritter.getData3f()
                vertrewritter.setData3f(v[0] - self.com[0], v[1] - self.com[1], v[2] - self.com[2])
            geomtf = gnd.getTransform()
            if cdtype is "triangle":
                geombmesh = BulletTriangleMesh()
                geombmesh.addGeom(geom)
                bulletshape = BulletTriangleMeshShape(geombmesh, dynamic=dynamic)
                bulletshape.setMargin(1e-6)
                self.addShape(bulletshape, geomtf)
            elif cdtype is "convex":
                bulletshape = BulletConvexHullShape()  # TODO: compute a convex hull?
                bulletshape.addGeom(geom, geomtf)
                bulletshape.setMargin(1e-6)
                self.addShape(bulletshape, geomtf)
            else:
                raise NotImplementedError
            pdmat4 = geomtf.getMat()
            pdv3 = pdmat4.xformPoint(Vec3(self.com[0], self.com[1], self.com[2]))
            homomat = dh.pdmat4_to_npmat4(pdmat4)
            pos = dh.pdvec3_to_npvec3(pdv3)
            homomat[:3, 3] = pos  # update center to com
            self.setTransform(TransformState.makeMat(dh.npmat4_to_pdmat4(homomat)))
        elif isinstance(objinit, BDBody):
            self.com = objinit.com.copy()
            self.setMass(objinit.getMass())
            self.setRestitution(objinit.restitution)
            self.setFriction(objinit.friction)
            self.setLinearDamping(.3)
            self.setAngularDamping(.3)
            if allowdeactivation:
                self.setDeactivationEnabled(True)
                self.setLinearSleepThreshold(0.001)
                self.setAngularSleepThreshold(0.001)
            else:
                self.setDeactivationEnabled(False)
            if allowccd:
                self.setCcdMotionThreshold(1e-6)
                self.setCcdSweptSphereRadius(0.0005)
            self.setTransform(TransformState.makeMat(dh.npmat4_to_pdmat4(objinit.gethomomat())))
            self.addShape(objinit.getShape(0), objinit.getShapeTransform(0))

    def getpos(self):
        """
        :return: 1x3 nparray
        """
        pdmat4 = self.getTransform().getMat()
        pdv3 = pdmat4.xformPoint(Vec3(-self.com[0], -self.com[1], -self.com[2]))
        # pos = dh.pdmat4_to_npmat4(pdmat4)
        pos = dh.pdvec3_to_npvec3(pdv3)
        return pos

    def setpos(self, npvec3):
        """
        :param npvec3: 1x3 nparray
        :return:
        """
        self.setPos(dh.pdvec3_to_npvec3(npvec3))

    def gethomomat(self):
        """
        get the pos considering the original local frame
        the dynamic body moves in a local frame defined at com (line 46 of this file), instead of returning the
        pos of the dynamic body, this file returns the pose of original local frame
        the returned pos can be used by collision bodies for rendering.
        :return:
        author: weiwei
        date: 2019?, 20201119
        """
        pdmat4 = self.getTransform().getMat()
        pdv3 = pdmat4.xformPoint(Vec3(-self.com[0], -self.com[1], -self.com[2]))
        homomat = dh.pdmat4_to_npmat4(pdmat4)
        pos = dh.pdvec3_to_npvec3(pdv3)
        homomat[:3, 3] = pos
        return homomat

    def sethomomat(self, homomat):
        """
        set the pose of the dynamic body
        :param homomat: the pos of the original frame (the collision model)
        :return:
        author: weiwei
        date: 2019?, 20201119
        """
        pos = rm.transform_points_by_homomat(homomat, self.com)
        rotmat = homomat[:3, :3]
        self.setTransform(TransformState.makeMat(dh.npv3mat3_to_pdmat4(pos, rotmat)))

    def setmass(self, mass):
        self.mass=mass
        self.setMass(mass)

    def copy(self):
        return BDBody(self)
