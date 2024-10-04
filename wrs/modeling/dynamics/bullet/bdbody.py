from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletTriangleMesh
from panda3d.bullet import BulletTriangleMeshShape
from panda3d.bullet import BulletConvexHullShape, BulletBoxShape
from panda3d.core import TransformState, Vec3, CollisionBox, Point3
import copy
from wrs import basis as dh, basis as rm, modeling as gm
import numpy as np


class BDBody(BulletRigidBodyNode):

    def __init__(self,
                 initor,
                 cdtype="triangles",
                 mass=.3,
                 restitution=0,
                 allow_deactivation=False,
                 allow_ccd=True,
                 friction=.2,
                 dynamic=True,
                 name="rbd"):
        """
        TODO: triangles do not seem to work (very slow) in the github version (20210418)
        Use convex if possible
        :param initor: could be itself (copy), or an instance of collision model
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
        if isinstance(initor, gm.GeometricModel):
            if initor._trm_mesh is None:
                raise ValueError("Only applicable to models with a trimesh!")
            self.com = initor.trm_mesh.center_mass * base.physics_scale
            self.setMass(mass)
            self.setRestitution(restitution)
            self.setFriction(friction)
            self.setLinearDamping(.3)
            self.setAngularDamping(.3)
            if allow_deactivation:
                self.setDeactivationEnabled(True)
                self.setLinearSleepThreshold(.01*base.physics_scale)
                self.setAngularSleepThreshold(.01*base.physics_scale)
            else:
                self.setDeactivationEnabled(False)
            if allow_ccd:  # continuous collision detection
                self.setCcdMotionThreshold(1e-7)
                self.setCcdSweptSphereRadius(0.0005*base.physics_scale)
            geom_np = initor.pdndp.getChild(0).find("+GeomNode")
            geom = copy.deepcopy(geom_np.node().getGeom(0))
            vdata = geom.modifyVertexData()
            vertices = copy.deepcopy(np.frombuffer(vdata.modifyArrayHandle(0).getData(), dtype=np.float32))
            vertices.shape=(-1,6)
            vertices[:, :3]=vertices[:, :3]*base.physics_scale-self.com
            vdata.modifyArrayHandle(0).setData(vertices.astype(np.float32).tobytes())
            geomtf = geom_np.getTransform()
            geomtf = geomtf.setPos(geomtf.getPos()*base.physics_scale)
            if cdtype == "triangles":
                geombmesh = BulletTriangleMesh()
                geombmesh.addGeom(geom)
                bulletshape = BulletTriangleMeshShape(geombmesh, dynamic=dynamic)
                bulletshape.setMargin(1e-6)
                self.addShape(bulletshape, geomtf)
            elif cdtype == "convex":
                bulletshape = BulletConvexHullShape()  # TODO: compute a convex hull?
                bulletshape.addGeom(geom, geomtf)
                bulletshape.setMargin(1e-6)
                self.addShape(bulletshape, geomtf)
            elif cdtype == 'box':
                minx = min(vertices[:,0])
                miny = min(vertices[:,1])
                minz = min(vertices[:,2])
                maxx = max(vertices[:,0])
                maxy = max(vertices[:,1])
                maxz = max(vertices[:,2])
                pcd_box = CollisionBox(Point3(minx, miny, minz),Point3(maxx, maxy, maxz))
                bulletshape = BulletBoxShape.makeFromSolid(pcd_box)
                bulletshape.setMargin(1e-6)
                self.addShape(bulletshape, geomtf)
            else:
                raise NotImplementedError
            pd_homomat = geomtf.getMat()
            pd_com_pos = pd_homomat.xformPoint(Vec3(self.com[0], self.com[1], self.com[2]))
            np_homomat = dh.pdmat4_to_npmat4(pd_homomat)
            np_com_pos = dh.pdvec3_to_npvec3(pd_com_pos)
            np_homomat[:3, 3] = np_com_pos  # update center to com
            self.setTransform(TransformState.makeMat(dh.npmat4_to_pdmat4(np_homomat)))
        elif isinstance(initor, BDBody):
            self.com = initor.com.copy()
            self.setMass(initor.getMass())
            self.setRestitution(initor.restitution)
            self.setFriction(initor.friction)
            self.setLinearDamping(.3)
            self.setAngularDamping(.3)
            if allow_deactivation:
                self.setDeactivationEnabled(True)
                self.setLinearSleepThreshold(.01*base.physics_scale)
                self.setAngularSleepThreshold(.01*base.physics_scale)
            else:
                self.setDeactivationEnabled(False)
            if allow_ccd:
                self.setCcdMotionThreshold(1e-7)
                self.setCcdSweptSphereRadius(0.0005*base.physics_scale)
            np_homomat = copy.deepcopy(initor.get_homomat())
            np_homomat[:3,3] = np_homomat[:3,3]*base.physics_scale
            self.setTransform(TransformState.makeMat(dh.npmat4_to_pdmat4(np_homomat)))
            self.addShape(initor.getShape(0), initor.getShapeTransform(0))

    def get_pos(self):
        pdmat4 = self.getTransform().getMat()
        pdv3 = pdmat4.xformPoint(Vec3(-self.com[0], -self.com[1], -self.com[2]))
        pos = dh.pdvec3_to_npvec3(pdv3) / base.physics_scale
        return pos

    def set_pos(self, npvec3):
        self.setPos(dh.pdvec3_to_npvec3(npvec3) * base.physics_scale)

    def get_homomat(self):
        """
        get the pos considering the original local frame
        the dynamic body moves in a local frame defined at com (line 46 of this file), instead of returning the
        pos of the dynamic body, this file returns the pose of original local frame
        the returned pos can be used by collision bodies for rendering.
        :return:
        author: weiwei
        date: 2019?, 20201119
        """
        pd_homomat = self.getTransform().getMat()
        pd_com_pos = pd_homomat.xformPoint(Vec3(-self.com[0], -self.com[1], -self.com[2]))
        np_homomat = dh.pdmat4_to_npmat4(pd_homomat)
        np_com_pos = dh.pdvec3_to_npvec3(pd_com_pos)
        np_homomat[:3, 3] = np_com_pos/base.physics_scale
        return np_homomat

    def set_homomat(self, homomat):
        """
        set the pose of the dynamic body
        :param homomat: the pos of the original frame (the collision model)
        :return:
        author: weiwei
        date: 2019?, 20201119
        """
        tmp_homomat = copy.deepcopy(homomat)
        tmp_homomat[:3, 3] = tmp_homomat[:3,3]*base.physics_scale
        pos = rm.transform_points_by_homomat(tmp_homomat, self.com)
        rotmat = tmp_homomat[:3, :3]
        self.setTransform(TransformState.makeMat(dh.npv3mat3_to_pdmat4(pos, rotmat)))

    def copy(self):
        return BDBody(self)
