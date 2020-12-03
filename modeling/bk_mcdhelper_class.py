# mesh collision detection
# this file has two classes: BMChecker, MChecker; MChecker inherits BMChecker

from panda3d.bullet import BulletDebugNode, BulletWorld, BulletRigidBodyNode, BulletPlaneShape, BulletBoxShape
from panda3d.bullet import BulletTriangleMeshShape, BulletTriangleMesh
from panda3d.core import TransformState, Vec3
import copy
import numpy as np
import basis.robotmath as rm
import basis.dataadapter as dh


class BMChecker(object):
    """
    BMChecker, box mesh collision checker
    """

    def __init__(self, toggledebug=False):
        self._bltworld = BulletWorld()
        self._toggledebug = toggledebug
        if self._toggledebug:
            bulletcolliderrender = base.render.attachNewNode("bulletboxcollider")
            debugNode = BulletDebugNode('Debug')
            debugNode.showWireframe(True)
            debugNode.showConstraints(True)
            debugNode.showBoundingBoxes(False)
            debugNode.showNormals(False)
            self._debugNP = bulletcolliderrender.attachNewNode(debugNode)
            self._bltworld.setDebugNode(self._debugNP.node())
        self._bltworldrigidbodylist = []
        self._is_updatebltworld_added = False

    def _updateblt(self, bltworld, task):
        bltworld.doPhysics(globalClock.getDt())
        return task.cont

    def is_cmcm_collided(self, objcm0, objcm1):
        """
        check if two objects objcm0 as objcm1 are in collision with each other
        the two objects are in the form of collision model
        the AABB boxlist will be used
        type "box" is required
        :param objcm0: the first object
        :param objcm1: the second object
        :return: boolean value showing if the two objects are in collision
        author: weiwei
        date: 20190313
        """
        objcm0boxbullnode = _gen_boxcdmesh(objcm0)
        objcm1boxbullnode = _gen_boxcdmesh(objcm1)
        result = self._bltworld.contactTestPair(objcm0boxbullnode, objcm1boxbullnode)
        if not result.getNumContacts():
            return False
        else:
            return True

    def is_cmcmlist_collided(self, objcm0, objcmlist=[]):
        """
        check if objcm0 is in collision with a list of collisionmodels in objcmlist
        each obj is in the form of a collision model
        :param objcm0:
        :param obcmjlist: a list of collision models
        :return: boolean value showing if the object and the list are in collision
        author: weiwei
        date: 20190313
        """
        objcm0boxbullnode = _gen_boxcdmesh(objcm0)
        objcm1boxbullnode = _gen_boxcdmesh_list(objcmlist)
        result = self._bltworld.contactTestPair(objcm0boxbullnode, objcm1boxbullnode)
        if not result.getNumContacts():
            return False
        else:
            return True

    def is_cmlistcmlist_collided(self, objcm0list=[], objcm1list=[]):
        """
        check if a list of collisionmodels in objcm0list is in collision with a list of collisionmodels in objcm1list
        each obj is in the form of a collision model
        :param objcm0list: a list of collision models
        :param obcmj1list: a list of collision models
        :return: boolean value showing if the object and the list are in collision
        author: weiwei
        date: 20190313
        """
        objcm0boxbullnode = _gen_boxcdmesh_list(objcm0list)
        objcm1boxbullnode = _gen_boxcdmesh_list(objcm1list)
        result = self._bltworld.contactTestPair(objcm0boxbullnode, objcm1boxbullnode)
        if not result.getNumContacts():
            return False
        else:
            return True

    def show(self, objcm):
        """
        show the AABB collision meshes of the given objects
        :param objcm
        author: weiwei
        date: 20190313
        :return:
        """
        if not self._toggledebug:
            print("Toggle debug on during defining the XCMchecker object to use showfunctions!")
            return
        if not self._is_updatebltworld_added:
            base.taskMgr.add(self._updateblt, "updateblt", extraArgs=[self._bltworld], appendTask=True)
        objcmboxbullnode = _gen_boxcdmesh(objcm)
        self._bltworld.attach(objcmboxbullnode)
        self._bltworldrigidbodylist.append(objcmboxbullnode)
        self._debugNP.show()

    def showlist(self, objcmlist):
        """
        show the AABB collision meshes of the given objects
        :param objcm0, objcm1
        author: weiwei
        date: 20190313
        :return:
        """
        if not self._toggledebug:
            print("Toggle debug on during defining the XCMchecker object to use showfunctions!")
            return
        if not self._is_updatebltworld_added:
            base.taskMgr.add(self._updateworld, "updateworld", extraArgs=[self._bltworld], appendTask=True)
        objcmboxbullnode = _gen_boxcdmesh_list(objcmlist)
        self._bltworld.attach(objcmboxbullnode)
        self._bltworldrigidbodylist.append(objcmboxbullnode)
        self._debugNP.show()

    def unshow(self):
        """
        unshow everything
        author: weiwei
        date: 20180621
        :return:
        """
        base.taskMgr.remove("updateworld")
        print(self._bltworldrigidbodylist)
        for cdnode in self._bltworldrigidbodylist:
            self._bltworld.remove(cdnode)
        self._bltworldrigidbodylist = []
        self._debugNP.hide()


class MChecker(BMChecker):
    """
    MChecker, mesh collision checker
    """

    def __init__(self, toggledebug=False):
        super().__init__(toggledebug)

    def is_cmcm_collided(self, objcm0, objcm1):
        """
        check if two objects objcm0 and objcm1 are in collision with each other
        the two objects are in the form of collision model
        the bulletmeshes will be used
        :param objcm0: the first object
        :param objcm1: the second object
        :return: boolean value showing if the two objects are in collision
        author: weiwei
        date: 20190313
        """
        objcm0bullnode = _gen_cdmesh(objcm0)
        objcm1bullnode = _gen_cdmesh(objcm1)
        result = self._bltworld.contactTestPair(objcm0bullnode, objcm1bullnode)
        return True if result.getNumContacts() else False

    def is_cmcmlist_collided(self, objcm0, objcm1list):
        """
        check if object objcm0 and objectlist objcm1list are in collision with each other
        the two objects are in the form of collision model
        the bulletmeshes will be used
        :param objcm0: the first collision model
        :param objcm1list: the second collision model list
        :return: boolean value showing if the object and object list are in collision
        author: weiwei
        date: 20190514
        """
        objcm0bullnode = _gen_cdmesh(objcm0)
        objcm1bullnode = _gen_cdmesh_list(objcm1list)
        result = self._bltworld.contactTestPair(objcm0bullnode, objcm1bullnode)
        return True if result.getNumContacts() else False

    def is_cmlistcmlist_collided(self, objcm0list, objcm1list):
        """
        check if two object lists objcm0list and objcm1list are in collision with each other
        the two objects are in the form of collision model
        the bulletmeshes will be used
        :param objcm0list: the first collision model list
        :param objcm1list: the second collision model list
        :return: boolean value showing if the two objects are in collision
        author: weiwei
        date: 20190514
        """
        objcm0bullnode = _gen_cdmesh_list(objcm0list)
        objcm1bullnode = _gen_cdmesh_list(objcm1list)
        result = self._bltworld.contactTestPair(objcm0bullnode, objcm1bullnode)
        return True if result.getNumContacts() else False

    def rayhit_closet(self, pfrom, pto, objcm):
        """
        :param pfrom:
        :param pto:
        :param objcm:
        :return:
        author: weiwei
        date: 20190805
        """
        tmptrimesh = objcm.trimesh.copy()
        tmptrimesh.apply_transform(objcm.gethomomat())
        geom = dh.pandageom_from_vf(tmptrimesh.vertices, tmptrimesh.face_normals, tmptrimesh.faces)
        targetobjmesh = BulletTriangleMesh()
        targetobjmesh.addGeom(geom)
        bullettmshape = BulletTriangleMeshShape(targetobjmesh, dynamic=True)
        bullettmshape.setMargin(1e-6)
        targetobjmeshnode = BulletRigidBodyNode('facet')
        targetobjmeshnode.addShape(bullettmshape)
        self._bltworld.attach(targetobjmeshnode)
        result = self._bltworld.rayTestClosest(dh.npv3_to_pdv3(pfrom), dh.npv3_to_pdv3(pto))
        self._bltworld.removeRigidBody(targetobjmeshnode)
        if result.hasHit():
            return [dh.pdv3_to_npv3(result.getHitPos()), dh.pdv3_to_npv3(result.getHitNormal())]
        else:
            return [None, None]

    def rayhit_all(self, pfrom, pto, objcm):
        """
        :param pfrom:
        :param pto:
        :param objcm:
        :return:
        author: weiwei
        date: 20190805
        """
        tmptrimesh = objcm.trimesh.copy()
        tmptrimesh.apply_transform(objcm.gethomomat())
        geom = dh.pandageom_from_vf(tmptrimesh.vertices, tmptrimesh.face_normals, tmptrimesh.faces)
        targetobjmesh = BulletTriangleMesh()
        targetobjmesh.addGeom(geom)
        bullettmshape = BulletTriangleMeshShape(targetobjmesh, dynamic=True)
        bullettmshape.setMargin(1e-6)
        targetobjmeshnode = BulletRigidBodyNode('facet')
        targetobjmeshnode.addShape(bullettmshape)
        self._bltworld.attach(targetobjmeshnode)
        result = self._bltworld.rayTestAll(dh.npv3_to_pdv3(pfrom), dh.npv3_to_pdv3(pto))
        self._bltworld.removeRigidBody(targetobjmeshnode)
        if result.hasHits():
            allhits = []
            for hit in result.getHits():
                allhits.append([dh.pdv3_to_npv3(hit.getHitPos()), dh.pdv3_to_npv3(-hit.getHitNormal())])
            return allhits
        else:
            return []

    def show(self, objcm):
        """
        show the collision meshes of the given objects
        :param objcm environment.collisionmodel
        :return:
        author: weiwei
        date: 20190313
        """
        if not self._toggledebug:
            print("Toggle debug on during defining the XCMchecker object to use showfunctions!")
            return
        if not self._is_updatebltworld_added:
            self.taskMgr.add(self._updateworld, "updateworld", extraArgs=[self._bltworld], appendTask=True)
        objcmmeshbullnode = _gen_cdmesh(objcm)
        self._bltworld.attach(objcmmeshbullnode)
        self._bltworldrigidbodylist.append(objcmmeshbullnode)
        self._debugNP.show()

    def showlist(self, objcmlist):
        """
        show the collision meshes of the given objects
        :param objcmlist environment.collisionmodel
        author: weiwei
        date: 20190313
        :return:
        """
        if not self._toggledebug:
            print("Toggle debug on during defining the XCMchecker object to use showfunctions!")
            return
        if not self._is_updatebltworld_added:
            base.taskMgr.add(self._updateworld, "updateworld", extraArgs=[self._bltworld], appendTask=True)
        objcmmeshbullnode = _gen_cdmesh_list(objcmlist)
        self._bltworld.attach(objcmmeshbullnode)
        self._bltworldrigidbodylist.append(objcmmeshbullnode)
        self._debugNP.show()


# functions
def _gen_boxcdmesh(obstaclecm, name='autogen'):
    """
    generate a bullet cd obj using the AABB boundary of a obstacle collision model
    :param obstaclecm: a collision model
    :return: bulletrigidbody
    author: weiwei
    date: 20190313, toyonaka
    """
    if obstaclecm.type is not "box":
        raise Exception("Wrong obstaclecm type! Box is required to genBulletCDBox.")
    bulletboxnode = BulletRigidBodyNode(name)
    cdsolid = obstaclecm.cdcn.getSolid(0)
    bulletboxshape = BulletBoxShape.makeFromSolid(cdsolid)
    rotmat4_pd = obstaclecm.getMat(base.render)
    bulletboxnode.addShape(bulletboxshape,
                           TransformState.makeMat(rotmat4_pd).
                           setPos(rotmat4_pd.xformPoint(cdsolid.getCenter())))
    return bulletboxnode


def _gen_boxcdmesh_list(obstaclecmlist, name='autogen'):
    """
    generate a bullet cd obj using the AABB boundaries stored in obstacle collision models
    :param obstaclecmlist: a list of collision models (cmshare doesnt work!)
    :return: bulletrigidbody
    author: weiwei
    date: 20190313, toyonaka
    """
    bulletboxlistnode = BulletRigidBodyNode(name)
    for obstaclecm in obstaclecmlist:
        if obstaclecm.type is not "box":
            raise Exception("Wrong obstaclecm type! Box is required to genBulletCDBox.")
        cdsolid = obstaclecm.cdcn.getSolid(0)
        bulletboxshape = BulletBoxShape.makeFromSolid(cdsolid)
        rotmatpd4 = obstaclecm.getMat(base.render)
        bulletboxlistnode.addShape(bulletboxshape,
                                   TransformState.makeMat(rotmatpd4).
                                   setPos(rotmatpd4.xformPoint(cdsolid.getCenter())))
    return bulletboxlistnode


def _gen_cdmesh(objcm, basenodepath=None, name='autogen'):
    """
    generate the collision mesh of a nodepath using nodepath
    this function suppose the nodepath has multiple models with many geomnodes
    use genCollisionMeshMultiNp instead of genCollisionMeshNp for generality
    :param nodepath: the panda3d nodepath of the object
    :param basenodepath: the nodepath to compute relative transform, identity if none
    :param name: the name of the rigidbody
    :return: bulletrigidbody
    author: weiwei
    date: 20161212, tsukuba
    """
    gndcollection = objcm.objnp.findAllMatches("+GeomNode")
    geombullnode = BulletRigidBodyNode(name)
    for gnd in gndcollection:
        geom = copy.deepcopy(gnd.node().getGeom(0))
        geomtf = gnd.getTransform(base.render)
        if basenodepath is not None:
            geomtf = gnd.getTransform(basenodepath)
        geombullmesh = BulletTriangleMesh()
        geombullmesh.addGeom(geom)
        bullettmshape = BulletTriangleMeshShape(geombullmesh, dynamic=True)
        bullettmshape.setMargin(0)
        geombullnode.addShape(bullettmshape, geomtf)
    return geombullnode


def _gen_cdmesh_list(objcmlist, basenodepath=None, name='autogen'):
    """
    generate the collision mesh of a nodepath using nodepathlist
    this function suppose the nodepathlist is a list of models with many geomnodes
    "Multi" means each nodepath in the nodepath list may have multiple nps (parent-child relations)
    use genCollisionMeshMultiNp instead if the meshes have parent-child relations
    :param nodepathlist: panda3d nodepathlist
    :param basenodepath: the nodepath to compute relative transform, identity if none
    :param name: the name of the rigidbody
    :return: bulletrigidbody
    author: weiwei
    date: 20190514
    """
    geombullnode = BulletRigidBodyNode(name)
    for objcm in objcmlist:
        gndcollection = objcm.objnp.findAllMatches("+GeomNode")
        for gnd in gndcollection:
            geom = copy.deepcopy(gnd.node().getGeom(0))
            geomtf = gnd.getTransform(base.render)
            if basenodepath is not None:
                geomtf = gnd.getTransform(basenodepath)
            geombullmesh = BulletTriangleMesh()
            geombullmesh.addGeom(geom)
            bullettmshape = BulletTriangleMeshShape(geombullmesh, dynamic=True)
            bullettmshape.setMargin(0)
            geombullnode.addShape(bullettmshape, geomtf)
    return geombullnode


def _gen_cdmesh_from_geom(geom, name='autogen'):
    """
    generate the collision mesh of a nodepath using geom
    :param geom: the panda3d geom of the object
    :param basenodepath: the nodepath to compute relative transform
    :return: bulletrigidbody
    author: weiwei
    date: 20161212, tsukuba
    """
    geomtf = TransformState.makeIdentity()
    geombullmesh = BulletTriangleMesh()
    geombullmesh.addGeom(geom)
    geombullnode = BulletRigidBodyNode(name)
    bullettmshape = BulletTriangleMeshShape(geombullmesh, dynamic=True)
    bullettmshape.setMargin(0)
    geombullnode.addShape(bullettmshape, geomtf)
    return geombullnode


def _gen_plane_cdmesh(updirection=np.array([0, 0, 1]), offset=0, name='autogen'):
    """
    generate a plane bulletrigidbody node
    :param updirection: the normal parameter of bulletplaneshape at panda3d
    :param offset: the d parameter of bulletplaneshape at panda3d
    :param name:
    :return: bulletrigidbody
    author: weiwei
    date: 20170202, tsukuba
    """
    bulletplnode = BulletRigidBodyNode(name)
    bulletplshape = BulletPlaneShape(Vec3(updirection[0], updirection[1], updirection[2]), offset)
    bulletplshape.setMargin(0)
    bulletplnode.addShape(bulletplshape)
    return bulletplnode


def _rayhit(pfrom, pto, geom):
    """
    TODO: To be deprecated, 20201119
    NOTE: this function is quite slow
    find the nearest collision point between vec(pto-pfrom) and the mesh of nodepath
    :param pfrom: starting point of the ray, Point3
    :param pto: ending point of the ray, Point3
    :param geom: meshmodel, a panda3d datatype
    :return: None or Point3
    author: weiwei
    date: 20161201
    """
    bulletworld = BulletWorld()
    facetmesh = BulletTriangleMesh()
    facetmesh.addGeom(geom)
    facetmeshnode = BulletRigidBodyNode('facet')
    bullettmshape = BulletTriangleMeshShape(facetmesh, dynamic=True)
    bullettmshape.setMargin(1e-6)
    facetmeshnode.addShape(bullettmshape)
    bulletworld.attach(facetmeshnode)
    result = bulletworld.rayTestClosest(pfrom, pto)
    return result.getHitPos() if result.hasHit() else None


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometricmodel as gm
    import modeling.collisionmodel as cm
    import math
    import numpy as np

    wd.World(camp=[1.0, 1,.0, 1.0], lookatpos=[0, 0, 0])
    objcm = cm.CollisionModel("./objects/bunnysim.stl")
    homomat = np.eye(4)
    homomat[:3, :3] = rm.rotmat_from_axangle([0, 0, 1], math.pi/2)
    homomat[:3, 3] = np.array([0, 0, 0])
    objcm.sethomomat(homomat)
    pfrom = np.array([0, 0, 0]) + np.array([1.0, 1.0, 1.0])
    pto = np.array([0, 0, 0]) + np.array([-1.0, -1.0, -0.9])
    mcmc = MChecker(toggledebug=True)
    hitpos, hitnrml = mcmc.rayhit_closet(pfrom=pfrom, pto=pto, objcm=objcm)
    objcm.reparent_to(base.render)
    gm.gensphere(hitpos, radius=.003, rgba=np.array([0, 1, 1, 1])).reparent_to(base.render)
    gm.genstick(spos=pfrom, epos=pto, thickness=.002).reparent_to(base.render)
    gm.genarrow(spos=hitpos, epos=hitpos + hitnrml * .07, thickness=.002, rgba=np.array([0, 1, 0, 1])).reparent_to(
        base.render)
    base.run()
