import itertools
import os
import math
import numpy as np
import scipy.spatial as ss
import basis.robot_math as rm
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletTriangleMesh
from panda3d.bullet import BulletTriangleMeshShape
from panda3d.bullet import BulletWorld
from panda3d.core import *
from shapely.geometry import Point
from shapely.geometry import Polygon
from sklearn.cluster import KMeans
from sklearn.neighbors import RadiusNeighborsClassifier
from . import segmentation as sg


class AntipodalContactPairs(object):

    def __init__(self,
                 objcm,
                 max_normal_bias_angle=math.pi/12,
                 min_distance_to_facet_edge=.003,
                 max_distance_to_facet_edge=30,
                 distance_adjacent_contacts=.01,
                 antipodal_angle=-0.7,
                 penetration_depth=5,
                 object_mass=20.0,
                 toggle_soft_finger_contact=True,
                 toggle_debug=True):
        """

        :param objcm:
        :param max_normal_bias_angle:
        :param facet_angle:
        :param min_distance_to_facet_edge:
        :param max_distance_to_facet_edge:
        :param distance_adjacent_contacts:
        :param antipodal_angle:
        :param penetration_depth:
        :param object_mass:
        :param toggle_soft_finger_contact:
        author: weiwei
        date: 20190525osaka, 20210119
        """
        if toggle_debug:
            import time
        self.objtrm = self.objcm.trimesh
        # generate facets
        if toggle_debug:
            tic = time.time()
        self.seg_nested_face_id_list, self.seg_seed_face_id_list, self.seg_normal_list, self.seg_curvature_list = \
            sg.over_segmentation(objcm, max_normal_bias_angle=max_normal_bias_angle)
        if toggle_debug:
            toc = time.time()
            print("facet cost", toc - tic)
        # sample points
        if toggle_debug:
            tic = time.time()
        self.raw_contact_points, self.raw_contact_point_face_ids = self.objtrm.sample_surface()
        self.raw_contact_normals = self.objtrm.face_normals[self.contact_point_face_ids.tolist()]
        if toggle_debug:
            toc = time.time()
            print("sampling cost", toc - tic)
        # remove points near boundaries
        if toggle_debug:
            tic = time.time()
        samples_for_edge_detection, _ = self.objtrm.sample_surface(radius=distance_adjacent_contacts)
        kdt = ss.cKDTree(samples_for_edge_detection)
        kdt.query_ball_point(, )
        self.contact_points = None
        self.samplenrmls_ref = None
        # facet2dbdries saves the 2d boundaries of each facet
        self.facet2dbdries = None
        self.removeBadSamples(mindist=refine1min, maxdist=refine1max)
        if toggle_debug:
            toc = time.time()
            print("remove bad sample cost", toc - tic)

        # the sampled points (clustered)
        tic = time.time()
        self.samplepnts_refcls = None
        self.samplenrmls_refcls = None
        self.clusterFacetSamplesRNN(reduceRadius=refine2radius)
        toc = time.time()
        print("cluster samples cost", toc - tic)

        # plan contact pairs
        self.facetpairs = None
        self.contactpairs = None
        self.contactpairnormals = None
        self.contactpairfacets = None
        self.contactrotangles = None
        # for three fingers
        self.realcontactpairs = None
        tic = time.time()
        self.plan_contact_pairs(hmax, fpairparallel, objmass, oppositeoffset=oppositeoffset, bypasssoftfgr=bypasssoftfgr)
        toc = time.time()
        print("plan contact pairs cost", toc - tic)

        # for plot
        self.counter = 0
        # self.facetcolorarray = pandageom.randomColorArray(self.facets.shape[0], nonrandcolor = [.5,.5,.7,1])
        self.facetcolorarray = pandageom.randomColorArray(self.facets.shape[0])

    def sampleObjModel(self, numpointsoververts=5):
        """
        sample the object model
        self.samplepnts and self.samplenrmls
        are filled in this function

        :param: numpointsoververts: the number of sampled points = numpointsoververts*mesh.vertices.shape[0]
        :return: nverts: the number of verts sampled

        author: weiwei
        date: 20160623 flight to tokyo
        """

        nverts = self.objtrm.vertices.shape[0]
        count = 1000 if nverts * numpointsoververts > 1000 else nverts * numpointsoververts
        samples, face_idx = sample.sample_surface_even_withfaceid(self.objtrm, count)
        # print nverts
        self.samplepnts = np.ndarray(shape=(self.facets.shape[0],), dtype=np.object)
        self.samplenrmls = np.ndarray(shape=(self.facets.shape[0],), dtype=np.object)
        for i, faces in enumerate(self.facets):
            for face in faces:
                sample_idx = np.where(face_idx == face)[0]
                if len(sample_idx) > 0:
                    if self.samplepnts[i] is not None:
                        self.samplepnts[i] = np.vstack((self.samplepnts[i], samples[sample_idx]))
                        self.samplenrmls[i] = np.vstack((self.samplenrmls[i],
                                                         [self.objtrm.face_normals[face]] * samples[sample_idx].shape[
                                                             0]))
                    else:
                        self.samplepnts[i] = np.array(samples[sample_idx])
                        self.samplenrmls[i] = np.array([self.objtrm.face_normals[face]] * samples[sample_idx].shape[0])
            if self.samplepnts[i] is None:
                self.samplepnts[i] = np.empty(shape=[0, 0])
                self.samplenrmls[i] = np.empty(shape=[0, 0])
        return nverts

    def removeBadSamples(self, mindist=2, maxdist=20):
        """
        Do the following refinement:
        (1) remove the samples who's minimum distance to facet boundary is smaller than mindist
        (2) remove the samples who's maximum distance to facet boundary is larger than mindist

        ## input
        mindist, maxdist
            as explained in the begining

        author: weiwei
        date: 20160623 flight to tokyo
        """

        # ref = refine
        self.samplepnts_ref = np.ndarray(shape=(self.facets.shape[0],), dtype=np.object)
        self.samplenrmls_ref = np.ndarray(shape=(self.facets.shape[0],), dtype=np.object)
        self.facet2dbdries = []
        for i, faces in enumerate(self.facets):
            # print "removebadsample"
            # print i,len(self.facets)
            facetp = None
            face0verts = self.objtrm.vertices[self.objtrm.faces[faces[0]]]
            facetmat = robotmath.rotmat_from_normalandpoints(self.facetnormals[i], face0verts[0], face0verts[1])
            # face samples
            samplepntsp = []
            for j, apnt in enumerate(self.samplepnts[i]):
                apntp = np.dot(facetmat, apnt)[:2]
                samplepntsp.append(apntp)
            # face boundaries
            for j, faceidx in enumerate(faces):
                vert0 = self.objtrm.vertices[self.objtrm.faces[faceidx][0]]
                vert1 = self.objtrm.vertices[self.objtrm.faces[faceidx][1]]
                vert2 = self.objtrm.vertices[self.objtrm.faces[faceidx][2]]
                vert0p = np.dot(facetmat, vert0)[:2]
                vert1p = np.dot(facetmat, vert1)[:2]
                vert2p = np.dot(facetmat, vert2)[:2]
                facep = Polygon([vert0p, vert1p, vert2p])
                if facetp is None:
                    facetp = facep
                else:
                    try:
                        facetp = facetp.union(facep)
                    except:
                        continue
            self.facet2dbdries.append(facetp)
            selectedele = []
            for j, apntp in enumerate(samplepntsp):
                try:
                    apntpnt = Point(apntp[0], apntp[1])
                    dbnds = []
                    dbnds.append(apntpnt.distance(facetp.exterior))
                    for fpinter in facetp.interiors:
                        dbnds.append(apntpnt.distance(fpinter))
                    dbnd = min(dbnds)
                    if dbnd < mindist or dbnd > maxdist:
                        pass
                    else:
                        selectedele.append(j)
                except:
                    pass
            self.samplepnts_ref[i] = np.asarray([self.samplepnts[i][j] for j in selectedele])
            self.samplenrmls_ref[i] = np.asarray([self.samplenrmls[i][j] for j in selectedele])
        self.facet2dbdries = np.array(self.facet2dbdries)

    def clusterFacetSamplesKNN(self, reduceRatio=3, maxNPnts=5):
        """
        cluster the samples of each facet using k nearest neighbors
        the cluster center and their correspondent normals will be saved
        in self.samplepnts_refcls and self.objsamplenrmals_refcls

        :param: reduceRatio: the ratio of points to reduce
        :param: maxNPnts: the maximum number of points on a facet
        :return: None

        author: weiwei
        date: 20161129, tsukuba
        """

        self.samplepnts_refcls = np.ndarray(shape=(self.facets.shape[0],), dtype=np.object)
        self.samplenrmls_refcls = np.ndarray(shape=(self.facets.shape[0],), dtype=np.object)
        for i, facet in enumerate(self.facets):
            self.samplepnts_refcls[i] = np.empty(shape=(0, 0))
            self.samplenrmls_refcls[i] = np.empty(shape=(0, 0))
            X = self.samplepnts_ref[i]
            nX = X.shape[0]
            if nX > reduceRatio:
                kmeans = KMeans(n_clusters=maxNPnts if nX / reduceRatio > maxNPnts else nX / reduceRatio,
                                random_state=0).fit(X)
                self.samplepnts_refcls[i] = kmeans.cluster_centers_
                self.samplenrmls_refcls[i] = np.tile(self.facetnormals[i], [self.samplepnts_refcls.shape[0], 1])

    def clusterFacetSamplesRNN(self, reduceRadius=3):
        """
        cluster the samples of each facet using radius nearest neighbours
        the cluster center and their correspondent normals will be saved
        in self.samplepnts_refcls and self.objsamplenrmals_refcls

        :param: reduceRadius: the neighbors that fall inside the reduceradius will be removed
        :return: None

        author: weiwei
        date: 20161130, osaka
        """

        self.samplepnts_refcls = np.ndarray(shape=(self.facets.shape[0],), dtype=np.object)
        self.samplenrmls_refcls = np.ndarray(shape=(self.facets.shape[0],), dtype=np.object)
        for i, facet in enumerate(self.facets):
            self.samplepnts_refcls[i] = []
            self.samplenrmls_refcls[i] = []
            X = self.samplepnts_ref[i]
            np.random.shuffle(X)
            nX = X.shape[0]
            if nX > 0:
                neigh = RadiusNeighborsClassifier(radius=1.0)
                neigh.fit(X, range(nX))
                neigharrays = neigh.radius_neighbors(X, radius=reduceRadius, return_distance=False)
                delset = set([])
                for j in range(nX):
                    if j not in delset:
                        self.samplepnts_refcls[i].append(np.array(X[j]))
                        self.samplenrmls_refcls[i].append(np.array(self.samplenrmls_ref[i][j]))
                        delset.update(neigharrays[j].tolist())
            if self.samplepnts_refcls[i]:
                self.samplepnts_refcls[i] = np.vstack(self.samplepnts_refcls[i])
                self.samplenrmls_refcls[i] = np.vstack(self.samplenrmls_refcls[i])
            else:
                self.samplepnts_refcls[i] = np.empty(shape=(0, 0))
                self.samplenrmls_refcls[i] = np.empty(shape=(0, 0))

    def plan_contact_pairs(self, hmax=10, fpairparallel=-0.8, objmass=20.0, fgrtipdist=82, oppositeoffset=0,
                           bypasssoftfgr=True):
        """
        find the grasps using parallel pairs

        :param: hmax a parameter used to control the stability of the planned grasps
        :param: fpairparallel a parameter used to control the parallelity of two facets
        :param: fgrtipdist the maximum dist between finger tips
        :param: oppositeoffset a positive value indicating the offset of two opposing fingers (contact centers are not along a line)
        :return:

        author: weiwei
        date: 20161130, harada office @ osaka university
        """

        # note that pairnormals and pairfacets are duplicated for each contactpair
        # the duplication is performed on purpose for convenient access
        # also, each contactpair"s" corresponds to a facetpair
        # it is empty when no contactpair is available
        self.contactpairs = []
        # contactpairnormals and contactpairfacets are not helpful
        # they are kept for convenience (they could be accessed using facetnormals and facetpairs)
        self.contactpairnormals = []
        self.contactpairfacets = []
        # facetparis for update
        updatedfacetpairs = []

        # for three fingers
        self.realcontactpairs = []

        # for raytracing
        bulletworldray = BulletWorld()
        nfacets = self.facets.shape[0]
        self.facetpairs = list(itertools.combinations(range(nfacets), 2))
        for facetpair in self.facetpairs:
            # if one of the facet doesnt have samples, jump to next
            if self.samplepnts_refcls[facetpair[0]].shape[0] is 0 or \
                    self.samplepnts_refcls[facetpair[1]].shape[0] is 0:
                # print "no sampled points"
                continue
            # check if the faces are opposite and parallel
            dotnorm = np.dot(self.facetnormals[facetpair[0]], self.facetnormals[facetpair[1]])
            if dotnorm < fpairparallel:
                tempcontactpairs = []
                tempcontactpairnormals = []
                tempcontactpairfacets = []
                # for the virtual pair of 3-f grippers
                temprealcontactpairs = []
                # check if any samplepnts's projection from facetpairs[i][0] falls in the polygon of facetpairs[i][1]
                facet0pnts = self.samplepnts_refcls[facetpair[0]]
                facet0normal = self.facetnormals[facetpair[0]]
                facet1normal = self.facetnormals[facetpair[1]]
                # generate collision mesh
                facetmesh = BulletTriangleMesh()
                faceidsonfacet = self.facets[facetpair[1]]
                geom = pandageom.packpandageom_fn(self.objtrm.vertices,
                                                  self.objtrm.face_normals[faceidsonfacet],
                                                  self.objtrm.faces[faceidsonfacet])
                facetmesh.addGeom(geom)
                facetmeshbullnode = BulletRigidBodyNode('facet')
                facetmeshbullnode.addShape(BulletTriangleMeshShape(facetmesh, dynamic=True))
                bulletworldray.attachRigidBody(facetmeshbullnode)
                # check the projection of a ray
                for facet0pnt in facet0pnts:
                    # p3dh.gensphere(pos=p3dh.npv3_to_pdv3(facet0pnt), radius=1).reparentTo(base.render)
                    ishitted = False
                    if oppositeoffset == 0:
                        pFrom = Point3(facet0pnt[0], facet0pnt[1], facet0pnt[2])
                        pTo = pFrom + p3dh.npv3_to_pdv3(facet1normal) * 9999
                        result = bulletworldray.rayTestClosest(pFrom, pTo)
                        if result.hasHit():
                            ishitted = True
                    else:
                        orthogonalv = robotmath.orthogonalunitvec(facet1normal)
                        # initrotmat = np.eye(3)
                        # initrotmat[:3,0] = orthogonalv
                        # initrotmat[:3,1] = np.cross(facet1normal, orthogonalv)
                        # initrotmat[:3,2] = facet1normal
                        # print(initrotmat)
                        for angle in range(0, 360, 15):
                            shiftpf = np.dot(robotmath.rodrigues(facet1normal, angle), orthogonalv) * oppositeoffset
                            pFrom = p3dh.npv3_to_pdv3(facet0pnt + shiftpf)
                            pTo = pFrom + p3dh.npv3_to_pdv3(facet1normal) * 9999
                            # p3dh.genarrow(spos=p3dh.pdv3_to_npv3(pFrom), epos=p3dh.pdv3_to_npv3(pTo)).reparentTo(base.render)
                            result = bulletworldray.rayTestClosest(pFrom, pTo)
                            if result.hasHit():
                                ishitted = True
                                break
                    if ishitted:
                        hitposorigin = result.getHitPos()
                        fgdist = np.linalg.norm(p3dh.pdv3_to_npv3(pFrom - hitposorigin))
                        # p3dh.gensphere(pos=hitposorigin, radius=1, rgba=[0,1,0,1]).reparentTo(base.render)
                        # self.objcm.reparentTo(base.render)
                        # base.run()
                        # print("fgdist", fgdist)
                        hitpos = p3dh.npv3_to_pdv3(facet0pnt + facet1normal * fgdist)
                        if fgdist < fgrtipdist and fgdist > 10:
                            fgrcenter = (np.array(facet0pnt.tolist()) + np.array(
                                [hitpos[0], hitpos[1], hitpos[2]])) / 2.0
                            # avoid large torque
                            curvature0 = self.facetcurvatures[facetpair[0]]
                            curvature1 = self.facetcurvatures[facetpair[1]]
                            curvature = curvature0
                            if curvature1 < curvature:
                                curvature = curvature1
                            facet1normal = self.facetnormals[facetpair[1]]
                            if bypasssoftfgr or np.linalg.norm(
                                    self.objtrm.center_mass - fgrcenter) / objmass < hmax / curvature - hmax * hmax:
                                tempcontactpairs.append(
                                    [np.array(facet0pnt.tolist()), np.array([hitpos[0], hitpos[1], hitpos[2]])])
                                tempcontactpairnormals.append(
                                    [np.array([facet0normal[0], facet0normal[1], facet0normal[2]]),
                                     np.array([facet1normal[0], facet1normal[1], facet1normal[2]])])
                                tempcontactpairfacets.append(facetpair)
                                temprealcontactpairs.append([np.array(facet0pnt.tolist()), np.array(
                                    [hitposorigin[0], hitposorigin[1], hitposorigin[2]])])
                bulletworldray.removeRigidBody(facetmeshbullnode)
                if len(tempcontactpairs) > 0:
                    updatedfacetpairs.append(facetpair)
                    self.contactpairs.append(tempcontactpairs)
                    self.contactpairnormals.append(tempcontactpairnormals)
                    self.contactpairfacets.append(tempcontactpairnormals)
                    self.realcontactpairs.append(temprealcontactpairs)

        # update the facet pairs
        self.facetpairs = updatedfacetpairs

    def segShow(self, base, togglesamples=False, togglenormals=False,
                togglesamples_ref=False, togglenormals_ref=False,
                togglesamples_refcls=False, togglenormals_refcls=False, alpha=.1):
        """

        :param base:
        :param togglesamples:
        :param togglenormals:
        :param togglesamples_ref: toggles the sampled points that fulfills the dist requirements
        :param togglenormals_ref:
        :return:
        """

        nfacets = self.facets.shape[0]
        facetcolorarray = self.facetcolorarray

        # offsetf = facet
        plotoffsetf = .0
        # plot the segments
        print("number of facets", len(self.facets))
        print("average triangles", np.array([len(facet) for facet in self.facets]).mean())
        for i, facet in enumerate(self.facets):
            geom = pandageom.packpandageom_fn(self.objtrm.vertices + np.tile(plotoffsetf * i * self.facetnormals[i],
                                                                             [self.objtrm.vertices.shape[0], 1]),
                                              self.objtrm.face_normals[facet], self.objtrm.faces[facet])
            node = GeomNode('piece')
            node.addGeom(geom)
            star = NodePath('piece')
            star.attachNewNode(node)
            star.setColor(Vec4(facetcolorarray[i][0], facetcolorarray[i][1],
                               facetcolorarray[i][2], alpha))
            star.setTransparency(TransparencyAttrib.MAlpha)

            star.setTwoSided(True)
            star.reparentTo(base.render)
            # sampledpnts = samples[sample_idxes[i]]
            # for apnt in sampledpnts:
            #     pandageom.plotSphere(base, star, pos=apnt, radius=1, rgba=rgba)
            rgbapnts0 = [1, 1, 1, 1]
            rgbapnts1 = [.5, .5, 0, 1]
            rgbapnts2 = [1, 0, 0, 1]
            if togglesamples:
                for j, apnt in enumerate(self.samplepnts[i]):
                    base.pggen.plotSphere(star, pos=apnt + plotoffsetf * i * self.facetnormals[i], radius=3,
                                          rgba=rgbapnts0)
            if togglenormals:
                for j, apnt in enumerate(self.samplepnts[i]):
                    base.pggen.plotArrow(star, spos=apnt + plotoffsetf * i * self.facetnormals[i],
                                         epos=apnt + plotoffsetf * i * self.facetnormals[i] + self.samplenrmls[i][j],
                                         rgba=rgbapnts0, length=10)
            if togglesamples_ref:
                for j, apnt in enumerate(self.samplepnts_ref[i]):
                    base.pggen.plotSphere(star, pos=apnt + plotoffsetf * i * self.facetnormals[i], radius=3,
                                          rgba=rgbapnts1)
            if togglenormals_ref:
                for j, apnt in enumerate(self.samplepnts_ref[i]):
                    base.pggen.plotArrow(star, spos=apnt + plotoffsetf * i * self.facetnormals[i],
                                         epos=apnt + plotoffsetf * i * self.facetnormals[i] + self.samplenrmls_ref[i][
                                             j],
                                         rgba=rgbapnts1, length=10)
            if togglesamples_refcls:
                for j, apnt in enumerate(self.samplepnts_refcls[i]):
                    base.pggen.plotSphere(star, pos=apnt + plotoffsetf * i * self.facetnormals[i], radius=3,
                                          rgba=rgbapnts2)
            if togglenormals_refcls:
                for j, apnt in enumerate(self.samplepnts_refcls[i]):
                    base.pggen.plotArrow(star, spos=apnt + plotoffsetf * i * self.facetnormals[i],
                                         epos=apnt + plotoffsetf * i * self.facetnormals[i] +
                                              self.samplenrmls_refcls[i][j],
                                         rgba=rgbapnts2, length=10)

    def segShow2(self, base, togglesamples=False, togglenormals=False,
                 togglesamples_ref=False, togglenormals_ref=False,
                 togglesamples_refcls=False, togglenormals_refcls=False, specificface=True):
        """

        :param base:
        :param togglesamples:
        :param togglenormals:
        :param togglesamples_ref: toggles the sampled points that fulfills the dist requirements
        :param togglenormals_ref:
        :return:
        """

        nfacets = self.facets.shape[0]
        facetcolorarray = self.facetcolorarray

        rgbapnts0 = [1, 1, 1, 1]
        rgbapnts1 = [0, 0, 1, 1]
        rgbapnts2 = [1, 0, 0, 1]

        # offsetf = facet
        plotoffsetf = .0
        faceplotted = False
        # plot the segments
        for i, facet in enumerate(self.facets):
            if not specificface:
                geom = pandageom.packpandageom_fn(self.objtrm.vertices + np.tile(plotoffsetf * i * self.facetnormals[i],
                                                                                 [self.objtrm.vertices.shape[0], 1]),
                                                  self.objtrm.face_normals[facet], self.objtrm.faces[facet])
                node = GeomNode('piece')
                node.addGeom(geom)
                star = NodePath('piece')
                star.attachNewNode(node)
                star.setColor(Vec4(.77, .17, 0, 1))
                star.setTransparency(TransparencyAttrib.MAlpha)

                star.setTwoSided(True)
                star.reparentTo(base.render)
                # sampledpnts = samples[sample_idxes[i]]
                # for apnt in sampledpnts:
                #     pandageom.plotSphere(base, star, pos=apnt, radius=1, rgba=rgba)
                if togglesamples:
                    for j, apnt in enumerate(self.samplepnts[i]):
                        base.pggen.plotSphere(star, pos=apnt + plotoffsetf * i * self.facetnormals[i], radius=2.8,
                                              rgba=rgbapnts0)
                if togglenormals:
                    for j, apnt in enumerate(self.samplepnts[i]):
                        base.pggen.plotArrow(star, spos=apnt + plotoffsetf * i * self.facetnormals[i],
                                             epos=apnt + plotoffsetf * i * self.facetnormals[i] + self.samplenrmls[i][
                                                 j],
                                             rgba=rgbapnts0, length=10)
                if togglesamples_ref:
                    for j, apnt in enumerate(self.samplepnts_ref[i]):
                        base.pggen.plotSphere(star, pos=apnt + plotoffsetf * i * self.facetnormals[i], radius=2.9,
                                              rgba=rgbapnts1)
                if togglenormals_ref:
                    for j, apnt in enumerate(self.samplepnts_ref[i]):
                        base.pggen.plotArrow(star, spos=apnt + plotoffsetf * i * self.facetnormals[i],
                                             epos=apnt + plotoffsetf * i * self.facetnormals[i] +
                                                  self.samplenrmls_ref[i][j],
                                             rgba=rgbapnts1, length=10)
                if togglesamples_refcls:
                    for j, apnt in enumerate(self.samplepnts_refcls[i]):
                        base.pggen.plotSphere(star, pos=apnt + plotoffsetf * i * self.facetnormals[i], radius=3,
                                              rgba=rgbapnts2)
                if togglenormals_refcls:
                    for j, apnt in enumerate(self.samplepnts_refcls[i]):
                        base.pggen.plotArrow(star, spos=apnt + plotoffsetf * i * self.facetnormals[i],
                                             epos=apnt + plotoffsetf * i * self.facetnormals[i] +
                                                  self.samplenrmls_refcls[i][j],
                                             rgba=rgbapnts2, length=10)
            if specificface:
                plotoffsetf = .3
                if faceplotted:
                    continue
                else:
                    if len(self.samplepnts[i]) > 85:
                        faceplotted = True
                        geom = pandageom.packpandageom_fn(
                            self.objtrm.vertices + np.tile(plotoffsetf * i * self.facetnormals[i],
                                                           [self.objtrm.vertices.shape[0], 1]),
                            self.objtrm.face_normals[facet], self.objtrm.faces[facet])
                        node = GeomNode('piece')
                        node.addGeom(geom)
                        star = NodePath('piece')
                        star.attachNewNode(node)
                        star.setColor(Vec4(facetcolorarray[i][0], facetcolorarray[i][1], facetcolorarray[i][2], 1))
                        star.setColor(Vec4(.7, .3, .3, 1))
                        star.setTransparency(TransparencyAttrib.MAlpha)

                        star.setTwoSided(True)
                        star.reparentTo(base.render)
                        # sampledpnts = samples[sample_idxes[i]]
                        # for apnt in sampledpnts:
                        #     pandageom.plotSphere(base, star, pos=apnt, radius=1, rgba=rgba)
                        if togglesamples:
                            for j, apnt in enumerate(self.samplepnts[i]):
                                base.pggen.plotSphere(star, pos=apnt + plotoffsetf * i * self.facetnormals[i],
                                                      radius=2.8, rgba=rgbapnts0)
                        if togglenormals:
                            for j, apnt in enumerate(self.samplepnts[i]):
                                base.pggen.plotArrow(star, spos=apnt + plotoffsetf * i * self.facetnormals[i],
                                                     epos=apnt + plotoffsetf * i * self.facetnormals[i] +
                                                          self.samplenrmls[i][j],
                                                     rgba=rgbapnts0, length=10)
                        if togglesamples_ref:
                            for j, apnt in enumerate(self.samplepnts_ref[i]):
                                base.pggen.plotSphere(star, pos=apnt + plotoffsetf * i * self.facetnormals[i],
                                                      radius=2.9, rgba=rgbapnts1)
                        if togglenormals_ref:
                            for j, apnt in enumerate(self.samplepnts_ref[i]):
                                base.pggen.plotArrow(star, spos=apnt + plotoffsetf * i * self.facetnormals[i],
                                                     epos=apnt + plotoffsetf * i * self.facetnormals[i] +
                                                          self.samplenrmls_ref[i][j],
                                                     rgba=rgbapnts1, length=10)
                        if togglesamples_refcls:
                            for j, apnt in enumerate(self.samplepnts_refcls[i]):
                                base.pggen.plotSphere(star, pos=apnt + plotoffsetf * i * self.facetnormals[i], radius=3,
                                                      rgba=rgbapnts2)
                        if togglenormals_refcls:
                            for j, apnt in enumerate(self.samplepnts_refcls[i]):
                                base.pggen.plotArrow(star, spos=apnt + plotoffsetf * i * self.facetnormals[i],
                                                     epos=apnt + plotoffsetf * i * self.facetnormals[i] +
                                                          self.samplenrmls_refcls[i][j],
                                                     rgba=rgbapnts2, length=10)

    def pairShow(self, base, togglecontacts=False, togglecontactnormals=False, togglerealcontacts=False):
        # the following sentence requires segshow to be executed first
        facetcolorarray = self.facetcolorarray
        # offsetfp = facetpair
        # plotoffsetfp = np.random.random()*50
        plotoffsetfp = 0.0
        # plot the pairs and their contacts
        # for i in range(self.counter+1, len(self.facetpairs)):
        #     if self.contactpairs[i]:
        #         self.counter = i
        #         break
        # if i is len(self.facetpairs):
        #     return
        # delete the facetpair after show
        # np0 = base.render.find("**/pair0")
        # if np0:
        #     np0.removeNode()
        # np1 = base.render.find("**/pair1")
        # if np1:
        #     np1.removeNode()
        print("counter", self.counter)
        print(self.counter, len(self.facetpairs))
        if self.counter >= len(self.facetpairs):
            # self.counter = 0
            return False
        # n = 500
        # if self.counter > n or self.counter <= n-1:
        #     # self.counter = 0
        #     return
        cttpnt0, cttpnt1 = self.contactpairs[self.counter][0]
        cttnrml0, cttnrml1 = self.contactpairnormals[self.counter][0]
        print(cttpnt1[2], cttpnt0[2])
        if not ((cttpnt1[2] > 10 and cttpnt0[2] < 5) or (cttpnt1[2] < 5 and cttpnt0[2] > 10)):
            self.counter += 1
            return True
        if abs(np.dot(cttnrml0, np.array([0, 0, 1]))) > .5 or abs(np.dot(cttnrml1, np.array([0, 0, 1]))) > .5:
            self.counter += 1
            return True
        if abs(np.dot(cttnrml0, np.array([1, 0, 0]))) < .5 or abs(np.dot(cttnrml1, np.array([1, 0, 0]))) < .5:
            self.counter += 1
            return True
        facetpair = self.facetpairs[self.counter]
        facetidx0 = facetpair[0]
        facetidx1 = facetpair[1]
        geomfacet0 = pandageom.packpandageom_fn(self.objtrm.vertices +
                                                np.tile(plotoffsetfp * self.facetnormals[facetidx0],
                                                        [self.objtrm.vertices.shape[0], 1]),
                                                self.objtrm.face_normals[self.facets[facetidx0]],
                                                self.objtrm.faces[self.facets[facetidx0]])
        geomfacet1 = pandageom.packpandageom_fn(self.objtrm.vertices +
                                                np.tile(plotoffsetfp * self.facetnormals[facetidx1],
                                                        [self.objtrm.vertices.shape[0], 1]),
                                                self.objtrm.face_normals[self.facets[facetidx1]],
                                                self.objtrm.faces[self.facets[facetidx1]])
        # show the facetpair
        node0 = GeomNode('pair0')
        node0.addGeom(geomfacet0)
        star0 = NodePath('pair0')
        star0.attachNewNode(node0)
        star0.setColor(Vec4(facetcolorarray[facetidx0][0], facetcolorarray[facetidx0][1],
                            facetcolorarray[facetidx0][2], facetcolorarray[facetidx0][3]))
        node1 = GeomNode('pair1')
        node1.addGeom(geomfacet1)
        star1 = NodePath('pair1')
        star1.attachNewNode(node1)
        # star1.setColor(Vec4(facetcolorarray[facetidx1][0], facetcolorarray[facetidx1][1],
        #                    facetcolorarray[facetidx1][2], facetcolorarray[facetidx1][3]))
        # set to the same color
        star1.setColor(Vec4(facetcolorarray[facetidx0][0], facetcolorarray[facetidx0][1],
                            facetcolorarray[facetidx0][2], facetcolorarray[facetidx0][3]))
        star0.setTwoSided(True)
        star0.reparentTo(base.render)
        star1.setTwoSided(True)
        star1.reparentTo(base.render)
        if togglecontacts:
            for j, contactpair in enumerate(self.contactpairs[self.counter]):
                cttpnt0 = contactpair[0]
                cttpnt1 = contactpair[1]
                base.pggen.plotSphere(star0, pos=cttpnt0 + plotoffsetfp * self.facetnormals[facetidx0], radius=4,
                                      rgba=[facetcolorarray[facetidx0][0], facetcolorarray[facetidx0][1],
                                            facetcolorarray[facetidx0][2], facetcolorarray[facetidx0][3]])
                # pandageom.plotSphere(star1, pos=cttpnt1+plotoffsetfp*self.facetnormals[facetidx1], radius=4,
                #                      rgba=[facetcolorarray[facetidx1][0], facetcolorarray[facetidx1][1],
                #                            facetcolorarray[facetidx1][2], facetcolorarray[facetidx1][3]])
                # use the same color
                base.pggen.plotSphere(star1, pos=cttpnt1 + plotoffsetfp * self.facetnormals[facetidx1], radius=4,
                                      rgba=[facetcolorarray[facetidx0][0], facetcolorarray[facetidx0][1],
                                            facetcolorarray[facetidx0][2], facetcolorarray[facetidx0][3]])
        if togglecontactnormals:
            for j, contactpair in enumerate(self.contactpairs[self.counter]):
                cttpnt0 = contactpair[0]
                cttpnt1 = contactpair[1]
                base.pggen.plotArrow(star0, spos=cttpnt0 + plotoffsetfp * self.facetnormals[facetidx0],
                                     epos=cttpnt0 + plotoffsetfp * self.facetnormals[facetidx0] +
                                          self.contactpairnormals[self.counter][j][0],
                                     rgba=[facetcolorarray[facetidx0][0], facetcolorarray[facetidx0][1],
                                           facetcolorarray[facetidx0][2], facetcolorarray[facetidx0][3]], length=10)
                p3dh.genarrow(spos=cttpnt0 + plotoffsetfp * self.facetnormals[facetidx0],
                              epos=cttpnt0 + plotoffsetfp * self.facetnormals[facetidx0] +
                                   70 * self.contactpairnormals[self.counter][j][1], thickness=60,
                              rgba=[facetcolorarray[facetidx0][0], facetcolorarray[facetidx0][1] / 2,
                                    facetcolorarray[facetidx0][2] / 2, facetcolorarray[facetidx0][3] * .3]).reparentTo(
                    star0)
                # pandageom.plotArrow(star1,  spos=cttpnt1+plotoffsetfp*self.facetnormals[facetidx1],
                #                 epos=cttpnt1 + plotoffsetfp*self.facetnormals[facetidx1] +
                #                      self.contactpairnormals[self.counter][j][1],
                #                 rgba=[facetcolorarray[facetidx1][0], facetcolorarray[facetidx1][1],
                #                       facetcolorarray[facetidx1][2], facetcolorarray[facetidx1][3]], length=10)
                # use the same color
                base.pggen.plotArrow(star1, spos=cttpnt1 + plotoffsetfp * self.facetnormals[facetidx1],
                                     epos=cttpnt1 + plotoffsetfp * self.facetnormals[facetidx1] +
                                          self.contactpairnormals[self.counter][j][1],
                                     rgba=[facetcolorarray[facetidx0][0], facetcolorarray[facetidx0][1],
                                           facetcolorarray[facetidx0][2], facetcolorarray[facetidx0][3]], length=10)
        if togglerealcontacts:
            for j, contactpair in enumerate(self.realcontactpairs[self.counter]):
                cttpnt0 = contactpair[0]
                cttpnt1 = contactpair[1]
                base.pggen.plotSphere(star0, pos=cttpnt0 + plotoffsetfp * self.facetnormals[facetidx0], radius=4,
                                      rgba=[facetcolorarray[facetidx0][0] / 2, facetcolorarray[facetidx0][1],
                                            facetcolorarray[facetidx0][2] / 2, facetcolorarray[facetidx0][3]])
                # pandageom.plotSphere(star1, pos=cttpnt1+plotoffsetfp*self.facetnormals[facetidx1], radius=4,
                #                      rgba=[facetcolorarray[facetidx1][0], facetcolorarray[facetidx1][1],
                #                            facetcolorarray[facetidx1][2], facetcolorarray[facetidx1][3]])
                # use the same color
                # base.pggen.plotSphere(star1, pos=cttpnt1+plotoffsetfp*self.facetnormals[facetidx1], radius=4,
                #                      rgba=[facetcolorarray[facetidx0][0]/2, facetcolorarray[facetidx0][1],
                #                            facetcolorarray[facetidx0][2]/2, facetcolorarray[facetidx0][3]])
                base.pggen.plotSphere(star1, pos=cttpnt1 + plotoffsetfp * self.facetnormals[facetidx1], radius=4,
                                      rgba=[0, 1, 0, 1])
        import manipulation.grip.hrp5three.hrp5pf3 as h53
        hrp5pf3rgt = h53.newHand(hndid='rgt')
        hrp5pf3rgt.setJawwidth(hrp5pf3rgt.jawwidthopen / 2.0)
        center = (self.contactpairs[self.counter][0][0] + self.contactpairs[self.counter][0][1]) / 2

        # cttnrml = robotmath.unitvec_safe(cttnrml0-cttnrml1)[1]
        cttnrml = robotmath.unitvec_safe(cttnrml0)[1]
        hrp5pf3rgt.gripAt(center[0], center[1], center[2], cttnrml[0], cttnrml[1], cttnrml[2], 50, jawwidth=50)
        hrp5pf3rgt.reparentTo(base.render)
        base.run()
        self.counter += 1
        return True
        # base.run()
        # break
        # except:
        #     print "You might need to loadmodel first!"


if __name__ == '__main__':
    import environment.collisionmodel as cm

    # ax1 = fig.add_subplot(121, projection='3d')
    #
    # mesh = trimesh.load_mesh('./circlestar.obj')
    # samples, face_idx = sample.sample_surface_even(mesh, mesh.vertices.shape[0] * 10)
    # facets, facets_area = mesh.facets(return_area=True)
    # sample_idxes = np.ndarray(shape=(facets.shape[0],),dtype=np.object)
    # for i,faces in enumerate(facets):
    #     sample_idx = np.empty([0,0], dtype=np.int)
    #     for face in faces:
    #         sample_idx = np.append(sample_idx, np.where(face_idx == face)[0])
    #     sample_idxes[i]=sample_idx
    #

    base = pandactrl.World(camp=[0, 300, -400], lookatpos=[0, 0, 0])
    this_dir, this_filename = os.path.split(__file__)
    # objpath = os.path.join(this_dir, "objects", "ttube.stl")
    # objpath = os.path.join(this_dir, "objects", "tool.stl")
    # objpath = os.path.join(this_dir, "objects", "tool_drcdriver.stl")
    # objpath = os.path.join(this_dir, "objects", "planefrontstay.stl")
    # objpath = os.path.join(this_dir, "objects", "planewheel.stl")
    # objpath = os.path.join(this_dir, "objects", "planelowerbody.stl")
    # objpath = os.path.join(this_dir, "objects", "planerearstay.stl")
    # objpath = os.path.join(this_dir, "objects", "sandpart.stl")
    objpath = "../objects/ttube.stl"
    freegriptst = Freecontactpairs(objpath, refine1min=0, refine1max=30, refine2radius=10, faceangle=.95, segangle=.95,
                                   fpairparallel=-0.9, oppositeoffset=30)
    # freegriptst.segShow(base, togglesamples=True, togglenormals=True,
    #                     togglesamples_ref=True, togglenormals_ref=True,
    #                     togglesamples_refcls=True, togglenormals_refcls=True, alpha = .7)
    freegriptst.segShow(base, togglesamples=False, togglenormals=False,
                        togglesamples_ref=False, togglenormals_ref=False,
                        togglesamples_refcls=False, togglenormals_refcls=False, alpha=.2)
    mark = True
    while mark:
        mark = freegriptst.pairShow(base, togglecontacts=True, togglecontactnormals=True, togglerealcontacts=True)
    base.run()

    # objnp = pandageom.packpandanp(freegriptst.objtrimesh.vertices,
    #                               freegriptst.objtrimesh.face_normals, freegriptst.objtrimesh.faces)
    # objnp.setColor(.3,.3,.3,1)
    # objnp.reparentTo(base.render)

    # freegriptst.segShow2(base, togglesamples=False, togglenormals=False,
    #                     togglesamples_ref=False, togglenormals_ref=False,
    #                     togglesamples_refcls=True, togglenormals_refcls=False, specificface = True)
    #
    # def updateshow0(freegriptst, task):
    #     npc = base.render.findAllMatches("**/piece")
    #     for np in npc:
    #         np.removeNode()
    #     freegriptst.segShow2(base, togglesamples=True, togglenormals=False,
    #                         togglesamples_ref=False, togglenormals_ref=False,
    #                         togglesamples_refcls=False, togglenormals_refcls=False, specificface = True)
    #     freegriptst.segShow(base, togglesamples=False, togglenormals=False,
    #                         togglesamples_ref=False, togglenormals_ref=False,
    #                         togglesamples_refcls=False, togglenormals_refcls=False)
    #     return task.done
    #
    # def updateshow1(freegriptst, task):
    #     npc = base.render.findAllMatches("**/piece")
    #     for np in npc:
    #         np.removeNode()
    #     freegriptst.segShow2(base, togglesamples=True, togglenormals=False,
    #                         togglesamples_ref=True, togglenormals_ref=False,
    #                         togglesamples_refcls=False, togglenormals_refcls=False, specificface = True)
    #     freegriptst.segShow(base, togglesamples=False, togglenormals=False,
    #                         togglesamples_ref=False, togglenormals_ref=False,
    #                         togglesamples_refcls=False, togglenormals_refcls=False)
    #     return task.done
    #
    # def updateshow2(freegriptst, task):
    #     np = base.render.find("**/piece")
    #     if np:
    #         np.removeNode()
    #     freegriptst.segShow2(base, togglesamples=True, togglenormals=False,
    #                         togglesamples_ref=True, togglenormals_ref=False,
    #                         togglesamples_refcls=True, togglenormals_refcls=False, specificface = True)
    #     freegriptst.segShow(base, togglesamples=False, togglenormals=False,
    #                         togglesamples_ref=False, togglenormals_ref=False,
    #                         togglesamples_refcls=False, togglenormals_refcls=False)
    #     return task.done
    #
    # taskMgr.doMethodLater(10, updateshow0, "tickTask", extraArgs=[freegriptst], appendTask=True)
    # taskMgr.doMethodLater(20, updateshow1, "tickTask", extraArgs=[freegriptst], appendTask=True)
    # taskMgr.doMethodLater(30, updateshow2, "tickTask", extraArgs=[freegriptst], appendTask=True)
    # base.run()
    freegriptst.pairShow(base, togglecontacts=True, togglecontactnormals=True, togglerealcontacts=True)

    # def updateshow(task):
    #     freegriptst.pairShow(base, togglecontacts=True, togglecontactnormals=True, togglerealcontacts=True)
    #     # print(task.delayTime)
    #     # if abs(task.delayTime-13) < 1:
    #     #     task.delayTime -= 12.85
    #     return task.again
    # taskMgr.doMethodLater(.1, updateshow, "tickTask")
    # base.run()

    # geom = None
    # for i, faces in enumerate(freegriptst.objtrimesh.facets()):
    #     rgba = [np.random.random(),np.random.random(),np.random.random(),1]
    #     # geom = pandageom.packpandageom(freegriptst.objtrimesh.vertices, freegriptst.objtrimesh.face_normals[faces], freegriptst.objtrimesh.faces[faces])
    #     # compute facet normal
    #     facetnormal = np.sum(freegriptst.objtrimesh.face_normals[faces], axis=0)
    #     facetnormal = facetnormal/np.linalg.norm(facetnormal)
    #     geom = pandageom.packpandageom(freegriptst.objtrimesh.vertices +
    #                             np.tile(0 * facetnormal,
    #                                     [freegriptst.objtrimesh.vertices.shape[0], 1]),
    #                             freegriptst.objtrimesh.face_normals[faces],
    #                             freegriptst.objtrimesh.faces[faces])
    #     node = GeomNode('piece')
    #     node.addGeom(geom)
    #     star = NodePath('piece')
    #     star.attachNewNode(node)
    #     star.setColor(Vec4(rgba[0],rgba[1],rgba[2],rgba[3]))
    #     # star.setColor(Vec4(.7,.4,0,1))
    #     star.setTwoSided(True)
    #     star.reparentTo(base.render)
    # sampledpnts = samples[sample_idxes[i]]
    # for apnt in sampledpnts:
    #     pandageom.plotSphere(base, star, pos=apnt, radius=1, rgba=rgba)
    # for j, apnt in enumerate(freegriptst.samplepnts[i]):
    #     pandageom.plotSphere(base, star, pos=apnt, radius=0.7, rgba=rgba)
    #     pandageom.plotArrow(base, star, spos=apnt, epos=apnt+freegriptst.samplenrmls[i][j], rgba=[1,0,0,1], length=5, thickness=0.1)
    # # selectedfacet = 2
    # geom = ppg.packpandageom(mesh.vertices, mesh.face_normals[facets[selectedfacet]], mesh.faces[facets[selectedfacet]])
    # node = GeomNode('piece')
    # node.addGeom(geom)
    # star = NodePath('piece')
    # star.attachNewNode(node)
    # star.setColor(Vec4(1,0,0,1))
    # star.setTwoSided(True)
    # star.reparentTo(base.render)

    # for i, face in enumerate(mesh.faces[facets[selectedfacet]]):
    #     vert = (mesh.vertices[face[0],:]+mesh.vertices[face[1],:]+mesh.vertices[face[2],:])/3
    #     pandageom.plotArrow(base, star, spos=vert, epos=vert+mesh.face_normals[facets[selectedfacet][i],:], rgba=[1,0,0,1], length = 5, thickness = 0.1)

    # for i, vert in enumerate(mesh.vertices):
    #     pandageom.plotArrow(base, star, spos=vert, epos=vert+mesh.vertex_normals[i,:], rgba=[1,0,0,1], length = 5, thickness = 0.1)

    # generator = MeshDrawer()
    # generatorNode = generator.getRoot()
    # generatorNode.reparentTo(base.render)
    # generatorNode.setDepthWrite(False)
    # generatorNode.setTransparency(True)
    # generatorNode.setTwoSided(True)
    # generatorNode.setBin("fixed", 0)
    # generatorNode.setLightOff(True)
    #
    # generator.begin(base.cam, base.render)
    # generator.segment(Vec3(0,0,0), Vec3(10,0,0), Vec4(1,1,1,1), 0.5, Vec4(0,1,0,1))
    # generator.end()
    # mesh.show()

    # for face in facets:
    #     mesh.visual.face_colors[np.asarray(face)] = [trimesh.visual.random_color()]*mesh.visual.face_colors[face].shape[0]
    # mesh.show()
    # samples = sample.sample_surface_even(mesh, mesh.vertices.shape[0]*10)
    # ax3d.plot(ax1, samples[:,0], samples[:,1], samples[:,2], 'r.')
    # ax3dequal.set_axes_equal(ax1)
    #
    # ax2 = fig.add_subplot(122, projection='3d')
    # for face in facets:
    #     rndcolor = trimesh.visual.random_color()
    #     for faceid in face:
    #         triarray = mesh.vertices[mesh.faces[faceid]]
    #         tri = art3d.Poly3DCollection([triarray])
    #         tri.set_facecolor(mesh.visual.face_colors[faceid])
    #         ax2.add_collection3d(tri)

    # ax3dequal.set_axes_equal(ax2)
    # plt.show()
    #
    # from direct.showbase.ShowBase import ShowBase
    # from panda3d.core import *
    # import plot.pandactrl as pandactrl
    # import plot.pandageom as pandageom
    #
    # geom = ppg.packpandageom(mesh.vertices, mesh.face_normals, mesh.faces)
    # node = GeomNode('star')
    # node.addGeom(geom)
    # star = NodePath('star')
    # star.attachNewNode(node)
    # star.setColor(1,0,0)
    #
    #
    # base = ShowBase()
    #
    # # for i, face in enumerate(mesh.faces):
    # #     vert = (mesh.vertices[face[0],:]+mesh.vertices[face[1],:]+mesh.vertices[face[2],:])/3
    # #     pandageom.plotArrow(base, star, spos=vert, epos=vert+mesh.face_normals[i,:], rgba=[1,0,0,1], length = 5, thickness = 0.1)
    #
    # # for i, vert in enumerate(mesh.vertices):
    # #     pandageom.plotArrow(base, star, spos=vert, epos=vert+mesh.vertex_normals[i,:], rgba=[1,0,0,1], length = 5, thickness = 0.1)
    #
    # pandactrl.setRenderEffect(base)
    # pandactrl.setLight(base)
    # pandactrl.setCam(base, 0, 100, 100, 'perspective')
    #
    # star.reparentTo(base.render)
    #
    # generator = MeshDrawer()
    # generatorNode = generator.getRoot()
    # generatorNode.reparentTo(base.render)
    # generatorNode.setDepthWrite(False)
    # generatorNode.setTransparency(True)
    # generatorNode.setTwoSided(True)
    # generatorNode.setBin("fixed", 0)
    # generatorNode.setLightOff(True)
    #
    # generator.begin(base.cam, base.render)
    # generator.segment(Vec3(0,0,0), Vec3(10,0,0), Vec4(1,1,1,1), 0.5, Vec4(0,1,0,1))
    # generator.end()
    #
    # base.run()
