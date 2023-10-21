#!/usr/bin/env python3

import sys
import pickle
import numpy as np
from panda3d.core import *
from sklearn.cluster import KMeans
from panda3d.bullet import BulletWorld
from shapely.geometry import Point, Polygon
from sklearn.neighbors import RadiusNeighborsClassifier
import trimesh

import pyhiro.robotmath as rm
import pyhiro.pandageom as pg
import pyhiro.sample as sample
import pyhiro.collisiondetection as cd


class Freesuc(object):

    def __init__(self, ompath, handpkg, ser=False, torqueresist=50):
        self.objtrimesh = None
        # the sampled points and their normals
        self.objsamplepnts = None
        self.objsamplenrmls = None
        # the sampled points (bad samples removed)
        self.objsamplepnts_ref = None
        self.objsamplenrmls_ref = None
        # the sampled points (bad samples removed + clustered)
        self.objsamplepnts_refcls = None
        self.objsamplenrmls_refcls = None
        # facets is used to avoid repeated computation
        self.facets = None
        # facetnormals is used to plot overlapped facets with different heights
        self.facetnormals = None
        # facet2dbdries saves the 2d boundaries of each facet
        self.facet2dbdries = None
        # for plot
        self.facetcolorarray = None
        self.counter = 0
        # for collision detection
        self.bulletworld = BulletWorld()
        self.hand = handpkg.newHandNM(hndcolor=[.2, 0.7, .2, .3])

        # collision free grasps
        self.sucrotmats = []
        self.succontacts = []
        self.succontactnormals = []

        # collided grasps
        self.sucrotmatscld = []
        self.succontactscld = []
        self.succontactnormalscld = []

        self.objcenter = [0, 0, 0]
        self.torqueresist = torqueresist

        if ser is False:
            self.loadObjModel(ompath)
            self.saveSerialized("/tmp/tmpfsc.pickle")
        else:
            self.loadSerialized("/tmp/tmpfsc.pickle", ompath)

    def loadObjModel(self, ompath):
        self.objtrimesh = trimesh.load_mesh(ompath)
        # oversegmentation
        self.facets, self.facetnormals = \
            self.objtrimesh.facets_over(faceangle=.95)
        self.facetcolorarray = \
            pg.randomColorArray(self.facets.shape[0])
        self.sampleObjModel()
        # prepare the model for collision detection
        self.objgeom = pg.packpandageom(
            self.objtrimesh.vertices,
            self.objtrimesh.face_normals,
            self.objtrimesh.faces)
        self.objmeshbullnode = cd.genCollisionMeshGeom(self.objgeom)
        self.bulletworld.attachRigidBody(self.objmeshbullnode)
        # object center
        self.objcenter = [0, 0, 0]

        for pnt in self.objtrimesh.vertices:
            self.objcenter[0] += pnt[0]
            self.objcenter[1] += pnt[1]
            self.objcenter[2] += pnt[2]
        self.objcenter[0] = \
            self.objcenter[0] / self.objtrimesh.vertices.shape[0]
        self.objcenter[1] = \
            self.objcenter[1] / self.objtrimesh.vertices.shape[0]
        self.objcenter[2] = \
            self.objcenter[2] / self.objtrimesh.vertices.shape[0]

    def loadSerialized(self, filename, ompath):
        self.objtrimesh = trimesh.load_mesh(ompath)
        try:
            self.facets,
            self.facetnormals,
            self.facetcolorarray,
            self.objsamplepnts,
            self.objsamplenrmls,
            self.objsamplepnts_ref,
            self.objsamplenrmls_ref,
            self.objsamplepnts_refcls,
            self.objsamplenrmls_refcls = \
                pickle.load(open(filename, mode="rb"))
        except Exception as e:
            print(str(sys.exc_info()[0]) + " cannot load tmpcp.pickle")
            raise

    def saveSerialized(self, filename):
        pickle.dump(
            [self.facets,
             self.facetnormals,
             self.facetcolorarray,
             self.objsamplepnts,
             self.objsamplenrmls,
             self.objsamplepnts_ref,
             self.objsamplenrmls_ref,
             self.objsamplepnts_refcls,
             self.objsamplenrmls_refcls],
            open(filename, mode="wb"))

    def sampleObjModel(self, numpointsoververts=5):
        """
        sample the object model
        self.objsamplepnts and self.objsamplenrmls
        are filled in this function

        :param: numpointsoververts: the number of sampled points =
                                    numpointsoververts*mesh.vertices.shape[0]
        :return: nverts: the number of verts sampled

        author: weiwei
        date: 20160623 flight to tokyo
        """

        nverts = self.objtrimesh.vertices.shape[0]
        samples, face_idx = sample.sample_surface_even(
            self.objtrimesh,
            count=(1000 if nverts * numpointsoververts > 1000
                   else nverts * numpointsoververts))
        self.objsamplepnts = np.ndarray(
            shape=(self.facets.shape[0],), dtype=np.object)
        self.objsamplenrmls = np.ndarray(
            shape=(self.facets.shape[0],), dtype=np.object)
        for i, faces in enumerate(self.facets):
            for face in faces:
                sample_idx = np.where(face_idx == face)[0]
                if len(sample_idx) > 0:
                    if self.objsamplepnts[i] is not None:
                        self.objsamplepnts[i] = \
                            np.vstack((
                                self.objsamplepnts[i],
                                samples[sample_idx]))
                        self.objsamplenrmls[i] = \
                            np.vstack((self.objsamplenrmls[i],
                                       [self.objtrimesh.face_normals[face]] *
                                       samples[sample_idx].shape[0]))
                    else:
                        self.objsamplepnts[i] = np.array(samples[sample_idx])
                        self.objsamplenrmls[i] = np.array(
                            [self.objtrimesh.face_normals[face]] *
                            samples[sample_idx].shape[0])
            if self.objsamplepnts[i] is None:
                self.objsamplepnts[i] = np.empty(shape=[0, 0])
                self.objsamplenrmls[i] = np.empty(shape=[0, 0])
        return nverts

    def removeBadSamples(self, mindist=7, maxdist=9999):
        '''
        Do the following refinement:
        (1) remove the samples who's minimum distance to facet boundary
            is smaller than mindist
        (2) remove the samples who's maximum distance to facet boundary
            is larger than mindist

        ## input
        mindist, maxdist
            as explained in the begining

        author: weiwei
        date: 20160623 flight to tokyo
        '''

        # ref = refine
        self.objsamplepnts_ref = np.ndarray(
            shape=(self.facets.shape[0],),
            dtype=np.object)
        self.objsamplenrmls_ref = np.ndarray(
            shape=(self.facets.shape[0],),
            dtype=np.object)
        self.facet2dbdries = []
        for i, faces in enumerate(self.facets):
            facetp = None
            face0verts = self.objtrimesh.vertices[
                self.objtrimesh.faces[faces[0]]]
            facetmat = rm.rotmatfacet(
                self.facetnormals[i],
                face0verts[0],
                face0verts[1])
            # face samples
            samplepntsp = []
            for j, apnt in enumerate(self.objsamplepnts[i]):
                apntp = np.dot(facetmat, apnt)[:2]
                samplepntsp.append(apntp)
            # face boundaries
            for j, faceidx in enumerate(faces):
                vert0 = self.objtrimesh.vertices[
                    self.objtrimesh.faces[faceidx][0]]
                vert1 = self.objtrimesh.vertices[
                    self.objtrimesh.faces[faceidx][1]]
                vert2 = self.objtrimesh.vertices[
                    self.objtrimesh.faces[faceidx][2]]
                vert0p = np.dot(facetmat, vert0)[:2]
                vert1p = np.dot(facetmat, vert1)[:2]
                vert2p = np.dot(facetmat, vert2)[:2]
                facep = Polygon([vert0p, vert1p, vert2p])
                if facetp is None:
                    facetp = facep
                else:
                    try:
                        facetp = facetp.union(facep)
                    except Exception as e:
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
                except Exception as e:
                    pass
            self.objsamplepnts_ref[i] = np.asarray(
                [self.objsamplepnts[i][j] for j in selectedele])
            self.objsamplenrmls_ref[i] = np.asarray(
                [self.objsamplenrmls[i][j] for j in selectedele])
        self.facet2dbdries = np.array(self.facet2dbdries)

    def clusterFacetSamplesKNN(self, reduceRatio=3, maxNPnts=5):
        """
        cluster the samples of each facet using k nearest neighbors
        the cluster center and their correspondent normals will be saved
        in self.objsamplepnts_refcls and self.objsamplenrmals_refcls

        :param: reduceRatio: the ratio of points to reduce
        :param: maxNPnts: the maximum number of points on a facet
        :return: None

        author: weiwei
        date: 20161129, tsukuba
        """

        self.objsamplepnts_refcls = np.ndarray(
            shape=(self.facets.shape[0],),
            dtype=np.object)
        self.objsamplenrmls_refcls = np.ndarray(
            shape=(self.facets.shape[0],),
            dtype=np.object)
        for i, facet in enumerate(self.facets):
            self.objsamplepnts_refcls[i] = np.empty(shape=(0, 0))
            self.objsamplenrmls_refcls[i] = np.empty(shape=(0, 0))
            X = self.objsamplepnts_ref[i]
            nX = X.shape[0]
            if nX > reduceRatio:
                kmeans = KMeans(
                    n_clusters=maxNPnts
                    if nX / reduceRatio > maxNPnts else nX / reduceRatio,
                    random_state=0).fit(X)
                self.objsamplepnts_refcls[i] = kmeans.cluster_centers_
                self.objsamplenrmls_refcls[i] = np.tile(
                    self.facetnormals[i],
                    [self.objsamplepnts_refcls.shape[0], 1])

    def clusterFacetSamplesRNN(self, reduceRadius=3):
        """
        cluster the samples of each facet using radius nearest neighbours
        the cluster center and their correspondent normals will be saved
        in self.objsamplepnts_refcls and self.objsamplenrmals_refcls

        :param: reduceRadius: the neighbors that fall inside
                              the reduceradius will be removed
        :return: None

        author: weiwei
        date: 20161130, osaka
        """

        self.objsamplepnts_refcls = np.ndarray(
            shape=(self.facets.shape[0],),
            dtype=np.object)
        self.objsamplenrmls_refcls = np.ndarray(
            shape=(self.facets.shape[0],),
            dtype=np.object)
        for i, facet in enumerate(self.facets):
            self.objsamplepnts_refcls[i] = []
            self.objsamplenrmls_refcls[i] = []
            X = self.objsamplepnts_ref[i]
            nX = X.shape[0]
            if nX > 0:
                neigh = RadiusNeighborsClassifier(radius=1.0)
                neigh.fit(X, range(nX))
                neigharrays = neigh.radius_neighbors(
                    X, radius=reduceRadius, return_distance=False)
                delset = set([])
                for j in range(nX):
                    if j not in delset:
                        self.objsamplepnts_refcls[i].append(
                            np.array(X[j]))
                        self.objsamplenrmls_refcls[i].append(
                            np.array(self.objsamplenrmls_ref[i][j]))
                        delset.update(neigharrays[j].tolist())
            if self.objsamplepnts_refcls[i]:
                self.objsamplepnts_refcls[i] = np.vstack(
                    self.objsamplepnts_refcls[i])
                self.objsamplenrmls_refcls[i] = np.vstack(
                    self.objsamplenrmls_refcls[i])
            else:
                self.objsamplepnts_refcls[i] = np.empty(shape=(0, 0))
                self.objsamplenrmls_refcls[i] = np.empty(shape=(0, 0))

    def removeHndcc(self, base, discretesize=8):
        """
        Handcc means hand collision detection

        :param discretesize: the number of hand orientations
        :return:

        author: weiwei
        date: 20161212, tsukuba
        """

        self.sucrotmats = []
        self.succontacts = []
        self.succontactnormals = []
        self.sucrotmatscld = []
        self.succontactscld = []
        self.succontactnormalscld = []

        plotoffsetfp = 3
        self.counter = 0

        while self.counter < self.facets.shape[0]:
            for i in range(self.objsamplepnts_refcls[self.counter].shape[0]):
                for angleid in range(discretesize):
                    cctpnt = self.objsamplepnts_refcls[self.counter][i] + \
                             plotoffsetfp * \
                             self.objsamplenrmls_refcls[self.counter][i]
                    # check torque resistance
                    if Vec3(cctpnt[0], cctpnt[1], cctpnt[2]).length() < \
                       self.torqueresist:
                        cctnrml = self.objsamplenrmls_refcls[self.counter][i]
                        rotangle = 360.0 / discretesize * angleid
                        tmphand = self.hand
                        tmphand.attachTo(
                            cctpnt[0], cctpnt[1], cctpnt[2],
                            cctnrml[0], cctnrml[1], cctnrml[2],
                            rotangle)
                        hndbullnode = cd.genCollisionMeshMultiNp(
                            base, tmphand.handnp, base.render)
                        result = self.bulletworld.contactTest(hndbullnode)

                        if not result.getNumContacts():
                            self.succontacts.append(
                                self.objsamplepnts_refcls[self.counter][i])
                            self.succontactnormals.append(
                                self.objsamplenrmls_refcls[self.counter][i])
                            self.sucrotmats.append(tmphand.getMat())
                        else:
                            self.succontactscld.append(
                                self.objsamplepnts_refcls[self.counter][i])
                            self.succontactnormalscld.append(
                                self.objsamplenrmls_refcls[self.counter][i])
                            self.sucrotmatscld.append(tmphand.getMat())
            self.counter += 1
        self.counter = 0

    def segShow(
            self,
            base,
            togglesamples=False,
            togglenormals=False,
            togglesamples_ref=False,
            togglenormals_ref=False,
            togglesamples_refcls=False,
            togglenormals_refcls=False,
            alpha=.1):
        """
        :param base:
        :param togglesamples:
        :param togglenormals:
        :param togglesamples_ref: toggles the sampled points that
                                  fulfills the dist requirements
        :param togglenormals_ref:
        :return:
        """

        nfacets = self.facets.shape[0]
        facetcolorarray = self.facetcolorarray

        # plot the segments
        plotoffsetf = .0
        for i, facet in enumerate(self.facets):
            geom = pg.packpandageom(
                self.objtrimesh.vertices +
                np.tile(plotoffsetf*i*self.facetnormals[i],
                        [self.objtrimesh.vertices.shape[0], 1]),
                self.objtrimesh.face_normals[facet],
                self.objtrimesh.faces[facet])
            node = GeomNode('piece')
            node.addGeom(geom)
            star = NodePath('piece')
            star.attachNewNode(node)
            star.setColor(Vec4(
                facetcolorarray[i][0],
                facetcolorarray[i][1],
                facetcolorarray[i][2],
                alpha))
            star.setTransparency(TransparencyAttrib.MAlpha)
            star.setTwoSided(True)
            star.reparentTo(base.render)
            rgbapnts0 = [1, 1, 1, 1]
            rgbapnts1 = [.5, .5, 0, 1]
            rgbapnts2 = [1, 0, 0, 1]
            if togglesamples:
                for j, apnt in enumerate(self.objsamplepnts[i]):
                    pg.plotSphere(
                        star,
                        pos=apnt + plotoffsetf * i *
                        self.facetnormals[i],
                        radius=3, rgba=rgbapnts0)
            if togglenormals:
                for j, apnt in enumerate(self.objsamplepnts[i]):
                    pg.plotArrow(
                        star,
                        spos=apnt + plotoffsetf * i *
                        self.facetnormals[i],
                        epos=apnt + plotoffsetf * i *
                        self.facetnormals[i] + self.objsamplenrmls[i][j],
                        rgba=rgbapnts0, length=10)
            if togglesamples_ref:
                for j, apnt in enumerate(self.objsamplepnts_ref[i]):
                    pg.plotSphere(
                        star,
                        pos=apnt + plotoffsetf * i *
                        self.facetnormals[i],
                        radius=3, rgba=rgbapnts1)
            if togglenormals_ref:
                for j, apnt in enumerate(self.objsamplepnts_ref[i]):
                    pg.plotArrow(
                        star,
                        spos=apnt + plotoffsetf * i *
                        self.facetnormals[i],
                        epos=apnt + plotoffsetf * i *
                        self.facetnormals[i] +
                        self.objsamplenrmls_ref[i][j],
                        rgba=rgbapnts1, length=10)
            if togglesamples_refcls:
                for j, apnt in enumerate(self.objsamplepnts_refcls[i]):
                    pg.plotSphere(
                        star,
                        pos=apnt + plotoffsetf * i *
                        self.facetnormals[i],
                        radius=3, rgba=rgbapnts2)
            if togglenormals_refcls:
                for j, apnt in enumerate(self.objsamplepnts_refcls[i]):
                    pg.plotArrow(
                        star,
                        spos=apnt + plotoffsetf * i *
                        self.facetnormals[i],
                        epos=apnt + plotoffsetf * i *
                        self.facetnormals[i] +
                        self.objsamplenrmls_refcls[i][j],
                        rgba=rgbapnts2, length=10)

    def segShow2(
            self,
            base,
            togglesamples=False,
            togglenormals=False,
            togglesamples_ref=False,
            togglenormals_ref=False,
            togglesamples_refcls=False,
            togglenormals_refcls=False,
            specificface=True):
        """
        :param base:
        :param togglesamples:
        :param togglenormals:
        :param togglesamples_ref: toggles the sampled points that
                                  fulfills the dist requirements
        :param togglenormals_ref:
        :return:
        """

        nfacets = self.facets.shape[0]
        facetcolorarray = self.facetcolorarray
        rgbapnts0 = [1, 1, 1, 1]
        rgbapnts1 = [0.2, 0.7, 1, 1]
        rgbapnts2 = [1, 0.7, 0.2, 1]

        plotoffsetf = .0
        faceplotted = False
        # plot the segments
        for i, facet in enumerate(self.facets):
            if not specificface:
                geom = pg.packpandageom(
                    self.objtrimesh.vertices+np.tile(
                        plotoffsetf*i*self.facetnormals[i],
                        [self.objtrimesh.vertices.shape[0], 1]),
                    self.objtrimesh.face_normals[facet],
                    self.objtrimesh.faces[facet])
                node = GeomNode('piece')
                node.addGeom(geom)
                star = NodePath('piece')
                star.attachNewNode(node)
                star.setColor(Vec4(.77, .17, 0, 1))
                star.setTransparency(TransparencyAttrib.MAlpha)

                star.setTwoSided(True)
                star.reparentTo(base.render)
                if togglesamples:
                    for j, apnt in enumerate(self.objsamplepnts[i]):
                        pg.plotSphere(
                            star,
                            pos=apnt + plotoffsetf * i *
                            self.facetnormals[i],
                            radius=2.8, rgba=rgbapnts0)
                if togglenormals:
                    for j, apnt in enumerate(self.objsamplepnts[i]):
                        pg.plotArrow(
                            star,
                            spos=apnt + plotoffsetf * i *
                            self.facetnormals[i],
                            epos=apnt + plotoffsetf * i *
                            self.facetnormals[i] +
                            self.objsamplenrmls[i][j],
                            rgba=rgbapnts0, length=10)
                if togglesamples_ref:
                    for j, apnt in enumerate(self.objsamplepnts_ref[i]):
                        pg.plotSphere(
                            star,
                            pos=apnt + plotoffsetf * i *
                            self.facetnormals[i],
                            radius=2.9, rgba=rgbapnts1)
                if togglenormals_ref:
                    for j, apnt in enumerate(self.objsamplepnts_ref[i]):
                        pg.plotArrow(
                            star,
                            spos=apnt + plotoffsetf * i *
                            self.facetnormals[i],
                            epos=apnt + plotoffsetf * i *
                            self.facetnormals[i] +
                            self.objsamplenrmls_ref[i][j],
                            rgba=rgbapnts1, length=10)
                if togglesamples_refcls:
                    for j, apnt in enumerate(self.objsamplepnts_refcls[i]):
                        pg.plotSphere(
                            star,
                            pos=apnt + plotoffsetf * i *
                            self.facetnormals[i],
                            radius=3, rgba=rgbapnts2)
                if togglenormals_refcls:
                    for j, apnt in enumerate(self.objsamplepnts_refcls[i]):
                        pg.plotArrow(
                            star,
                            spos=apnt + plotoffsetf * i *
                            self.facetnormals[i],
                            epos=apnt + plotoffsetf * i *
                            self.facetnormals[i] +
                            self.objsamplenrmls_refcls[i][j],
                            rgba=rgbapnts2, length=10)
            if specificface:
                plotoffsetf = .1
                if faceplotted:
                    continue
                else:
                    if len(self.objsamplepnts[i]) > 25:
                        faceplotted = True
                        geom = pg.packpandageom(
                            self.objtrimesh.vertices +
                            np.tile(plotoffsetf * i * self.facetnormals[i],
                                    [self.objtrimesh.vertices.shape[0], 1]),
                            self.objtrimesh.face_normals[facet],
                            self.objtrimesh.faces[facet])
                        node = GeomNode('piece')
                        node.addGeom(geom)
                        star = NodePath('piece')
                        star.attachNewNode(node)
                        star.setColor(Vec4(
                            facetcolorarray[i][0],
                            facetcolorarray[i][1],
                            facetcolorarray[i][2], 1))
                        star.setTransparency(TransparencyAttrib.MAlpha)

                        star.setTwoSided(True)
                        star.reparentTo(base.render)
                        if togglesamples:
                            for j, apnt in enumerate(
                                    self.objsamplepnts[i]):
                                pg.plotSphere(
                                    star,
                                    pos=apnt + plotoffsetf * i *
                                    self.facetnormals[i],
                                    radius=2.8, rgba=rgbapnts0)
                        if togglenormals:
                            for j, apnt in enumerate(
                                    self.objsamplepnts[i]):
                                pg.plotArrow(
                                    star,
                                    spos=apnt + plotoffsetf * i *
                                    self.facetnormals[i],
                                    epos=apnt + plotoffsetf * i *
                                    self.facetnormals[i] +
                                    self.objsamplenrmls[i][j],
                                    rgba=rgbapnts0, length=10)
                        if togglesamples_ref:
                            for j, apnt in enumerate(
                                    self.objsamplepnts_ref[i]):
                                pg.plotSphere(
                                    star,
                                    pos=apnt + plotoffsetf * i *
                                    self.facetnormals[i],
                                    radius=3, rgba=rgbapnts1)
                        if togglenormals_ref:
                            for j, apnt in enumerate(
                                    self.objsamplepnts_ref[i]):
                                pg.plotArrow(
                                    star,
                                    spos=apnt + plotoffsetf * i *
                                    self.facetnormals[i],
                                    epos=apnt + plotoffsetf * i *
                                    self.facetnormals[i] +
                                    self.objsamplenrmls_ref[i][j],
                                    rgba=rgbapnts1, length=10)
                        if togglesamples_refcls:
                            for j, apnt in enumerate(
                                    self.objsamplepnts_refcls[i]):
                                pg.plotSphere(
                                    star,
                                    pos=apnt + plotoffsetf * i *
                                    self.facetnormals[i],
                                    radius=3.5, rgba=rgbapnts2)
                        if togglenormals_refcls:
                            for j, apnt in enumerate(
                                    self.objsamplepnts_refcls[i]):
                                pg.plotArrow(
                                    star,
                                    spos=apnt + plotoffsetf * i *
                                    self.facetnormals[i],
                                    epos=apnt + plotoffsetf * i *
                                    self.facetnormals[i] +
                                    self.objsamplenrmls_refcls[i][j],
                                    rgba=rgbapnts2, length=10)
