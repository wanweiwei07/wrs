from direct.filter import FilterManager as pfm
from panda3d.core import Texture, CardMaker, NodePath, AuxBitplaneAttrib, LightRampAttrib, Camera, OrthographicLens, \
    GraphicsOutput, WindowProperties, FrameBufferProperties, GraphicsPipe


class FilterManager(pfm.FilterManager):

    def __init__(self, win, cam):
        super().__init__(win, cam)

    def renderSceneInto(self, depthtex=None, colortex=None, auxtex=None, auxbits=0, textures=None, fbprops=None,
                        clamping=None):
        """
        overload direct.filters.FilterManager.renderSceneInto
        :param depthtex:
        :param colortex:
        :param auxtex:
        :param auxbits:
        :param textures:
        :param fbprops:
        :param clamping:
        :return:
        """
        if (textures):
            colortex = textures.get("color", None)
            depthtex = textures.get("depth", None)
            auxtex = textures.get("aux", None)
            auxtex0 = textures.get("aux0", auxtex)
            auxtex1 = textures.get("aux1", None)
        else:
            auxtex0 = auxtex
            auxtex1 = None
        if (colortex == None):
            colortex = Texture("filter-base-color")
            colortex.setWrapU(Texture.WMClamp)
            colortex.setWrapV(Texture.WMClamp)
        texgroup = (depthtex, colortex, auxtex0, auxtex1)
        # Choose the size of the offscreen buffer.
        (winx, winy) = self.getScaledSize(1, 1, 1)
        if fbprops is not None:
            buffer = self.createBuffer("filter-base", winx, winy, texgroup, fbprops=fbprops)
        else:
            buffer = self.createBuffer("filter-base", winx, winy, texgroup)
        if (buffer == None):
            return None
        cm = CardMaker("filter-base-quad")
        cm.setFrameFullscreenQuad()
        quad = NodePath(cm.generate())
        quad.setDepthTest(0)
        quad.setDepthWrite(0)
        quad.setTexture(colortex)
        quad.setColor(1, 0.5, 0.5, 1)
        cs = NodePath("dummy")
        cs.setState(self.camstate)
        # Do we really need to turn on the Shader Generator?
        # cs.setShaderAuto()
        if (auxbits):
            cs.setAttrib(AuxBitplaneAttrib.make(auxbits))
        if clamping is False:
            # Disables clamping in the shader generator.
            cs.setAttrib(LightRampAttrib.make_identity())
        self.camera.node().setInitialState(cs.getState())
        quadcamnode = Camera("filter-quad-cam")
        lens = OrthographicLens()
        lens.setFilmSize(2, 2)
        lens.setFilmOffset(0, 0)
        lens.setNearFar(-1000, 1000)
        quadcamnode.setLens(lens)
        quadcam = quad.attachNewNode(quadcamnode)
        self.region.setCamera(quadcam)
        self.setStackedClears(buffer, self.rclears, self.wclears)
        if (auxtex0):
            buffer.setClearActive(GraphicsOutput.RTPAuxRgba0, 1)
            buffer.setClearValue(GraphicsOutput.RTPAuxRgba0, (0.5, 0.5, 1.0, 0.0))
        if (auxtex1):
            buffer.setClearActive(GraphicsOutput.RTPAuxRgba1, 1)
        self.region.disableClears()
        if (self.isFullscreen()):
            self.win.disableClears()
        dr = buffer.makeDisplayRegion()
        dr.disableClears()
        dr.setCamera(self.camera)
        dr.setActive(1)
        self.buffers.append(buffer)
        self.sizes.append((1, 1, 1))
        return quad

    def createBuffer(self, name, xsize, ysize, texgroup, depthbits=1, fbprops=None):
        """
        overload direct.filters.FilterManager.createBuffer
        :param name:
        :param xsize:
        :param ysize:
        :param texgroup:
        :param depthbits:
        :param fbprops:
        :return:
        """
        winprops = WindowProperties()
        winprops.setSize(xsize, ysize)
        props = FrameBufferProperties(FrameBufferProperties.getDefault())
        props.setBackBuffers(0)
        props.setRgbColor(1)
        props.setDepthBits(depthbits)
        props.setStereo(self.win.isStereo())
        if fbprops is not None:
            props.addProperties(fbprops)
        depthtex, colortex, auxtex0, auxtex1 = texgroup
        if (auxtex0 != None):
            props.setAuxRgba(1)
        if (auxtex1 != None):
            props.setAuxRgba(2)
        buffer=base.graphicsEngine.makeOutput(
            self.win.getPipe(), name, -1,
            props, winprops, GraphicsPipe.BFRefuseWindow | GraphicsPipe.BFResizeable,
            self.win.getGsg(), self.win)
        if (buffer == None):
            return buffer
        if (colortex):
            buffer.addRenderTexture(colortex, GraphicsOutput.RTMBindOrCopy, GraphicsOutput.RTPColor)
        if (depthtex):
            buffer.addRenderTexture(depthtex, GraphicsOutput.RTMBindOrCopy, GraphicsOutput.RTPDepth)
        if (auxtex0):
            buffer.addRenderTexture(auxtex0, GraphicsOutput.RTMBindOrCopy, GraphicsOutput.RTPAuxRgba0)
        if (auxtex1):
            buffer.addRenderTexture(auxtex1, GraphicsOutput.RTMBindOrCopy, GraphicsOutput.RTPAuxRgba1)
        buffer.setSort(self.nextsort)
        buffer.disableClears()
        self.nextsort += 1
        return buffer