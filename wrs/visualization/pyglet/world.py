import pyglet
import pyglet.gl as gl
import wrs.basis.robot_math as rm
import wrs.visualization.pyglet.rendering as rendering

class World(pyglet.window.Window):
    def __init__(self,
                 cam_pos=rm.np.array([2.0, 0.5, 2.0]),
                 lookat_pos=rm.np.zeros(3),
                 up=rm.np.array([0, 0, 1]),
                 fov=40,
                 w=1920,
                 h=1080,
                 auto_rotate=False):
        super().__init__(w, h, "Pyglet2 Cartoon Demo", resizable=True)
        self.background = None
        self.init_gl()

    def init_gl(self):
        # if user passed a background color use it
        if self.background is None:
            # default background color is white
            background = rm.np.ones(4)
        self._gl_set_background(background)
        # use camera setting for depth
        self._gl_enable_depth()
        # self._gl_enable_color_material()
        # self._gl_enable_blending()
        # self._gl_enable_smooth_lines(**self.line_settings)
        # self._gl_enable_lighting(self.scene)

    @staticmethod
    def _gl_set_background(background):
        gl.glClearColor(*background)

    @staticmethod
    def _gl_unset_background():
        gl.glClearColor(*[0, 0, 0, 0])

    @staticmethod
    def _gl_enable_depth():
        """
        Enable depth test in OpenGL using distances
        from `scene.camera`.
        """
        gl.glClearDepth(1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)

    @staticmethod
    def _gl_enable_color_material():
        # do some openGL things
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glShadeModel(gl.GL_SMOOTH)

        gl.glMaterialfv(
            gl.GL_FRONT,
            gl.GL_AMBIENT,
            rendering.vector_to_gl(0.192250, 0.192250, 0.192250),
        )
        gl.glMaterialfv(
            gl.GL_FRONT,
            gl.GL_DIFFUSE,
            rendering.vector_to_gl(0.507540, 0.507540, 0.507540),
        )
        gl.glMaterialfv(
            gl.GL_FRONT,
            gl.GL_SPECULAR,
            rendering.vector_to_gl(0.5082730, 0.5082730, 0.5082730),
        )

        gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, 0.4 * 128.0)

    @staticmethod
    def _gl_enable_blending():
        # enable blending for transparency
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    @staticmethod
    def _gl_enable_smooth_lines(line_width=4, point_size=4):
        # make the lines from Path3D objects less ugly
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        # set the width of lines to 4 pixels
        gl.glLineWidth(line_width)
        # set PointCloud markers to 4 pixels in size
        gl.glPointSize(point_size)

    @staticmethod
    def _gl_enable_lighting(scene):
        """
        Take the lights defined in scene.lights and
        apply them as openGL lights.
        """
        gl.glEnable(gl.GL_LIGHTING)
        # opengl only supports 7 lights?
        for i, light in enumerate(scene.lights[:7]):
            # the index of which light we have
            lightN = eval(f"gl.GL_LIGHT{i}")

            # get the transform for the light by name
            matrix = scene.graph.get(light.name)[0]

            # convert light object to glLightfv calls
            multiargs = rendering.light_to_gl(
                light=light, transform=matrix, lightN=lightN
            )

            # enable the light in question
            gl.glEnable(lightN)
            # run the glLightfv calls
            for args in multiargs:
                gl.glLightfv(*args)


if __name__ == '__main__':
    world = World()
    pyglet.app.run()
