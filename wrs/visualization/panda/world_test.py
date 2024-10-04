from direct.showbase.ShowBase import ShowBase
from panda3d.core import Trackball, Shader, Vec3, AmbientLight, DirectionalLight, Vec4

class CartoonApp(ShowBase):
    def __init__(self):
        super().__init__()
        # Load a model.
        self.panda = self.loader.loadModel("models/panda")
        self.panda.setScale(0.005, 0.005, 0.005)  # Adjust the model's scale.
        self.panda.reparentTo(self.render)

        self.disableMouse()
        # Create a trackball control for the camera.
        self.trackball = Trackball('trackball')
        self.trackballNodePath = self.render.attachNewNode(self.trackball)
        self.trackballNodePath.setPos(0, 100, 0)
        self.mouseWatcherNode.setPythonTag("trackball", self.trackball)
        self.cam.reparentTo(self.trackballNodePath)
        # Set initial camera and trackball position.
        self.taskMgr.add(self.updateTrackball, "updateTrackballTask")
        # Set up mouse wheel event listeners for zooming.
        self.accept('wheel_up', self.zoom, [1])
        self.accept('wheel_down', self.zoom, [-1])


        # Define the vertex shader.
        vertex_shader = """
        #version 330 core
        layout(location = 0) in vec4 p3d_Vertex;
        layout(location = 2) in vec3 p3d_Normal;
        uniform mat4 p3d_ModelViewProjectionMatrix;
        uniform mat3 p3d_NormalMatrix;
        uniform vec3 lightDirection;
        out float intensity;
        void main() {
            gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
            vec3 normal = normalize(p3d_NormalMatrix * p3d_Normal);
            intensity = max(dot(normal, lightDirection), 0.0);
        }
        """

        # Define the fragment shader.
        fragment_shader = """
        #version 330 core
        in float intensity;
        out vec4 FragColor;
        void main() {
            if (intensity > 0.5) FragColor = vec4(1, 1, 1, 1);
            else if (intensity > 0.2) FragColor = vec4(0.6, 0.6, 0.6, 1);
            else FragColor = vec4(0.3, 0.3, 0.3, 1);
        }
        """

        # Compile and apply the shader.
        shader = Shader.make(Shader.SL_GLSL, vertex=vertex_shader, fragment=fragment_shader)
        self.panda.setShader(shader)

        # Manually set the light direction.
        self.panda.setShaderInput("lightDirection", Vec3(-0.5, -0.5, -0.5).normalized())

        # Set up ambient light.
        ambientLight = AmbientLight("ambientLight")
        ambientLight.setColor(Vec4(0.2, 0.2, 0.2, 1))
        ambientLightNP = self.render.attachNewNode(ambientLight)
        self.render.setLight(ambientLightNP)

        # Set up directional light.
        directionalLight = DirectionalLight("directionalLight")
        directionalLight.setColor(Vec4(0.8, 0.8, 0.8, 1))
        directionalLightNP = self.render.attachNewNode(directionalLight)
        directionalLightNP.setHpr(-30, -30, 0)
        self.render.setLight(directionalLightNP)

    def updateTrackball(self, task):
        if self.mouseWatcherNode.hasMouse():
            # Example of processing mouse inputs -- adjust as needed.
            x, y = self.mouseWatcherNode.getMouseX(), self.mouseWatcherNode.getMouseY()
            # Implement custom trackball control logic here.
            # For simplicity, this example does not implement specific control logic.
            # You can manipulate self.trackball here based on mouse inputs.
        return task.cont


    def zoom(self, direction):
        # Adjust the zoom speed and limits as necessary
        zoomSpeed = 10
        newPos = self.trackballNodePath.getY() + direction * zoomSpeed
        self.trackballNodePath.setY(newPos)

if __name__ == "__main__":
    app = CartoonApp()
    app.run()
