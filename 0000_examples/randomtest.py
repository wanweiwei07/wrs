import copy
from panda3d.core import NodePath
from direct.showbase.ShowBase import ShowBase

class MyApp(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)

        # Load the environment model.
        self.scene = self.loader.loadModel("models/panda")
        # Reparent the model to render.
        self.scene.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        self.scene.setScale(0.25, 0.25, 0.25)
        self.scene.setPos(-8, 41, 0)


if __name__ == '__main__':
    app = MyApp()
    print(app.scene.getScale())
    scene2 = copy.deepcopy(app.scene)
    print(scene2.getScale())
    app.run()