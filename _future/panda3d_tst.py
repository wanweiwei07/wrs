from direct.showbase.ShowBase import ShowBase
from panda3d.core import Geom, GeomNode, GeomVertexData, GeomVertexFormat, GeomVertexWriter, GeomTriangles, DirectionalLight, AmbientLight, NodePath

class SimpleViewer(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # 创建 Panda3D 渲染节点
        self.geom_node = GeomNode('simple_geom')
        self.node_path = NodePath(self.geom_node)
        self.node_path.reparent_to(self.render)

        # 设置摄像机位置
        self.cam.set_pos(0, -10, 10)
        self.cam.look_at(0, 0, 0)

        # 添加三点光源
        self.add_lighting()

        # 创建一个简单的立方体
        self.create_simple_cube()

    def add_lighting(self):
        # 主光源
        key_light = DirectionalLight('key_light')
        key_light.setColor((1, 1, 1, 1))
        key_light_np = self.render.attach_new_node(key_light)
        key_light_np.setHpr(-30, -30, 0)
        self.render.setLight(key_light_np)

        # 补光源
        fill_light = DirectionalLight('fill_light')
        fill_light.setColor((0.5, 0.5, 0.5, 1))
        fill_light_np = self.render.attach_new_node(fill_light)
        fill_light_np.setHpr(30, -30, 0)
        self.render.setLight(fill_light_np)

        # 背光源
        back_light = DirectionalLight('back_light')
        back_light.setColor((0.2, 0.2, 0.2, 1))
        back_light_np = self.render.attach_new_node(back_light)
        back_light_np.setHpr(0, 60, 0)
        self.render.setLight(back_light_np)

        # 环境光
        ambient_light = AmbientLight('ambient_light')
        ambient_light.setColor((0.2, 0.2, 0.2, 1))
        ambient_light_np = self.render.attach_new_node(ambient_light)
        self.render.setLight(ambient_light_np)

    def create_simple_cube(self):
        vertex_format = GeomVertexFormat.get_v3n3()
        vertex_data = GeomVertexData('cube', vertex_format, Geom.UH_static)

        vertex_writer = GeomVertexWriter(vertex_data, 'vertex')
        normal_writer = GeomVertexWriter(vertex_data, 'normal')

        # 立方体顶点
        vertices = [
            (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
            (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)
        ]

        # 立方体法线
        normals = [
            (0, 0, -1), (0, 0, 1), (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)
        ]

        # 立方体三角形索引
        indices = [
            (0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
            (0, 3, 7), (0, 7, 4), (1, 2, 6), (1, 6, 5)
        ]

        for v in vertices:
            vertex_writer.add_data3(*v)

        # 每个顶点对应一个法线
        for i in range(6):
            normal = normals[i]
            for _ in range(4):
                normal_writer.add_data3(*normal)

        triangles = GeomTriangles(Geom.UH_static)
        for tri in indices:
            triangles.add_vertices(*tri)

        geom = Geom(vertex_data)
        geom.add_primitive(triangles)
        self.geom_node.add_geom(geom)

# 创建并运行简单的立方体视图
viewer = SimpleViewer()
viewer.run()
