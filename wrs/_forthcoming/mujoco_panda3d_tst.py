import mujoco
import numpy as np
import os
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Geom, GeomNode, GeomVertexData, GeomVertexFormat, GeomVertexWriter, GeomTriangles, \
    DirectionalLight, AmbientLight, NodePath, LVector3, Material, CullFaceAttrib


class MuJoCoSimulator(ShowBase):
    def __init__(self, model_path):
        ShowBase.__init__(self)

        # 设置 MuJoCo 模型
        print(f"Loading model from {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        print("Model loaded successfully")

        # Panda3D 初始化
        self.taskMgr.add(self.update_simulation, "update_simulation")

        # 创建 Panda3D 渲染节点
        self.geom_node = GeomNode('mujoco_geom')
        self.node_path = NodePath(self.geom_node)
        self.node_path.reparent_to(self.render)

        # 设置摄像机位置
        self.cam.set_pos(5, 5,5)  # 调整摄像机位置更靠近几何体
        self.cam.look_at(0, 0, 1)
        print(f"Camera position: {self.cam.get_pos()}, looking at: {self.cam.get_hpr()}")

        # 添加三点光源
        self.add_lighting()

    def add_lighting(self):
        # 主光源
        key_light = DirectionalLight('key_light')
        key_light.setColor((1, 1, 1, 1))
        key_light_np = self.render.attach_new_node(key_light)
        key_light_np.setHpr(-30, -30, 0)
        self.render.setLight(key_light_np)

        # 补光源
        fill_light = DirectionalLight('fill_light')
        fill_light.setColor((0.7, 0.7, 0.7, 1))
        fill_light_np = self.render.attach_new_node(fill_light)
        fill_light_np.setHpr(30, -10, 0)
        self.render.setLight(fill_light_np)

        # 背光源
        back_light = DirectionalLight('back_light')
        back_light.setColor((0.4, 0.4, 0.4, 1))
        back_light_np = self.render.attach_new_node(back_light)
        back_light_np.setHpr(0, 60, 0)
        self.render.setLight(back_light_np)

        # 环境光
        ambient_light = AmbientLight('ambient_light')
        ambient_light.setColor((0.3, 0.3, 0.3, 1))
        ambient_light_np = self.render.attach_new_node(ambient_light)
        self.render.setLight(ambient_light_np)

    def create_geom_from_mujoco(self):
        # 清除现有几何
        self.geom_node.remove_all_geoms()
        print("Creating geometry from MuJoCo data")

        # 从 MuJoCo 数据创建几何体
        vertex_format = GeomVertexFormat.get_v3n3()
        vertex_data = GeomVertexData('mujoco', vertex_format, Geom.UH_static)

        vertex_writer = GeomVertexWriter(vertex_data, 'vertex')
        normal_writer = GeomVertexWriter(vertex_data, 'normal')

        triangles = GeomTriangles(Geom.UH_static)

        for body_id in range(self.model.nbody):
            pos = self.data.xpos[body_id]
            print(f"Body {body_id} position: {pos}")
            # 创建一个简单的几何，例如一个立方体或球体
            size = .1  # 确保几何体大小适当
            vertices = [
                (pos[0] - size, pos[1] - size, pos[2] - size),
                (pos[0] + size, pos[1] - size, pos[2] - size),
                (pos[0] + size, pos[1] + size, pos[2] - size),
                (pos[0] - size, pos[1] + size, pos[2] - size),
                (pos[0] - size, pos[1] - size, pos[2] + size),
                (pos[0] + size, pos[1] - size, pos[2] + size),
                (pos[0] + size, pos[1] + size, pos[2] + size),
                (pos[0] - size, pos[1] + size, pos[2] + size)
            ]

            # 添加调试信息，打印顶点位置
            for v in vertices:
                print(f"Vertex: {v}")

            normals = [
                LVector3(0, 0, -1), LVector3(0, 0, -1), LVector3(0, 0, -1), LVector3(0, 0, -1),
                LVector3(0, 0, 1), LVector3(0, 0, 1), LVector3(0, 0, 1), LVector3(0, 0, 1)
            ]

            for v, n in zip(vertices, normals):
                vertex_writer.add_data3(*v)
                normal_writer.add_data3(n)
                print(f"Vertex: {v}, Normal: {n}")  # 打印顶点和法线位置

            # 添加三角形
            indices = [
                (0, 1, 2), (2, 3, 0),  # Front face
                (4, 5, 6), (6, 7, 4),  # Back face
                (0, 1, 5), (5, 4, 0),  # Bottom face
                (2, 3, 7), (7, 6, 2),  # Top face
                (0, 3, 7), (7, 4, 0),  # Left face
                (1, 2, 6), (6, 5, 1)  # Right face
            ]
            base_index = body_id * 8
            for tri in indices:
                triangles.add_vertices(base_index + tri[0], base_index + tri[1], base_index + tri[2])
                print(f"Triangle: {base_index + tri[0]}, {base_index + tri[1]}, {base_index + tri[2]}")  # 打印三角形顶点索引

        geom = Geom(vertex_data)
        geom.add_primitive(triangles)
        self.geom_node.add_geom(geom)
        print("Geometry created successfully")

        # 设置材质，使几何体具有颜色
        material = Material()
        material.set_diffuse((0.6, 0.4, 0.2, 1))
        self.node_path.set_material(material)
        self.node_path.set_attrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone))  # 启用双面渲染

        print("Geometry created successfully")

    def update_simulation(self, task):
        mujoco.mj_step(self.model, self.data)
        print("MuJoCo step completed")

        # 更新 Panda3D 几何
        self.create_geom_from_mujoco()

        return task.cont


# 设置 humanoid 模型文件路径
model_path = "humanoid.xml"  # 确保你的模型文件在当前目录中

# 检查模型路径是否存在
if not os.path.exists(model_path):
    raise Exception(f"Model file not found at {model_path}")

# 创建并运行 MuJoCo 仿真
sim = MuJoCoSimulator(model_path)
sim.run()
