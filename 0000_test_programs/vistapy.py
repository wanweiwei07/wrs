import pyvista as pv
import numpy as np

# 3 个点
points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
])

# 一个三角面：3 个顶点
faces = np.hstack([[3, 0, 1, 2]])

mesh = pv.PolyData(points, faces)
mesh.plot()