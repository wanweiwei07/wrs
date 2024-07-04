import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成一个简单的MURA掩模示例
def mura_mask(size):
    p = np.zeros((size, size), dtype=int)
    for y in range(size):
        for x in range(size):
            p[y, x] = 1 if (x*y) % size < size // 2 else 0
    return p

# 生成一个5x5的MURA掩模
size = 5
mura = mura_mask(size)

# 计算自相关函数
def auto_correlation(mask):
    size = mask.shape[0]
    correlation = np.zeros((2*size-1, 2*size-1))
    for y in range(size):
        for x in range(size):
            for j in range(size):
                for i in range(size):
                    correlation[y+j, x+i] += mask[y, x] * mask[j, i]
    return correlation

correlation = auto_correlation(mura)

# 创建网格
x = np.arange(-size+1, size)
y = np.arange(-size+1, size)
x, y = np.meshgrid(x, y)
z = correlation

# 绘制三维图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')

ax.set_title('3D Auto-correlation of MURA Mask')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Correlation')

plt.show()
