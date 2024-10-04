import matplotlib.pyplot as plt
from wrs import basis as rm

xy_array = rm.gen_2d_equilateral_verts(5, edge_length=.9)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(xy_array[:,0], xy_array[:,1],'-o')
plt.plot(xy_array[0,0], xy_array[0,1],'ro')
ax.set_aspect('equal', adjustable='box')
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.show()