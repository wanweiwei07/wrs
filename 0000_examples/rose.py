import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define parameters
n = 800
A = 1.995653
B = 1.27689
C = 8

r = np.linspace(0, 1, n)
theta = np.linspace(-2, 20 * np.pi, n)
R, THETA = np.meshgrid(r, theta)

# Define the number of petals we want per cycle
petalNum = 3.6
x = 1 - (1/2) * ((5/4) * (1 - np.mod(petalNum * THETA, 2 * np.pi) / np.pi) ** 2 - 1/4) ** 2
phi = (np.pi / 2) * np.exp(-THETA / (C * np.pi))
y = A * (R ** 2) * (B * R - 1) ** 2 * np.sin(phi)
R2 = x * (R * np.sin(phi) + y * np.cos(phi))
X = R2 * np.sin(THETA)
Y = R2 * np.cos(THETA)
Z = x * (R * np.cos(phi) - y * np.sin(phi))

# Define a red colormap that matches the shape of the surface
color_values = (Z - Z.min()) / (Z.max() - Z.min())  # Normalize values
colors = plt.cm.Reds(color_values)

# Plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, facecolors=colors, linewidth=0, antialiased=True)
ax.view_init(elev=42, azim=-40.5)

plt.show()
