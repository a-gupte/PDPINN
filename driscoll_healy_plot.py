import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Create a sphere
r = 1
pi = np.pi
cos = np.cos
sin = np.sin

N = 50

azimuth = np.linspace(0, np.pi, N)
polar = np.linspace(0, 2 * np.pi, 2*N)
azimuth, polar = np.meshgrid(azimuth, polar)
x = sin(azimuth) * cos(polar)
y = sin(azimuth) * sin(polar)
z = cos(azimuth)

#Set colours and render
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ax.plot_surface(
#     x, y, z,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)

ax.scatter(x, y, z, color="k", s=1)

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
# ax.set_aspect("equal")
plt.tight_layout()
plt.show()
