from onlikhorn.dataset import get_3d_data
import matplotlib.pyplot as plt

def display_cloud(ax, measure, color):
    w_i, x_i = measure[0], measure[1]

    ax.view_init(elev=110, azim=-90)
    # ax.set_aspect('equal')

    weights = w_i / w_i.sum()
    ax.scatter(x_i[:, 0], x_i[:, 1], x_i[:, 2],
               s=25 * 500 * weights, c=color)

    ax.axes.set_xlim3d(left=-1.4, right=1.4)
    ax.axes.set_ylim3d(bottom=-1.4, top=1.4)
    ax.axes.set_zlim3d(bottom=-1.4, top=1.4)

from mpl_toolkits.mplot3d import Axes3D


dragon, sphere = get_3d_data()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

display_cloud(ax, dragon, 'red')
display_cloud(ax, sphere, 'blue')
plt.show()
