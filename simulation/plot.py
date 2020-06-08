import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors

div_norm = colors.DivergingNorm(vmin=0.9, vcenter=1.25, vmax=2.2)
theta1 = np.arange(-4, 4.1, step = 0.4)
theta2 = np.arange(-4, 4.1, step= 0.4)
T2, T1 = np.meshgrid(theta2, theta1)

fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = [17, 5])


b = np.load('data/test_stat_0.npy')
im0 = ax[0].pcolormesh(T1, T2, b, shading='gouraud', norm = div_norm, cmap = 'seismic')
ax[0].contour(T1, T2, b, levels = [1.25, ])
ax[0].set_title('Rotation $0^O$')
ax[0].set_xlabel('$\\theta_1$', fontsize = 'x-large')
ax[0].set_ylabel('$\\theta_2$' , fontsize = 'x-large')


b = np.load('data/test_stat_5.npy')
im1 = ax[1].pcolormesh(T1, T2, b, shading='gouraud', norm = div_norm, cmap = 'seismic')
ax[1].set_title('Rotation $5^O$')
ax[1].set_xlabel('$\\theta_1$', fontsize = 'x-large')
ax[1].set_ylabel('$\\theta_2$', fontsize = 'x-large')
ax[1].contour(T1, T2, b, levels = [1.25, ])

b = np.load('data/test_stat_10.npy')
im2 = ax[2].pcolormesh(T1, T2, b, shading='gouraud', norm = div_norm, cmap = 'seismic')
ax[2].set_title('Rotation $10^O$')
ax[2].set_xlabel('$\\theta_1$', fontsize = 'x-large')
ax[2].set_ylabel('$\\theta_2$', fontsize = 'x-large')
ax[2].contour(T1, T2, b, levels = [1.25, ])


fig.colorbar(im2, ax = ax, orientation='horizontal', fraction = 0.1)
plt.savefig('plots/mean_ratios.pdf')