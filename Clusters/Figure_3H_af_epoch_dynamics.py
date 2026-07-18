'''
Figure 3H: average epoch dynamics (probe/interval/evoke/swim) for the
SloMO_Memory brain cluster, compared across CL vs OL trials.
'''

from utils import *
dat_folder = 'depreciated/_dat_brain_clusters_dynamcis/epochs/'

label_ = 'SloMO_Memory'
dat_evoke_list = []
dat_probe_list = []
dat_int_list = []
dat_swim_list = []
for n in [0, 2, 4]:
    _ = np.load(dat_folder+f'fish_{n}_{label_}.npz', allow_pickle=True)
    dat_evoke = _['dat_evoke']
    dat_probe = _['dat_probe']
    dat_int = _['dat_int']
    dat_swim = _['dat_swim']
    dat_evoke_list.append(dat_evoke[:, -15:] - dat_probe[:, :5].mean())
    dat_probe_list.append(dat_probe - dat_probe[:, :5].mean())
    if dat_int.shape[1]>90:
        dat_int_list.append(dat_int[:, 0:90:3] - dat_probe[:, :5].mean())
    else:
        dat_int_list.append(dat_int[:, :30] - dat_probe[:, :5].mean())
    dat_swim_list.append(dat_swim - dat_probe[:, :5].mean())
dat_evoke_list = np.array(dat_evoke_list)
dat_probe_list = np.array(dat_probe_list)
dat_int_list = np.array(dat_int_list)
dat_swim_list = np.array(dat_swim_list)

fig, ax = plt.subplots(1, 4, figsize=(8, 3))
ax = ax.flatten()
fig.subplots_adjust(wspace=0)

ymin, ymax = -5, 5
plot_shade_err(np.arange(-5, 35)/3, dat_probe_list[:, 0, :], axis=0, \
               plt=ax[0], linespec='-k', shadespec='k', err_f=2)
plot_shade_err(np.arange(-5, 35)/3, dat_probe_list[:, 1, :], axis=0, \
               plt=ax[0], linespec='-r', shadespec='r', err_f=2)
ax[0].vlines(0, -1, 10, linestyles='--', colors='k')
ax[0].set_xlim([-5/3, 34/3])
ax[0].set_ylim([ymin, ymax])

plot_shade_err(np.arange(0, 30)/3, dat_int_list[:, 0, :], axis=0, \
               plt=ax[1], linespec='-k', shadespec='k')
plot_shade_err(np.arange(0, 30)/3, dat_int_list[:, 1, :], axis=0, \
               plt=ax[1], linespec='-r', shadespec='r')
ax[1].set_xlim([0, 30/3])
ax[1].set_ylim([ymin, ymax])
ax[1].set_yticks([])

plot_shade_err(np.arange(0, 15), dat_evoke_list[:, 0, :], axis=0, \
               plt=ax[3], linespec='-k', shadespec='k')
plot_shade_err(np.arange(0, 15), dat_evoke_list[:, 1, :], axis=0, \
               plt=ax[3], linespec='-r', shadespec='r')
ax[3].set_xlim([0, 15])
ax[3].set_ylim([ymin, ymax])
ax[3].set_yticks([])



plot_shade_err(np.arange(-15, 5)/3, dat_swim_list[:, 0, -20:], axis=0, \
               plt=ax[2], linespec='-k', shadespec='k')
plot_shade_err(np.arange(-15, 5)/3, dat_swim_list[:, 1, -20:], axis=0, \
               plt=ax[2], linespec='-r', shadespec='r')
ax[2].set_xlim([-15/3, 4/3])
ax[2].vlines(0, -1, 10, linestyles='--', colors='k')
ax[2].set_ylim([ymin, ymax])
ax[2].set_yticks([])

sns.despine()
# plt.savefig('AF_cluster_SLoMO_probe.svg')
plt.show()