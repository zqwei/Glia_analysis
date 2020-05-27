from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np


def comp_plot(nz, nc, brain_map=None, w=None, x=None, y=None, z=None):
    plt.figure(figsize=(10, 6))
    plt.imshow(brain_map[nz], cmap=plt.cm.gray)
    idx = (np.abs(w[:,nc])>0) & (z==nz)
    c_max=np.abs(w[:,1]).max()
    plt.scatter(x[idx], y[idx], c=w[idx,nc], cmap=plt.cm.Spectral, vmax=c_max, vmin=-c_max)
    plt.axis('off')
    plt.xlim([0, brain_map.shape[2]])
    plt.ylim([0, brain_map.shape[1]])
    plt.show()
    return nz, nc