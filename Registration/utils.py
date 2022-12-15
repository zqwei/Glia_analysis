import numpy as np
import matplotlib.pyplot as plt
import os
from fish_proc.utils import dask_ as fdask
from fish_proc.wholeBrainDask.cellProcessing_single_WS import *
from glob import glob
import matplotlib.pyplot as plt
import nibabel as nib
import SimpleITK as sitk


def read_h5(filename, dset_name='default'):
    import h5py
    with h5py.File(filename, 'r') as hf:
        return hf[dset_name][()]

    
def getCameraInfo(file):
    from xml.dom import minidom
    camera_info = dict()
    xmldoc = minidom.parse(file)
    itemlist = xmldoc.getElementsByTagName('info')
    for s in itemlist:
        camera_info.update(dict(s.attributes.items()))
    itemlist = xmldoc.getElementsByTagName('action')
    for s in itemlist:
        camera_info.update(dict(s.attributes.items()))
    return camera_info


def read_reg_mat(file):
    with open(file, 'r') as f:
        l = [[float(num) for num in line.replace(' \n', '').split(' ')] for line in f]
    return np.array(l)