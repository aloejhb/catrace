import numpy as np
from scipy.io import loadmat

data = loadmat('../../../neuRoi/colormap/clut2b.mat')
colormap_data = data['clut2b']

np.save('../colormap/clut2b.npy', colormap_data)

