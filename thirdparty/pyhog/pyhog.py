import numpy as np
from scipy.misc import imrotate
import features_pedro_py


def compute_pedro(img, sbin):
    imgf = img.copy('F')
    hogf = features_pedro_py.process(imgf, sbin)
    return hogf

def visualize_weights(w, bs=20):
    """ Visualize positive HOG weights.
    ported to numpy from https://github.com/CSAILVision/ihog/blob/master/showHOG.m
    """
    bim1 = np.zeros((bs, bs))
    bim1[:,round(bs/2)-1:round(bs/2)] = 1
    bim = np.zeros((9,)+bim1.shape)
    for i in range(9):
      bim[i] = imrotate(bim1, -i*20)/255.0
    s = w.shape
    w = w.copy()
    w[w < 0] = 0
    im = np.zeros((bs*s[0], bs*s[1]))
    for i in range(s[0]):
      iis = slice( i*bs, (i+1)*bs )
      for j in range(s[1]):
        jjs = slice( j*bs, (j+1)*bs )
        for k in range(9):
          im[iis,jjs] += bim[k] * w[i,j,k+18]
    return im/np.max(w)
