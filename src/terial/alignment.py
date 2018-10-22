import numpy as np
from skimage.filters import gaussian
from skimage.transform import resize

import pyhog


def compute_features(image, bin_size, im_shape):
    image = resize(image, im_shape, anti_aliasing=True, mode='constant')
    image = gaussian(image, sigma=0.1, multichannel=True)
    return _compute_hog_features(image, bin_size=bin_size)


def _compute_hog_features(image, bin_size=8):
    image = image.astype(dtype=np.float32)
    for c in range(3):
        image[:, :, c] -= image.mean()
        image[:, :, c] /= image.std()

    padded = np.dstack([np.pad(image[:, :, d], bin_size,
                               mode='constant', constant_values=image.mean())
                        for d in range(image.shape[-1])])
    feat = pyhog.compute_pedro(padded.astype(dtype=np.float64), bin_size)
    feat = feat[:, :, -8:]
    feat = feat.reshape((1, -1))
    return feat.astype(np.float32)
