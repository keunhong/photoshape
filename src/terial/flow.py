import numpy as np
from toolbox.images import resize
from skimage.color import rgb2hsv, hsv2rgb


def visualize_flow(vx, vy):
    vx = vx.copy()
    vy = vy.copy()
    vx = vx / (vx.shape[1] / 2)
    vy = vy / (vy.shape[0] / 2)
    h = np.arctan2(vy, vx) / 2 + 0.5
    s = np.sqrt(vx ** 2 + vy ** 2).clip(0, 1)
    v = np.ones(vx.shape)
    return (hsv2rgb(np.dstack((h, s, v))) * 255).astype(np.uint8)


def apply_flow(image, vx, vy):
    yy, xx = np.mgrid[0:vx.shape[0], 0:vx.shape[1]]
    yy, xx = np.round(yy + vy), np.round(xx + vx)
    yy = np.clip(yy, 0, vx.shape[0]-1).astype(dtype=int)
    xx = np.clip(xx, 0, vx.shape[1]-1).astype(dtype=int)
    warped = image[yy, xx]
    return warped


def resize_flow(vx, vy, shape):
    scaley = shape[0] / vx.shape[0]
    scalex = shape[1] / vx.shape[1]
    vx = resize(vx * scalex, shape, order=3)
    vy = resize(vy * scaley, shape, order=3)
    return vx, vy
