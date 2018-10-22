import io

from PIL import Image

from toolbox.images import apply_mask


def compute_saturated_frac(image, fg_mask):
    if fg_mask.sum() <= 0:
        return 1.0
    saturated_mask = (apply_mask(image, fg_mask) >= 255).sum(axis=2) >= 2
    num_saturated = saturated_mask.sum()
    return num_saturated / fg_mask.sum()


def image_to_bytes(image: Image, buf=None):
    if buf is None:
        buf = io.BytesIO()
    image.save(buf, format='JPEG')
    return buf.getvalue()


def bytes_to_image(b, buf=None):
    if buf is None:
        buf = io.BytesIO()
    buf.seek(0)
    buf.write(b)
    buf.seek(0)
    return Image.open(buf)
