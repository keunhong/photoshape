import math

import PIL.Image
from torchvision import transforms


CROP_SCALE_RANGE = (0.5, 1.0)
MAX_ROTATION = 5
MAX_BRIGHTNESS_JITTER = 0.2
MAX_CONTRAST_JITTER = 0.2
MAX_SATURATION_JITTER = 0.2
MAX_HUE_JITTER = 0.1


denormalize_transform = transforms.Compose([
    transforms.Normalize(mean=[0, 0, 0, 0],
                         std=[1/0.229, 1/0.224, 1/0.225, 1]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406, 0],
                         std=[1, 1, 1, 1]),
])


def rotate_padded(fill, resample, input_size, max_rotation):
    pad_radius = input_size / 2 * math.sqrt(2)
    pad_size = pad_radius - input_size / 2
    return transforms.Compose([
        transforms.Pad(int(math.ceil(pad_size)), fill=fill),
        transforms.RandomRotation(max_rotation, resample=resample),
        transforms.CenterCrop(input_size),
    ])


def train_image_transform(input_size,
                          crop_scales=CROP_SCALE_RANGE,
                          max_rotation=MAX_ROTATION,
                          max_brightness_jitter=MAX_BRIGHTNESS_JITTER,
                          max_contrast_jitter=MAX_CONTRAST_JITTER,
                          max_saturation_jitter=MAX_SATURATION_JITTER,
                          max_hue_jitter=MAX_HUE_JITTER,
                          pad=0,
                          pad_fill=(255, 255, 255)):
    tforms = []
    if pad > 0:
        tforms.append(transforms.Pad(pad, fill=pad_fill))
    return transforms.Compose([
        # transforms.ToPILImage(mode='RGB'),
        *tforms,
        transforms.RandomResizedCrop(size=input_size, scale=crop_scales),
        transforms.RandomHorizontalFlip(),
        rotate_padded(fill=(255, 255, 255), resample=PIL.Image.BILINEAR,
                      input_size=input_size, max_rotation=max_rotation),
        transforms.ColorJitter(brightness=max_brightness_jitter,
                               contrast=max_contrast_jitter,
                               saturation=max_saturation_jitter,
                               hue=max_hue_jitter),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def train_mask_transform(input_size,
                         crop_scales=CROP_SCALE_RANGE,
                         max_rotation=MAX_ROTATION,
                         pad=0,
                         pad_fill=0):
    tforms = []
    if pad > 0:
        tforms.append(transforms.Pad(pad, fill=pad_fill))
    return transforms.Compose([
        transforms.ToPILImage(),
        *tforms,
        transforms.RandomResizedCrop(size=input_size,
                                     scale=crop_scales,
                                     interpolation=PIL.Image.NEAREST),
        transforms.RandomHorizontalFlip(),
        rotate_padded(fill=0,
                      resample=PIL.Image.NEAREST,
                      input_size=input_size,
                      max_rotation=max_rotation),
        transforms.ToTensor(),
    ])


def inference_image_transform(output_size,
                              input_size,
                              pad=0,
                              pad_fill=(255, 255, 255),
                              to_pil=False,
                              interpolation=PIL.Image.BILINEAR):
    tform_list = []
    if to_pil:
        tform_list.append(transforms.ToPILImage(mode='RGB'))
    if pad > 0:
        tform_list.append(transforms.Pad(pad, fill=pad_fill))
    tform_list.extend([
        transforms.Resize(input_size, interpolation=interpolation),
        transforms.CenterCrop(size=output_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transforms.Compose(tform_list)


def inference_mask_transform(input_size,
                             output_size,
                             pad=0,
                             pad_fill=0):
    tform_list = [transforms.ToPILImage()]

    if pad > 0:
        tform_list.append(transforms.Pad(pad, fill=pad_fill))
    tform_list.extend([
        transforms.Resize(input_size, interpolation=PIL.Image.NEAREST),
        transforms.CenterCrop(size=output_size),
        transforms.ToTensor(),
    ])
    return transforms.Compose(tform_list)
