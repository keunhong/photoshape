import operator
import random
from functools import reduce

import torch
from jinja2 import Template
from torch.nn import functional as F

import numpy as np

import toolbox.colors
from terial.config import SUBSTANCES
from terial.classifier.transforms import denormalize_transform
from toolbox.colors import visualize_lab_color_hist, lab_rgb_gamut_bin_mask


class MeterTable(object):
    _tmpl = Template('''
    <h5>{{title}}</h5>
    <table class="table table-small table-bordered" 
           style="font-family: monaco; font-size: 10px !important;">
        <thead>
            <tr>
                <th>Name</th>
                <th>Current</th>
                <th>Mean</th>
                <th>Std</th>
            </tr>
        </thead>
        <tbody>
            {% for name, meter in meters %}
                <tr>
                    <td>{{name}}</td>
                    <td>{{meter.val}}</td>
                    <td>{{meter.mean}}</td>
                    <td>{{meter.std}}</td>
                </tr>
            {% endfor %}
        </tbody>
    </table>
    ''')
    def __init__(self, title='', meters=None):
        self.title = title
        if meters is None:
            self.meters = []
        else:
            self.meters = meters

    def add(self, meter, name):
        self.meters.append((name, meter))

    def render(self):
        return self._tmpl.render(title=self.title,
                                 meters=self.meters)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_precision(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    if len(res) > 1:
        return res
    else:
        return res[0]


def material_to_substance_output(mat_output, subst_mat_labels):
    mat_output = F.softmax(mat_output, dim=1)
    subst_output = torch.zeros(mat_output.size(0), len(SUBSTANCES)).cuda()
    for subst_label, mat_labels in subst_mat_labels.items():
        subst_output[:, subst_label] = mat_output[:, mat_labels].sum(dim=1)
    return torch.log(subst_output)


def decay_learning_rate(optimizer, epoch, init_lr, decay_epochs, decay_frac):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (decay_frac ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def visualize_input(batch_dict, type='side_by_side'):
    rand_idx = random.randrange(0, batch_dict['image'].size(0))
    # WARNING: transforms do operations in-place, we MUST clone here.
    rand_im = denormalize_transform(batch_dict['image'][rand_idx].clone())

    if type == 'side_by_side':
        return torch.cat(
            (rand_im[:3],
             rand_im[3].unsqueeze(0).expand(*rand_im[:3].size())), dim=2)
    elif type == 'overlay':
        mask_vis = torch.cat((
            rand_im[3].unsqueeze(0),
            torch.zeros(1, *rand_im[3].size()),
            torch.zeros(1, *rand_im[3].size()),
        ), dim=0)
        return rand_im[:3]/2 + mask_vis/2


class ColorBinner(object):
    def __init__(self, space, shape, sigma):
        self.space = space
        self.shape = shape
        self.sigma = sigma

        if space not in {'lab', 'rgb'}:
            raise ValueError(f"Unknown color space {self.space}")

        if space == 'lab':
            self.mask, _ = lab_rgb_gamut_bin_mask(shape)
            self.size = self.mask.sum().item()
        else:
            self.size = reduce(operator.mul, shape)
            self.mask = np.ones(shape, dtype=bool)

    def compute(self, image, mask):
        if not hasattr(image, 'shape'):
            image = np.array(image)

        colors = image[mask].reshape((-1, 1, 3))
        if self.space == 'lab':
            hist = toolbox.colors.compute_lab_histogram(
                colors, self.shape, self.sigma)
        else:
            hist = toolbox.colors.compute_rgb_histogram(
                colors, self.shape, self.sigma)

        return hist

    def visualize(self, hist):
        if self.space == 'lab':
            return np.hstack(
                visualize_lab_color_hist(
                    hist.cpu().detach().numpy(),
                    num_bins=self.shape))
        else:
            return hist.view(*self.shape).cpu().detach().numpy()

    @property
    def name(self):
        shape_str = '_'.join([str(s) for s in self.shape])
        sigma_str = '_'.join(['{:.01f}'.format(s) for s in self.sigma])
        return f"{self.space}_{shape_str}_{sigma_str}"
