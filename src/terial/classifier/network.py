import math

import torch
from torch import nn
from torchvision.models import resnet

from terial.config import SUBSTANCES


class RendNet3(nn.Module):
    @classmethod
    def from_checkpoint(cls, checkpoint):
        state_dict = checkpoint['state_dict']
        num_classes = state_dict['fc_material.bias'].size(0)

        output_substance = 'fc_substance.weight' in state_dict
        num_substances = (state_dict['fc_substance.bias'].size(0)
                          if output_substance else 0)

        output_roughness = 'fc_roughness.weight' in state_dict
        num_roughness_classes = (state_dict['fc_roughness.bias'].size(0)
                                 if output_roughness else 0)

        output_color = 'fc_color.weight' in state_dict
        num_color_bins = (state_dict['fc_color.bias'].size(0)
                          if output_color else 0)

        if 'params' in checkpoint:
            base_model_fn = {
                'resnet18': resnet.resnet18,
                'resnet34': resnet.resnet34,
                'resnet50': resnet.resnet50,
            }.get(checkpoint['params'].get('base_model', 'resnet18'))
        else:
            base_model_fn = resnet.resnet18

        model = cls(num_classes=num_classes,
                    num_roughness_classes=num_roughness_classes,
                    num_substances=num_substances,
                    output_substance=output_substance,
                    output_roughness=output_roughness,
                    output_color=output_color,
                    num_color_bins=num_color_bins,
                    base_model=base_model_fn(pretrained=False))
        model.load_state_dict(state_dict)
        return model

    def __init__(self,
                 num_substances=0,
                 num_classes=0,
                 num_roughness_classes=20,
                 num_color_bins=None,
                 output_material=True,
                 output_substance=False,
                 output_roughness=False,
                 output_color=False,
                 base_model=resnet.resnet18):
        super().__init__()

        self.conv1 = nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if hasattr(base_model, 'forward'):
            self.base_model = base_model
        else:
            self.base_model = base_model()

        self.fc_material = (
            nn.Linear(512 * resnet.BasicBlock.expansion, num_classes)
            if output_material
            else None)

        self.fc_substance = (
            nn.Linear(512 * resnet.BasicBlock.expansion, num_substances)
            if output_substance
            else None)

        self.num_roughness_classes = num_roughness_classes

        self.fc_roughness = (
            nn.Linear(512 * resnet.BasicBlock.expansion, num_roughness_classes)
            if output_roughness
            else None)

        self.num_color_bins = num_color_bins
        self.fc_color = (
            nn.Linear(512 * resnet.BasicBlock.expansion, num_color_bins)
            if output_color
            else None)

        self.conv1.weight.data[:, :3, :, :].copy_(
            self.base_model.conv1.weight.data[:, :3, :, :])

        # Pretrained network does not have alpha channel, so initialize it with
        # random normal.
        for m in [self.conv1]:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data[:, 3, :, :].normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        x = self.base_model.avgpool(x)
        x = x.view(x.size(0), -1)

        output = {}

        if self.fc_material:
            output['material'] = self.fc_material(x)

        if self.fc_substance:
            output['substance'] = self.fc_substance(x)

        if self.fc_roughness:
            output['roughness'] = self.fc_roughness(x)

        if self.fc_color:
            output['color'] = self.fc_color(x)

        return output
