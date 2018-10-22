import argparse
import math
import shutil
from collections import defaultdict
from pathlib import Path
import ujson as json

import time

import numpy as np
import torch
from torch.nn import functional as F
import visdom
from torch.utils.data import DataLoader
from torch import nn, optim
from torchnet.meter import AverageValueMeter
from torchvision.models import resnet
from tqdm import tqdm

from terial.classifier import transforms
from terial.config import SUBSTANCES
from terial.classifier.opensurfaces import dataset
from toolbox.colors import visualize_lab_color_hist

tqdm.monitor_interval = 0

from terial.classifier.utils import (decay_learning_rate,
                                     compute_precision, visualize_input,
                                     MeterTable, ColorBinner)
from terial.classifier.network import RendNet3

INPUT_SIZE = 224
SHAPE = (384, 384)

CROPPED_MAX_ROTATION = 90
CROPPED_CROP_RANGE = (0.1, 1.0)

MAX_ROTATION = 15
CROP_RANGE = (0.4, 1.0)


parser = argparse.ArgumentParser()
parser.add_argument('--opensurfaces-dir', type=Path,
                    default='/local1/kpar/data/opensurfaces')
parser.add_argument('--checkpoint-dir', type=Path, required=True)
parser.add_argument('--resume', type=Path)
parser.add_argument('--color-hist-name', type=str)
parser.add_argument('--split-file', type=str, default='20180521-split.json')
parser.add_argument('--model-name', type=str, required=True)
parser.add_argument('--base-model', type=str, default='resnet34')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--init-lr', type=float, default=0.00005)
parser.add_argument('--lr-decay-epochs', type=int, default=60)
parser.add_argument('--lr-decay-frac', type=float, default=0.5)
parser.add_argument('--substance-loss', type=str,
                    choices=['fc'])
parser.add_argument('--color-hist-space', type=str,
                    default='lab')
parser.add_argument('--color-hist-shape', type=str,
                    default='3,5,5')
parser.add_argument('--color-hist-sigma', type=str,
                    default='0.5,1.0,1.0')
parser.add_argument('--color-loss', type=str,
                    choices=['cross_entropy',
                             'kl_divergence',
                             'none'])
parser.add_argument('--use-variance', action='store_true')
parser.add_argument('--substance-variance-init', type=float, default=-3.0)
parser.add_argument('--color-variance-init', type=float, default=-3.0)
parser.add_argument('--substance-loss-weight', type=float, default=0.5)
parser.add_argument('--color-loss-weight', type=float, default=0.2)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--from-scratch', action='store_true')
parser.add_argument('--use-cropped', action='store_true')
parser.add_argument('--p-cropped', default=0.5)
parser.add_argument('--visdom-port', type=int, default=8097)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--show-freq', type=int, default=4)
args = parser.parse_args()

if args.substance_loss == 'none':
    args.substance_loss = None
if args.color_loss == 'none':
    args.color_loss = None


# Parse color histogram information.
if args.color_loss:
    color_binner = ColorBinner(
        args.color_hist_space,
        tuple(int(i) for i in args.color_hist_shape.split(',')),
        tuple(float(i) for i in args.color_hist_sigma.split(',')),
    )
else:
    color_binner = None


vis = visdom.Visdom(env=f'pretrain-{args.model_name}',
                    port=args.visdom_port)


def main():
    checkpoint_dir = args.checkpoint_dir / args.model_name
    split_path = args.opensurfaces_dir / args.split_file

    with split_path.open('r') as f:
        split_dict = json.load(f)

    print(
        f' * opensurface_dir = {args.opensurfaces_dir!s}\n'
        f' * split_path = {split_path!s}\n'
        f' * checkpoint_dir = {checkpoint_dir!s}'
    )

    print(' * Loading datasets')

    output_substance = (True if args.substance_loss == 'fc' else False)
    output_color = color_binner is not None

    train_dataset = dataset.OpenSurfacesDataset(
        base_dir=args.opensurfaces_dir,
        color_binner=color_binner,
        photo_ids=split_dict['train'],
        image_transform=transforms.train_image_transform(
            INPUT_SIZE,
            crop_scales=CROP_RANGE,
            max_rotation=MAX_ROTATION,
            max_brightness_jitter=0.2,
            max_contrast_jitter=0.2,
            max_saturation_jitter=0.2,
            max_hue_jitter=0.1,
        ),
        mask_transform=transforms.train_mask_transform(
            INPUT_SIZE,
            crop_scales=CROP_RANGE,
            max_rotation=MAX_ROTATION),
        cropped_image_transform=transforms.train_image_transform(
            INPUT_SIZE,
            pad=200,
            crop_scales=CROPPED_CROP_RANGE,
            max_rotation=CROPPED_MAX_ROTATION,
            max_brightness_jitter=0.2,
            max_contrast_jitter=0.2,
            max_saturation_jitter=0.2,
            max_hue_jitter=0.1,
        ),
        cropped_mask_transform=transforms.train_mask_transform(
            INPUT_SIZE,
            pad=200,
            crop_scales=CROPPED_CROP_RANGE,
            max_rotation=CROPPED_MAX_ROTATION),
        use_cropped=args.use_cropped,
        p_cropped=args.p_cropped,
    )

    validation_dataset = dataset.OpenSurfacesDataset(
        base_dir=args.opensurfaces_dir,
        color_binner=color_binner,
        photo_ids=split_dict['validation'],
        image_transform=transforms.inference_image_transform(
            INPUT_SIZE, INPUT_SIZE),
        mask_transform=transforms.inference_mask_transform(
            INPUT_SIZE, INPUT_SIZE),
        cropped_image_transform=transforms.inference_image_transform(
            INPUT_SIZE, INPUT_SIZE),
        cropped_mask_transform=transforms.inference_mask_transform(
            INPUT_SIZE, INPUT_SIZE),
        use_cropped=args.use_cropped,
        p_cropped=args.p_cropped,
    )

    model_params = dict(
        num_substances=len(SUBSTANCES),
        output_material=False,
        output_roughness=False,
        output_substance=output_substance,
        output_color=output_color,
        num_color_bins=color_binner.size if color_binner else 0,
    )

    base_model_fn = {
        'resnet18': resnet.resnet18,
        'resnet34': resnet.resnet34,
        'resnet50': resnet.resnet50,
    }[args.base_model]

    if args.from_scratch:
        base_model = base_model_fn(pretrained=False)
    else:
        base_model = base_model_fn(pretrained=True)

    model = RendNet3(**model_params,
                     base_model=base_model).train().cuda()

    train_cum_stats = defaultdict(list)
    val_cum_stats = defaultdict(list)
    if args.resume:
        print(f" * Loading weights from {args.resume!s}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        stats_dict_path = checkpoint_dir / 'model_train_stats.json'
        if stats_dict_path.exists():
            with stats_dict_path.open('r') as f:
                stats_dict = json.load(f)
                train_cum_stats = stats_dict['train']
                val_cum_stats = stats_dict['validation']

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False,
    )

    loss_weights = {
        # 'material': args.material_loss_weight,
        # 'roughness': args.roughness_loss_weight,
        'substance': args.substance_loss_weight,
        'color': args.color_loss_weight,
    }

    model_params = {
        'init_lr': args.init_lr,
        'lr_decay_epochs': args.lr_decay_epochs,
        'lr_decay_frac': args.lr_decay_frac,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'from_scratch': args.from_scratch,
        'base_model': args.base_model,
        'resumed_from': str(args.resume),

        'use_substance_loss': args.substance_loss is not None,
        'substance_loss': args.substance_loss,
        'color_loss': args.color_loss,
        'color_hist_name': args.color_hist_name,
        'use_variance': args.use_variance,
        'loss_weights': loss_weights,
        'use_cropped': args.use_cropped,
        'p_cropped': args.p_cropped,

        'model_params': model_params,
    }

    if color_binner:
        model_params = {
            **model_params,
            'color_hist_name': color_binner.name,
            'color_hist_shape': color_binner.shape,
            'color_hist_space': color_binner.space,
            'color_hist_sigma': color_binner.sigma,
        }

    vis.text(f'<pre>{json.dumps(model_params, indent=2)}</pre>', win='params')

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_params_path = (checkpoint_dir / 'model_params.json')
    if not model_params_path.exists():
        with model_params_path.open('w') as f:
            print(f' * Saving model params to {model_params_path!s}')
            json.dump(model_params, f, indent=2)

    subst_criterion = nn.CrossEntropyLoss().cuda()

    if args.color_loss == 'cross_entropy':
        color_criterion = nn.BCEWithLogitsLoss().cuda()
    elif args.color_loss == 'kl_divergence':
        color_criterion = nn.KLDivLoss().cuda()
    else:
        color_criterion = None
    loss_variances = {
        'substance': torch.tensor([args.substance_variance_init],
                                  requires_grad=True),
        'color': torch.tensor([args.color_variance_init], requires_grad=True),
    }

    optimizer = optim.SGD([*model.parameters(),
                           *loss_variances.values()], args.init_lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    if 'subst_fc_prec1' in val_cum_stats:
        best_prec1 = max(val_cum_stats['subst_fc_prec1'])
    else:
        best_prec1 = 0

    pbar = tqdm(total=args.epochs)
    for epoch in range(args.start_epoch, args.epochs):
        pbar.set_description(f'Epoch {epoch}')
        decay_learning_rate(optimizer,
                            epoch,
                            args.init_lr,
                            args.lr_decay_epochs,
                            args.lr_decay_frac)

        train_stats = train_epoch(
            train_loader, model, epoch,
            subst_criterion,
            color_criterion,
            optimizer,
            loss_variances=loss_variances,
            loss_weights=loss_weights,
        )

        # evaluate on validation set
        val_stats = validate_epoch(
            validation_loader, model, epoch,
            subst_criterion,
            color_criterion,
            loss_variances=loss_variances,
            loss_weights=loss_weights,
        )

        # remember best prec@1 and save checkpoint
        is_best = val_stats['subst_fc_prec1'] > best_prec1
        best_prec1 = max(val_stats['subst_fc_prec1'], best_prec1)

        for stat_name, stat_val in train_stats.items():
            train_cum_stats[stat_name].append(stat_val)

        for stat_name, stat_val in val_stats.items():
            val_cum_stats[stat_name].append(stat_val)

        stats_dict = {
            'epoch': epoch + 1,
            'train': train_cum_stats,
            'validation': val_cum_stats,
        }

        with (checkpoint_dir / 'model_train_stats.json').open('w') as f:
            json.dump(stats_dict, f)

        if is_best:
            with (checkpoint_dir / 'model_best_stats.json').open('w') as f:
                json.dump({
                    'epoch': epoch + 1,
                    'train': {k: v for k, v in train_stats.items()
                              if not math.isnan(v)},
                    'validation': {k: v for k, v in val_stats.items()
                                   if not math.isnan(v)},
                }, f)

        save_checkpoint(
            {
                'epoch': epoch + 1,
                'model_name': args.model_name,
                'state_dict': model.state_dict(),
                'loss_variances': loss_variances,
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
                'params': model_params,
            },
            checkpoint_dir=checkpoint_dir,
            filename=f'model.{args.model_name}.epoch{epoch:03d}.pth.tar',
            is_best=is_best)

        # Plot graphs.
        for stat_name, train_stat_vals in train_cum_stats.items():
            if stat_name not in val_cum_stats:
                continue
            tqdm.write(f"Plotting {stat_name}")
            val_stat_vals = val_cum_stats[stat_name]
            vis.line(Y=np.column_stack((train_stat_vals, val_stat_vals)),
                     X=np.array(list(range(0, len(train_stat_vals)))),
                     win=f'plot-{stat_name}',
                     name=stat_name,
                     opts={
                         'legend': ['training', 'validation'],
                         'title': stat_name,
                     })


def compute_substance_loss(subst_label, subst_output, criterion):
    subst_labels = subst_label.cuda()
    subst_loss = criterion(subst_output, subst_labels)
    subst_prec1 = compute_precision(subst_output, subst_labels)

    return subst_loss, subst_prec1


def compute_roughness_loss(roughness, roughness_output, criterion):
    roughness = roughness.float()
    roughness_prec1 = [0]
    if  args.roughness_loss == 'cross_entropy':
        roughness_labels = (
            (roughness * (args.num_roughness_classes - 1))
                .floor().long().cuda())
        roughness_loss = criterion(roughness_output, roughness_labels)
        roughness_prec1 = compute_precision(
            roughness_output, roughness_labels)
    elif args.roughness_loss == 'mse':
        roughness_loss = (roughness.cuda() - roughness_output).pow(2).mean()
        roughness_prec1 = roughness_loss.item()
    else:
        roughness_loss = 0

    return roughness_loss, roughness_prec1


def combine_losses(losses,
                   loss_weights=None,
                   loss_vars=None,
                   use_variance=False):
    total_loss = 0
    for name, loss in losses.items():
        var = loss_vars[name].cuda()
        if use_variance:
            total_loss += (
                torch.exp(-var) * losses[name] + var
            )
        else:
            total_loss += loss_weights[name] * loss

    return total_loss


def train_epoch(train_loader, model, epoch,
                subst_criterion,
                color_criterion,
                optimizer,
                loss_variances,
                loss_weights):
    meter_dict = defaultdict(AverageValueMeter)
    pbar = tqdm(total=len(train_loader), desc='Training Epoch')

    last_end_time = time.time()
    for batch_idx, batch_dict in enumerate(train_loader):
        meter_dict['data_time'].add(time.time() - last_end_time)

        input_var = batch_dict['image'].cuda()
        subst_labels = batch_dict['substance_label'].cuda()

        batch_start_time = time.time()
        output = model.forward(input_var)

        losses = {}

        subst_output = output['substance']
        subst_fc_prec1 = compute_precision(subst_output, subst_labels)
        meter_dict['subst_fc_prec1'].add(subst_fc_prec1)
        losses['substance'] = subst_criterion(subst_output, subst_labels)

        if args.color_loss is not None:
            color_output = output['color']
            color_target = batch_dict['color_hist'].cuda()
            if isinstance(color_criterion, nn.KLDivLoss):
                color_output = F.log_softmax(color_output)
            losses['color'] = color_criterion(color_output, color_target)
        else:
            color_output = None
            color_target = None

        loss = combine_losses(losses, loss_weights, loss_variances,
                              args.use_variance)

        # Add losses to meters.
        meter_dict['loss'].add(loss.item())
        for loss_name, loss_tensor in losses.items():
            meter_dict[f'loss_{loss_name}'].add(loss_tensor.item())

        for loss_name, var_tensor in loss_variances.items():
            meter_dict[f'{loss_name}_variance(s_hat)'].add(var_tensor.item())
            meter_dict[f'{loss_name}_variance(exp[-s_hat])'].add(
                torch.exp(-var_tensor).item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        meter_dict['batch_time'].add(time.time() - batch_start_time)
        last_end_time = time.time()
        pbar.update()

        if batch_idx % args.show_freq == 0:
            # Visualize batch input.
            vis.image(visualize_input(batch_dict),
                      win='train-batch-example',
                      opts=dict(title='Train Batch Example'))
            # Visualize color prediction.
            if 'color' in losses:
                visualize_color_hist_output(
                    color_output[0],
                    color_target[0],
                    win='train-color-hist',
                    title='Training Color Histogram',
                    color_hist_shape=color_binner.shape,
                    color_hist_space=color_binner.space,
                )
            # Show meters.
            meters = [(k, v) for k, v in meter_dict.items()]
            meter_table = MeterTable(
                f"Epoch {epoch} Iter {batch_idx+1}/{len(train_loader)}",
                meters)

            vis.text(
                meter_table.render(),
                win='train-status', opts=dict(title='Training Status'))

    mean_stat_dict = {
        k: v.mean for k, v in meter_dict.items()
    }

    return mean_stat_dict


def validate_epoch(val_loader, model, epoch,
                   subst_criterion,
                   color_criterion,
                   loss_variances,
                   loss_weights):
    meter_dict = defaultdict(AverageValueMeter)

    # switch to evaluate mode
    model.eval()

    pbar = tqdm(total=len(val_loader), desc='Validating Epoch')

    last_end_time = time.time()
    for batch_idx, batch_dict in enumerate(val_loader):
        input_var = batch_dict['image'].cuda()
        subst_labels = batch_dict['substance_label'].cuda()

        output = model.forward(input_var)

        losses = {}

        subst_output = output['substance']
        subst_fc_prec1 = compute_precision(subst_output, subst_labels)
        meter_dict['subst_fc_prec1'].add(subst_fc_prec1)
        losses['substance'] = subst_criterion(subst_output, subst_labels)

        if args.color_loss is not None:
            color_output = output['color']
            color_target = batch_dict['color_hist'].cuda()
            if isinstance(color_criterion, nn.KLDivLoss):
                color_output = F.log_softmax(color_output)
            losses['color'] = color_criterion(color_output, color_target)
        else:
            color_output = None
            color_target = None

        loss = combine_losses(losses, loss_weights, loss_variances,
                              args.use_variance)

        # Add losses to meters.
        meter_dict['loss'].add(loss.item())
        for loss_name, loss_tensor in losses.items():
            meter_dict[f'loss_{loss_name}'].add(loss_tensor.item())

        meter_dict['batch_time'].add(time.time() - last_end_time)
        last_end_time = time.time()
        pbar.update()

        if batch_idx % args.show_freq == 0:
            vis.image(visualize_input(batch_dict),
                      win='validation-batch-example',
                      opts=dict(title='Validation Batch Example'))
            if 'color' in losses:
                visualize_color_hist_output(
                    color_output[0], color_target[0],
                    'validation-color-hist',
                    'Validation Color Histogram',
                    color_hist_shape=color_binner.shape,
                    color_hist_space=color_binner.space,
                )
        meters = [(k, v) for k, v in meter_dict.items()]
        meter_table = MeterTable(
            f"Epoch {epoch} Iter {batch_idx+1}/{len(val_loader)}",
            meters)

        vis.text(
            meter_table.render(),
            win='validation-status', opts=dict(title='Validation Status'))

        # measure elapsed time
        meter_dict['batch_time'].add(time.time() - last_end_time)
        last_end_time = time.time()

    mean_stat_dict = {
        k: v.mean for k, v in meter_dict.items()
    }
    return mean_stat_dict


def save_checkpoint(state, checkpoint_dir, filename,
                    is_best=False):
    save_path = str(Path(checkpoint_dir, filename))
    torch.save(state, save_path)
    if is_best:
        best_path = str(Path(checkpoint_dir, 'model_best.pth.tar'))
        shutil.copyfile(save_path, best_path)


def visualize_color_hist_output(output, target, win, title,
                                color_hist_shape,
                                color_hist_space):
    if color_hist_space == 'lab':
        color_output_vis = np.hstack(
            visualize_lab_color_hist(
                F.softmax(output.cpu().detach(), dim=0).numpy(),
                num_bins=color_hist_shape))
        color_target_vis = np.hstack(
            visualize_lab_color_hist(
                target.cpu().detach().numpy(),
                num_bins=color_hist_shape))
    else:
        color_output_vis = np.hstack(
            F.softmax(output.cpu().detach(), dim=0)
                .view(*color_hist_shape)
                .numpy())
        color_target_vis = np.hstack(
            target.view(*color_hist_shape).cpu().detach().numpy())
    vis.heatmap(np.vstack((color_output_vis, color_target_vis)),
                win=win,
                opts=dict(title=title))


if __name__ == '__main__':
    main()

