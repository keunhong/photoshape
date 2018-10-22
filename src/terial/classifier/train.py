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

import terial.classifier.utils
from terial import models
from terial.classifier import transforms, rendering_dataset
from terial.config import SUBSTANCES
from terial.database import session_scope

tqdm.monitor_interval = 0

from terial.classifier.utils import (decay_learning_rate,
                                     compute_precision, visualize_input,
                                     MeterTable, material_to_substance_output)

from .network import RendNet3

INPUT_SIZE = 224
SHAPE = (384, 384)


parser = argparse.ArgumentParser()
parser.add_argument('--snapshot-dir', type=Path, required=True)
parser.add_argument('--checkpoint-dir', type=Path, required=True)
parser.add_argument('--resume', type=Path)
parser.add_argument('--model-name', type=str, required=True)
parser.add_argument('--base-model', type=str, default='resnet34')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--init-lr', type=float, default=0.00005)
parser.add_argument('--lr-decay-epochs', type=int, default=60)
parser.add_argument('--lr-decay-frac', type=float, default=0.5)
parser.add_argument('--substance-loss', type=str,
                    choices=['fc', 'from_material', 'none'])
parser.add_argument('--roughness-loss', type=str,
                    default='none',
                    choices=['mse', 'cross_entropy', 'none'])
parser.add_argument('--color-hist-space', type=str,
                    default='lab')
parser.add_argument('--color-hist-shape', type=str,
                    default='3,5,5')
parser.add_argument('--color-hist-sigma', type=str,
                    default='0.5,1.0,1.0')
parser.add_argument('--color-loss', type=str,
                    choices=['cross_entropy',
                             'base_color_ce',
                             'none'])
parser.add_argument('--use-variance', action='store_true')
parser.add_argument('--material-variance-init', type=float, default=0.0)
parser.add_argument('--substance-variance-init', type=float, default=-3.0)
parser.add_argument('--color-variance-init', type=float, default=-3.0)
parser.add_argument('--material-loss-weight', type=float, default=1.0)
parser.add_argument('--substance-loss-weight', type=float, default=0.5)
parser.add_argument('--roughness-loss-weight', type=float, default=0.0)
parser.add_argument('--color-loss-weight', type=float, default=0.2)
parser.add_argument('--num-roughness-classes', type=int, default=20)
parser.add_argument('--mask-noise-p', type=float, default=0.003)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--from-scratch', action='store_true')
parser.add_argument('--visdom-port', type=int, default=8097)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--show-freq', type=int, default=4)
parser.add_argument('--dry-run', action='store_true')
args = parser.parse_args()

if args.substance_loss == 'none':
    args.substance_loss = None
if args.roughness_loss == 'none':
    args.roughness_loss = None
if args.color_loss == 'none':
    args.color_loss = None


# Parse color histogram information.
if args.color_loss:
    color_binner = terial.classifier.utils.ColorBinner(
        args.color_hist_space,
        tuple(int(i) for i in args.color_hist_shape.split(',')),
        tuple(float(i) for i in args.color_hist_sigma.split(',')),
    )
else:
    color_binner = None


vis = visdom.Visdom(env=f'classifier-train-{Path(args.snapshot_dir).name}-{args.model_name}',
                    port=args.visdom_port)


def main():
    snapshot_dir = args.snapshot_dir
    checkpoint_dir = args.checkpoint_dir / args.model_name

    train_path = Path(snapshot_dir, 'train')
    validation_path = Path(snapshot_dir, 'validation')
    meta_path = Path(snapshot_dir, 'meta.json')

    print(
        f' * train_path = {train_path!s}\n'
        f' * validation_path = {validation_path!s}\n'
        f' * meta_path = {meta_path!s}\n'
        f' * checkpoint_dir = {checkpoint_dir!s}'
    )

    with meta_path.open('r') as f:
        meta_dict = json.load(f)

    with session_scope() as sess:
        materials = sess.query(models.Material).all()
        mat_by_id = {m.id: m for m in materials}


    mat_id_to_label = {int(k): v
                       for k, v in meta_dict['mat_id_to_label'].items()}
    subst_mat_labels = defaultdict(list)
    for mat_id, mat_label in mat_id_to_label.items():
        material = mat_by_id[mat_id]
        subst_mat_labels[material.substance].append(mat_label)
    subst_mat_labels = {
        SUBSTANCES.index(k): torch.LongTensor(v).cuda()
        for k, v in subst_mat_labels.items()
    }
    mat_label_to_subst_label = {
        label: mat_by_id[mat_id].substance
        for mat_id, label in mat_id_to_label.items()}

    num_classes = max(mat_id_to_label.values()) + 1

    if args.roughness_loss:
        num_roughness_classes = (args.num_roughness_classes
                                 if args.roughness_loss == 'cross_entropy'
                                 else 1)
        output_roughness = True
    else:
        num_roughness_classes = 0
        output_roughness = False

    output_substance = (True if args.substance_loss == 'fc' else False)

    model_params = dict(
        num_classes=num_classes,
        num_substances=len(SUBSTANCES),
        num_roughness_classes=num_roughness_classes,
        output_roughness=output_roughness,
        output_substance=output_substance,
        output_color=color_binner is not None,
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
                     base_model=base_model,
                     ).train().cuda()

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


    print(' * Loading datasets')

    train_dataset = rendering_dataset.MaterialRendDataset(
        train_path,
        meta_dict,
        color_binner=color_binner,
        shape=SHAPE,
        lmdb_name=snapshot_dir.name,
        image_transform=transforms.train_image_transform(INPUT_SIZE, pad=0),
        mask_transform=transforms.train_mask_transform(INPUT_SIZE, pad=0),
        mask_noise_p=args.mask_noise_p)

    validation_dataset = rendering_dataset.MaterialRendDataset(
        validation_path,
        meta_dict,
        color_binner=color_binner,
        shape=SHAPE,
        lmdb_name=snapshot_dir.name,
        image_transform=transforms.inference_image_transform(
            INPUT_SIZE, INPUT_SIZE),
        mask_transform=transforms.inference_mask_transform(
            INPUT_SIZE, INPUT_SIZE))

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
        'material': args.material_loss_weight,
        'roughness': args.roughness_loss_weight,
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
        'mask_noise_p': args.mask_noise_p,

        'use_substance_loss': args.substance_loss is not None,
        'substance_loss': args.substance_loss,
        'roughness_loss': args.roughness_loss,
        'material_variance_init': args.material_variance_init,
        'substance_variance_init': args.substance_variance_init,
        'color_variance_init': args.color_variance_init,
        'color_loss': args.color_loss,
        'num_classes': num_classes,
        'num_roughness_classes': (args.num_roughness_classes
                                  if args.roughness_loss else None),
        'use_variance': args.use_variance,
        'loss_weights': loss_weights,

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
    if not args.dry_run and not model_params_path.exists():
        with model_params_path.open('w') as f:
            print(f' * Saving model params to {model_params_path!s}')
            json.dump(model_params, f, indent=2)

    mat_criterion = nn.CrossEntropyLoss().cuda()
    if args.substance_loss == 'from_material':
        subst_criterion = nn.NLLLoss().cuda()
    else:
        subst_criterion = nn.CrossEntropyLoss().cuda()

    if args.color_loss == 'cross_entropy':
        color_criterion = nn.BCEWithLogitsLoss().cuda()
    elif args.color_loss == 'kl_divergence':
        color_criterion = nn.KLDivLoss().cuda()
    else:
        color_criterion = None

    loss_variances = {
        'material': torch.tensor([args.material_variance_init],
                                 requires_grad=True),
        'substance': torch.tensor([args.substance_variance_init],
                                  requires_grad=True),
        'color': torch.tensor([args.color_variance_init], requires_grad=True),
    }

    optimizer = optim.SGD([*model.parameters(),
                           *loss_variances.values()], args.init_lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    best_prec1 = 0
    pbar = tqdm(total=args.epochs, dynamic_ncols=True)
    for epoch in range(args.start_epoch, args.epochs):
        pbar.set_description(f'Epoch {epoch}')
        decay_learning_rate(optimizer,
                            epoch,
                            args.init_lr,
                            args.lr_decay_epochs,
                            args.lr_decay_frac)

        train_stats = train_epoch(
            train_loader, model, epoch,
            mat_criterion,
            subst_criterion,
            color_criterion,
            optimizer,
            subst_mat_labels,
            loss_variances=loss_variances,
            loss_weights=loss_weights,
        )

        # evaluate on validation set
        val_stats = validate_epoch(
            validation_loader, model, epoch,
            mat_criterion,
            subst_criterion,
            color_criterion,
            subst_mat_labels,
            loss_variances=loss_variances,
            loss_weights=loss_weights,
        )

        # remember best prec@1 and save checkpoint
        is_best = val_stats['mat_prec1'] > best_prec1
        best_prec1 = max(val_stats['mat_prec1'], best_prec1)

        for stat_name, stat_val in train_stats.items():
            train_cum_stats[stat_name].append(stat_val)

        for stat_name, stat_val in val_stats.items():
            val_cum_stats[stat_name].append(stat_val)

        stats_dict = {
            'epoch': epoch + 1,
            'train': train_cum_stats,
            'validation': val_cum_stats,
        }

        if not args.dry_run:
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
                mat_criterion,
                subst_criterion,
                color_criterion,
                optimizer,
                subst_mat_labels,
                loss_variances,
                loss_weights):
    meter_dict = defaultdict(AverageValueMeter)
    pbar = tqdm(total=len(train_loader), desc='Training Epoch',
                dynamic_ncols=True)

    last_end_time = time.time()
    for batch_idx, batch_dict in enumerate(train_loader):
        meter_dict['data_time'].add(time.time() - last_end_time)

        input_var = batch_dict['image'].cuda()
        mat_labels = batch_dict['material_label'].cuda()
        subst_labels = batch_dict['substance_label'].cuda()

        batch_start_time = time.time()
        output = model.forward(input_var)

        mat_output = output['material']

        losses = {}

        if args.substance_loss is not None:
            if args.substance_loss == 'fc':
                subst_output = output['substance']
                subst_fc_prec1 = compute_precision(subst_output, subst_labels)
                meter_dict['subst_fc_prec1'].add(subst_fc_prec1)
            elif args.substance_loss == 'from_material':
                subst_output = material_to_substance_output(
                    mat_output, subst_mat_labels)
            else:
                raise ValueError('Invalid value for substance_loss')
            losses['substance'] = subst_criterion(subst_output, subst_labels)

        mat_subst_output = material_to_substance_output(
            mat_output, subst_mat_labels)

        mat_subst_prec1 = compute_precision(mat_subst_output, subst_labels)
        meter_dict['subst_from_mat_prec1'].add(mat_subst_prec1)

        if args.roughness_loss is not None:
            roughness_output = output['roughness']
            losses['roughness'], roughness_prec1 = compute_roughness_loss(
                batch_dict['roughness'], roughness_output, mat_criterion)
            meter_dict['roughness_prec1'].add(roughness_prec1[0])

        if args.color_loss is not None:
            color_output = output['color']
            color_target = batch_dict['color_hist'].cuda()
            if isinstance(color_criterion, nn.KLDivLoss):
                color_output = F.log_softmax(color_output)
            losses['color'] = color_criterion(color_output, color_target)
        else:
            color_output = None
            color_target = None

        losses['material'] = mat_criterion(mat_output, mat_labels)

        loss = combine_losses(losses, loss_weights, loss_variances,
                              args.use_variance)

        # Add losses to meters.
        meter_dict['loss'].add(loss.item())
        for loss_name, loss_tensor in losses.items():
            meter_dict[f'loss_{loss_name}'].add(loss_tensor.item())

        mat_prec1, mat_prec5 = compute_precision(
            mat_output, mat_labels, topk=(1, 5))
        meter_dict['mat_prec1'].add(mat_prec1)
        meter_dict['mat_prec5'].add(mat_prec5)
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
                color_output_vis = color_binner.visualize(
                    F.softmax(color_output[0].cpu().detach(), dim=0))
                color_target_vis = color_binner.visualize(color_target[0])
                color_hist_vis = np.vstack((color_output_vis, color_target_vis))
                vis.heatmap(color_hist_vis, win='train-color-hist',
                            opts=dict(title='Training Color Histogram'))
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
                   mat_criterion,
                   subst_criterion,
                   color_criterion,
                   subst_mat_labels,
                   loss_variances,
                   loss_weights):
    meter_dict = defaultdict(AverageValueMeter)

    # switch to evaluate mode
    model.eval()

    pbar = tqdm(total=len(val_loader), desc='Validating Epoch',
                dynamic_ncols=True)

    last_end_time = time.time()
    for batch_idx, batch_dict in enumerate(val_loader):
        input_var = batch_dict['image'].cuda()
        mat_labels = batch_dict['material_label'].cuda()
        subst_labels = batch_dict['substance_label'].cuda()

        output = model.forward(input_var)

        mat_output = output['material']

        losses = {}

        if args.substance_loss is not None:
            if args.substance_loss == 'fc':
                subst_output = output['substance']
                subst_fc_prec1 = compute_precision(subst_output, subst_labels)
                meter_dict['subst_fc_prec1'].add(subst_fc_prec1)
            elif args.substance_loss == 'from_material':
                subst_output = material_to_substance_output(
                    mat_output, subst_mat_labels)
            else:
                raise ValueError('Invalid value for substance_loss')
            losses['substance'] = subst_criterion(subst_output, subst_labels)

        mat_subst_output = material_to_substance_output(
            mat_output, subst_mat_labels)

        subst_prec1 = compute_precision(mat_subst_output, subst_labels)
        meter_dict['subst_from_mat_prec1'].add(subst_prec1)

        if args.roughness_loss:
            roughness_output = output['roughness']
            losses['roughness'], roughness_prec1 = compute_roughness_loss(
                batch_dict['roughness'], roughness_output, mat_criterion)
            meter_dict['roughness_prec1'].add(roughness_prec1)

        if args.color_loss is not None:
            color_output = output['color']
            color_target = batch_dict['color_hist'].cuda()
            if isinstance(color_criterion, nn.KLDivLoss):
                color_output = F.log_softmax(color_output)
            losses['color'] = color_criterion(color_output, color_target)
        else:
            color_output = None
            color_target = None

        losses['material'] = mat_criterion(mat_output, mat_labels)
        loss = combine_losses(losses, loss_weights, loss_variances,
                              args.use_variance)

        # Add losses to meters.
        meter_dict['loss'].add(loss.item())
        for loss_name, loss_tensor in losses.items():
            meter_dict[f'loss_{loss_name}'].add(loss_tensor.item())

        # measure accuracy and record loss
        mat_prec1, mat_prec5 = compute_precision(
            mat_output, mat_labels, topk=(1, 5))
        meter_dict['mat_prec1'].add(mat_prec1)
        meter_dict['mat_prec5'].add(mat_prec5)
        meter_dict['batch_time'].add(time.time() - last_end_time)
        last_end_time = time.time()
        pbar.update()

        if batch_idx % args.show_freq == 0:
            vis.image(visualize_input(batch_dict),
                      win='validation-batch-example',
                      opts=dict(title='Validation Batch Example'))
            if 'color' in losses:
                color_output_vis = color_binner.visualize(
                    F.softmax(color_output[0].cpu().detach(), dim=0))
                color_target_vis = color_binner.visualize(color_target[0])
                color_hist_vis = np.vstack((color_output_vis, color_target_vis))
                vis.heatmap(color_hist_vis, win='validation-color-hist',
                            opts=dict(title='Validation Color Histogram'))
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


if __name__ == '__main__':
    main()

