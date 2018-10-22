from pathlib import Path

import collections
import ujson as json
import click
import torch
from torch.utils.data import DataLoader
from torchnet.logger import VisdomLogger
from torchvision.models import resnet
import torchnet as tnt
from tqdm import tqdm

from terial import models
from terial.classifier import transforms, rendering_dataset
from terial.config import SUBSTANCES
from terial.classifier.network import RendNet3
from terial.database import session_scope

INPUT_SIZE = 224


@click.command()
@click.option('--checkpoint-path', required=True, type=click.Path(exists=True))
@click.option('--batch-size', type=int, default=64)
@click.option('--normalized', is_flag=True)
@click.option('--visdom-port', default=8097)
def main(checkpoint_path, batch_size, normalized,
         visdom_port):

    checkpoint_path = Path(checkpoint_path)
    snapshot_path = checkpoint_path.parent.parent.parent / 'snapshot.json'

    with snapshot_path.open('r') as f:
        snapshot_dict = json.load(f)
        mat_id_to_label = snapshot_dict['mat_id_to_label']
        label_to_mat_id = {int(v): int(k) for k, v in mat_id_to_label.items()}
        num_classes = len(label_to_mat_id) + 1

    print(f'Loading model checkpoint from {checkpoint_path!r}')
    checkpoint = torch.load(checkpoint_path)

    model = RendNet3(num_classes=num_classes,
                 num_roughness_classes=20,
                 num_substances=len(SUBSTANCES),
                 base_model=resnet.resnet18(pretrained=False))
    model.load_state_dict(checkpoint['state_dict'])
    model.train(False)
    model = model.cuda()

    validation_dataset = rendering_dataset.MaterialRendDataset(
        snapshot_dict,
        snapshot_dict['examples']['validation'],
        shape=(384, 384),
        image_transform=transforms.inference_image_transform(INPUT_SIZE),
        mask_transform=transforms.inference_mask_transform(INPUT_SIZE))

    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        collate_fn=rendering_dataset.collate_fn)

    pred_counts = collections.defaultdict(collections.Counter)

    # switch to evaluate mode
    model.eval()

    confusion_meter = tnt.meter.ConfusionMeter(
        k=num_classes, normalized=normalized)

    pbar = tqdm(validation_loader)
    for batch_idx, batch_dict in enumerate(pbar):
        input_tensor = batch_dict['image'].cuda()
        labels = batch_dict['material_label'].cuda()

        # compute output
        output = model.forward(input_tensor)
        pbar.set_description(f"{output['material'].size()}")

        # _, pred = output['material'].topk(k=1, dim=1, largest=True, sorted=True)

        confusion_meter.add(output['material'].cpu(), labels.cpu())

    with session_scope() as sess:
        materials = sess.query(models.Material).filter_by(enabled=True).all()
        material_id_to_name = {m.id: m.name for m in materials}
        mat_by_id = {m.id: m for m in materials}

    class_names = ['background']
    class_names.extend([
        mat_by_id[label_to_mat_id[i]].name for i in range(1, num_classes)
    ])

    print(len(class_names), )

    confusion_matrix = confusion_meter.value()
    # sorted_confusion_matrix = confusion_matrix[:, inds]
    # sorted_confusion_matrix = sorted_confusion_matrix[inds, :]

    # sorted_class_names = [class_names[i] for i in inds]
    confusion_logger = VisdomLogger(
        'heatmap', opts={
            'title': 'Confusion matrix',
            'columnnames': class_names,
            'rownames': class_names,
            'xtickfont': {'size': 8},
            'ytickfont': {'size': 8},
        },
        env='brdf-classifier-confusion',
    port=visdom_port)

    confusion_logger.log(confusion_matrix)




if __name__ == '__main__':
    main()
