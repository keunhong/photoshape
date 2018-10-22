import argparse

from terial import models
from terial import config
from terial.database import session_scope
from tqdm import tqdm
from collections import defaultdict
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--thres', default=0.1)
args = parser.parse_args()



def main():
    with session_scope() as sess:
        exemplars = sess.query(models.Exemplar).order_by(
            models.Exemplar.id).filter_by(exclude=False).all()

    print(f"Fetched {len(exemplars)} exemplars. Loading features...")

    feats = []
    for e1 in tqdm(exemplars):
        f1 = e1.load_data(config.EXEMPLAR_ALIGN_DATA_NAME)
        feats.append(f1)

    feats = torch.tensor(feats).squeeze()
    if args.cuda:
        feats = feats.cuda()

    print(f"Computed {feats.size(0)} features each with {feats.size(1)} dims.")
    print(f"Computing duplicates...")
    duplicates = defaultdict(list)
    duplicate_dists = defaultdict(list)

    for i, f in enumerate(tqdm(feats)):
        dists = (feats - f.unsqueeze(0).expand(*feats.size())).pow(2).sum(dim=1)
        dists[i] = 99999
        duplicate_inds = torch.nonzero(dists < 0.1)
        if len(duplicate_inds) > 0:
            duplicates[i].extend(duplicate_inds[:, 0].tolist())
            duplicate_dists[i].extend(dists[duplicate_inds][:, 0].tolist())

    print(f"Found duplicates from {len(duplicates)} exemplars.")
    yy = input("Commit, this cannot be undone (easily)? (y/n) ")
    if yy != 'y':
        return

    with session_scope() as sess:
        for i, dup_list in duplicates.items():
            e1 = exemplars[i]

            # Refetch from DB and check if it's already excluded.
            # If th
            e1 = sess.query(models.Exemplar).get(e1.id)
            if e1.exclude:
                continue

            for j in dup_list:
                e2 = exemplars[j]
                print(f'Set exemplar {e2.id}.exclude = True')
                sess.query(models.Exemplar).get(e2.id).exclude = True
                sess.commit()
                # print(e1.id, e2.id)


if __name__ == '__main__':
    main()
