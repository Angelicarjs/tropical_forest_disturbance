"""
Train the U-Net decoder for disturbance-type segmentation on frozen CROMA embeddings.

Key choices:
  - Split is done by FID (stratified by class), not by tile, to avoid spatial leakage.ß
  - Weighted cross-entropy: background and "Clear-cut bare soil" dominate, so
    rare classes are up-weighted by inverse pixel frequency.
  - Metric: per-class IoU + mean IoU (mIoU), so we see how each type does.
"""

import os
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from seg_dataset import DisturbanceSegDataset, NUM_CLASSES, CLASS_TO_ID
from seg_model import SegDecoder

ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
ID_TO_CLASS[0] = "background"

#split FIDs into train/val, stratified by the FID's disturbance class
def split_fids(ds, val_frac=0.2, seed=42):
    """Split FIDs into train/val, stratified by the FID's disturbance class."""
    # dictionary of FID with their class
    fid_cls = {}

    for s in ds.samples:
        # if the FID is not in the dictionary
        if s["fid"] not in fid_cls:
            fid_cls[s["fid"]] = ds.fid_polys[s["fid"]][0][1]  # primary class id
    
    print(fid_cls)
    # group FIDs by class        
    cls_fids = defaultdict(list)
    for fid, c in fid_cls.items():
        cls_fids[c].append(fid)
    print(cls_fids)

    rng = random.Random(seed)
    train_fids, val_fids = set(), set()
    for c, fids in cls_fids.items():
        fids = sorted(fids)
        rng.shuffle(fids)
        n_val = max(1, round(len(fids) * val_frac)) if len(fids) > 1 else 0
        val_fids.update(fids[:n_val])
        train_fids.update(fids[n_val:])
    print(f"Train FIDs: {len(train_fids)}, Val FIDs: {len(val_fids)}")
    return train_fids, val_fids

# return indices of samples in ds whose FID is in the given set of FIDs
def subset_indices(ds, fids):
    return [i for i, s in enumerate(ds.samples) if s["fid"] in fids]

#compute the pixeles in each class in the train set and give weights inversely proportional to those frequencies, so the model does not ignore rare classes
def compute_class_weights(ds, indices, num_classes, cap=400):
    """Inverse pixel-frequency weights, estimated over up to `cap` train masks."""
    rng = random.Random(0)
    idx = list(indices)
    rng.shuffle(idx)
    counts = np.zeros(num_classes, dtype=np.float64)
    for i in idx[:cap]:
        _, y = ds[i]
        counts += np.bincount(y.numpy().ravel(), minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts)   # high weight for rare classes
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    """Return per-class IoU over the loader using a confusion matrix."""
    model.eval()
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu().numpy()   # (B,120,120)
        t = y.numpy()
        k = (t >= 0) & (t < num_classes)
        conf += np.bincount(num_classes * t[k].astype(int) + pred[k],
                            minlength=num_classes ** 2).reshape(num_classes, num_classes)
    ious = []
    for c in range(num_classes):
        tp = conf[c, c]
        denom = conf[:, c].sum() + conf[c, :].sum() - tp
        ious.append(tp / denom if denom > 0 else float("nan"))
    return ious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings-root", default="embeddings")
    ap.add_argument("--tiles-root", default=os.path.expanduser("~/thesis_tiles_120px"))
    ap.add_argument("--shp", default="data_shp/label_polygons.shp")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default="seg_model_best.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    ds = DisturbanceSegDataset(args.embeddings_root, args.tiles_root, args.shp)
    print(f"[data] {len(ds)} tiles (evt+aft)")

    train_fids, val_fids = split_fids(ds)
    tr_idx = subset_indices(ds, train_fids)
    va_idx = subset_indices(ds, val_fids)
    print(f"[split] train: {len(train_fids)} FIDs / {len(tr_idx)} tiles | "
          f"val: {len(val_fids)} FIDs / {len(va_idx)} tiles")

    train_loader = DataLoader(Subset(ds, tr_idx), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(Subset(ds, va_idx), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    print("[weights] computing class weights...")
    w = compute_class_weights(ds, tr_idx, NUM_CLASSES).to(device)
    print("   " + "  ".join(f"{ID_TO_CLASS[c]}={float(w[c]):.1f}" for c in range(NUM_CLASSES)))

    model = SegDecoder(in_dim=768, num_classes=NUM_CLASSES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=w)

    best_miou = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)     # logits (B,7,120,120), target (B,120,120)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        train_loss = running / max(len(tr_idx), 1)

        ious = evaluate(model, val_loader, device, NUM_CLASSES)
        miou = float(np.nanmean(ious))
        print(f"epoch {epoch:02d} | loss {train_loss:.3f} | mIoU {miou:.3f}", flush=True)
        if epoch % 5 == 0 or epoch == args.epochs:
            for c in range(NUM_CLASSES):
                print(f"    IoU {ID_TO_CLASS[c]:<30} {ious[c]:.3f}")
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), args.out)

    print(f"[done] best mIoU {best_miou:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
