#!/usr/bin/env python3
"""
Visualize CROMA embeddings (optical, radar, and joint) from downloaded tiles
using FiftyOne for interactive exploration.

GOAL: We want to see if the frozen (not trained) CROMA encoder produces
embeddings that separate "before deforestation" (bef) vs "event" (evt) tiles.
We extract 768-d feature vectors from the pretrained model and project them
to 2D with UMAP so we can visually inspect clustering.

Creates a FiftyOne dataset with RGB previews of each tile, attaches CROMA
embeddings, and computes UMAP visualizations for optical, radar, and joint modes.

Usage:
  python visualize_croma_embeddings.py                         # All FIDs, all modes
  python visualize_croma_embeddings.py --fids 193 194 195      # Specific FIDs
  python visualize_croma_embeddings.py --mode optical           # Optical only
  python visualize_croma_embeddings.py --mode radar             # Radar only
  python visualize_croma_embeddings.py --mode joint             # Joint (multimodal)
  python visualize_croma_embeddings.py --mode compare           # All three
  python visualize_croma_embeddings.py --no-launch              # Don't open browser
"""

import os
import glob
import argparse
import numpy as np
import torch
import rasterio            # library for reading GeoTIFF satellite images
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
# torchgeo wraps the CROMA model so we can load pretrained weights easily
from torchgeo.models import croma_base, CROMABase_Weights
import fiftyone as fo      # interactive dataset visualization tool
import fiftyone.brain as fob  # fiftyone's ML brain module (computes UMAP, etc.)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Where our 120x120 tiles live (12-band S2 and 2-band S1, in CROMA format)
TILES_BASE = os.path.expanduser('~/Desktop/Master/thesis/tiles/croma')
# Where we save RGB .png previews (FiftyOne needs regular images, not GeoTIFFs)
RGB_DIR = os.path.expanduser('~/Desktop/Master/thesis/tiles/rgb_previews')


# ──────────────────────────────────────────────────────────────────────────────
# Normalization (same as CROMA training: per-channel mean ± 2*std)
# ──────────────────────────────────────────────────────────────────────────────

def normalize(x):
    """Per-channel robust normalization to [0, 1].

    This is the same normalization used during CROMA pretraining (SatMAE style):
    for each spectral band, compute mean and std across the batch, then clip to
    [mean - 2*std, mean + 2*std] and rescale to [0, 1].

    This removes extreme outliers (very bright/dark pixels) while preserving
    the relative spectral information between bands.

    Args:
        x: tensor of shape (batch_size, num_channels, H, W)
    Returns:
        normalized tensor of same shape, values in [0, 1]
    """
    x = x.float()
    imgs = []
    for ch in range(x.shape[1]):           # loop over each spectral band
        channel = x[:, ch, :, :]           # shape: (batch_size, H, W)
        min_val = channel.mean() - 2 * channel.std()   # lower clip boundary
        max_val = channel.mean() + 2 * channel.std()   # upper clip boundary
        # Rescale so min_val -> 0 and max_val -> 1
        img = (channel - min_val) / (max_val - min_val + 1e-10)
        img = torch.clamp(img, 0, 1)      # clip anything outside [0, 1]
        imgs.append(img.unsqueeze(1))      # restore channel dimension
    return torch.cat(imgs, dim=1)          # reassemble all channels


# ──────────────────────────────────────────────────────────────────────────────
# RGB preview generation
# ──────────────────────────────────────────────────────────────────────────────

def make_s2_rgb(s2_path, output_path):
    """Create a human-viewable RGB PNG from a 12-band S2 tile.

    FiftyOne can't display 12-band GeoTIFFs, so we extract the 3 visible bands
    (B4=Red, B3=Green, B2=Blue) and save as a standard PNG image.

    The same mean±2*std normalization is used to stretch pixel values to 0-255.
    """
    if os.path.exists(output_path):
        return output_path
    with rasterio.open(s2_path) as ds:
        data = ds.read().astype(np.float32)
    # CROMA band order: B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12
    # Pick indices 3,2,1 = B4(Red), B3(Green), B2(Blue)
    rgb = data[[3, 2, 1], :, :]  # shape: (3, 120, 120)
    for ch in range(3):
        vmin = rgb[ch].mean() - 2 * rgb[ch].std()
        vmax = rgb[ch].mean() + 2 * rgb[ch].std()
        rgb[ch] = np.clip((rgb[ch] - vmin) / (vmax - vmin + 1e-10) * 255, 0, 255)
    rgb = rgb.astype(np.uint8)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # matplotlib expects (H, W, 3) so we transpose from (3, H, W)
    plt.imsave(output_path, np.transpose(rgb, (1, 2, 0)))
    return output_path


def make_s1_rgb(s1_path, output_path):
    """Create a false-color PNG from a 2-band S1 (SAR) tile.

    SAR images only have 2 channels (VV and VH polarizations), so we can't
    make a true RGB. Instead we create a false-color composite:
      Red   = VV polarization
      Green = VH polarization
      Blue  = VV - VH (dB difference, highlights scattering properties)
    """
    if os.path.exists(output_path):
        return output_path
    real_path = os.path.realpath(s1_path)
    with rasterio.open(real_path) as ds:
        data = ds.read().astype(np.float32)
    vv, vh = data[0], data[1]
    ratio = vv - vh  # dB difference between polarizations
    rgb = np.stack([vv, vh, ratio])  # shape: (3, 120, 120)
    for ch in range(3):
        vmin = rgb[ch].mean() - 2 * rgb[ch].std()
        vmax = rgb[ch].mean() + 2 * rgb[ch].std()
        rgb[ch] = np.clip((rgb[ch] - vmin) / (vmax - vmin + 1e-10) * 255, 0, 255)
    rgb = rgb.astype(np.uint8)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.imsave(output_path, np.transpose(rgb, (1, 2, 0)))
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_tile(path):
    """Load a GeoTIFF tile as a numpy array of shape (C, H, W).
    C = number of spectral bands (12 for S2, 2 for S1).
    """
    real_path = os.path.realpath(path)  # resolve symlinks if any
    with rasterio.open(real_path) as ds:
        data = ds.read()  # rasterio returns (bands, height, width)
    return data.astype(np.float32)


def parse_date_from_image_id(image_id, sensor):
    """Extract acquisition date from the image folder name.

    S2 image IDs start with the date: e.g. "20200615T..."  -> first 8 chars
    S1 image IDs have the date at position 17: e.g. "S1A_IW_GRDH_1SDV_20200615..."
    """
    if sensor == 's2':
        return datetime.strptime(image_id[:8], '%Y%m%d')
    else:
        return datetime.strptime(image_id[17:25], '%Y%m%d')


def find_paired_tiles(fid_dir, max_day_gap=3):
    """Find S2 and S1 tiles that can be paired by acquisition date.

    For each deforestation site (FID), we have S2 and S1 tiles from different dates.
    This function matches S2 and S1 tiles that were acquired within `max_day_gap` days
    of each other, so they represent roughly the same ground conditions.

    Tiles that can't be paired (no matching S1 for an S2, or vice versa) are still
    kept as unpaired records (with s1_path=None or s2_path=None).
    """
    fid = os.path.basename(fid_dir)
    records = []

    # Collect all tiles organized by sensor type
    # Key = (evt_bef, tile_idx) so we can match S1 and S2 tiles at the same
    # spatial location and deforestation status
    s2_tiles = {}
    s1_tiles = {}

    # Walk through all 4 subdirectories: s2_evt, s2_bef, s1_evt, s1_bef
    for category in ['s2_evt', 's2_bef', 's1_evt', 's1_bef']:
        cat_dir = os.path.join(fid_dir, category)
        if not os.path.isdir(cat_dir):
            continue
        sensor = 's2' if 's2' in category else 's1'
        evt_bef = 'evt' if 'evt' in category else 'bef'

        # Each image_dir is named by the acquisition ID (contains the date)
        for image_dir in sorted(glob.glob(os.path.join(cat_dir, '*'))):
            image_id = os.path.basename(image_dir)
            try:
                date = parse_date_from_image_id(image_id, sensor)
            except ValueError:
                continue
            # Each .tif inside is one spatial tile from that acquisition
            for tile_path in sorted(glob.glob(os.path.join(image_dir, '*.tif'))):
                tile_idx = os.path.basename(tile_path).replace('.tif', '')
                key = (evt_bef, tile_idx)
                entry = (image_id, date, tile_path)
                if sensor == 's2':
                    s2_tiles.setdefault(key, []).append(entry)
                else:
                    s1_tiles.setdefault(key, []).append(entry)

    # --- Pairing: match each S2 tile to the closest-in-time S1 tile ---
    paired_s2 = set()  # track which S2 tiles have been paired
    paired_s1 = set()  # track which S1 tiles have been paired

    all_keys = set(list(s2_tiles.keys()) + list(s1_tiles.keys()))
    for key in sorted(all_keys):
        evt_bef, tile_idx = key
        s2_list = sorted(s2_tiles.get(key, []), key=lambda x: x[1])
        s1_list = sorted(s1_tiles.get(key, []), key=lambda x: x[1])

        # For each S2 tile, find the S1 tile with the smallest time gap
        for s2_id, s2_date, s2_path in s2_list:
            best_s1 = None
            best_gap = timedelta(days=max_day_gap + 1)
            for s1_id, s1_date, s1_path in s1_list:
                if (evt_bef, tile_idx, s1_id) in paired_s1:
                    continue  # already used by another S2 tile
                gap = abs(s2_date - s1_date)
                if gap <= timedelta(days=max_day_gap) and gap < best_gap:
                    best_s1 = (s1_id, s1_path)
                    best_gap = gap
            if best_s1:
                paired_s2.add((evt_bef, tile_idx, s2_id))
                paired_s1.add((evt_bef, tile_idx, best_s1[0]))
                records.append({
                    'fid': fid, 'evt_bef': evt_bef, 'tile_idx': tile_idx,
                    's2_path': s2_path, 's1_path': best_s1[1],
                    's2_id': s2_id, 's1_id': best_s1[0],
                })

    # --- Add unpaired S2 tiles (no matching S1 within max_day_gap) ---
    for key in sorted(s2_tiles.keys()):
        evt_bef, tile_idx = key
        for s2_id, _, s2_path in s2_tiles[key]:
            if (evt_bef, tile_idx, s2_id) not in paired_s2:
                records.append({
                    'fid': fid, 'evt_bef': evt_bef, 'tile_idx': tile_idx,
                    's2_path': s2_path, 's1_path': None,
                    's2_id': s2_id, 's1_id': None,
                })

    # --- Add unpaired S1 tiles (no matching S2 within max_day_gap) ---
    for key in sorted(s1_tiles.keys()):
        evt_bef, tile_idx = key
        for s1_id, _, s1_path in s1_tiles[key]:
            if (evt_bef, tile_idx, s1_id) not in paired_s1:
                records.append({
                    'fid': fid, 'evt_bef': evt_bef, 'tile_idx': tile_idx,
                    's2_path': None, 's1_path': s1_path,
                    's2_id': None, 's1_id': s1_id,
                })

    return records


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_embeddings(records, device, batch_size=32):
    """Extract optical, radar, and joint embeddings from tile records.

    This is the FROZEN feature extraction step: we load pretrained CROMA weights
    and pass each tile through the encoder WITHOUT updating any weights
    (torch.no_grad). The output is a 768-d vector per tile (the "GAP" = Global
    Average Pooling of all patch encodings).

    We load 3 separate models because torchgeo's CROMA requires specifying
    modalities at init time:
      - optical-only model  -> produces optical_GAP  (768-d per S2 tile)
      - SAR-only model      -> produces sar_GAP      (768-d per S1 tile)
      - joint model         -> produces joint_GAP    (768-d per S1+S2 pair)
    """
    # Pre-allocate: one embedding slot per record, None if that modality is missing
    optical_embs = [None] * len(records)
    radar_embs = [None] * len(records)
    joint_embs = [None] * len(records)

    # Load all tiles into memory first (they're small: 120x120 px)
    s2_tensors = []
    s1_tensors = []
    for rec in records:
        s2_tensors.append(load_tile(rec['s2_path']) if rec['s2_path'] else None)
        s1_tensors.append(load_tile(rec['s1_path']) if rec['s1_path'] else None)

    n = len(records)

    # ---- Optical-only model ----
    # Encodes 12-band S2 tiles into 768-d vectors
    has_s2 = any(t is not None for t in s2_tensors)
    if has_s2:
        print("  Loading optical-only model...")
        model_opt = croma_base(weights=CROMABase_Weights.CROMA_VIT, modalities=['optical'])
        model_opt = model_opt.to(device)
        # attn_bias = the 2D-ALiBi positional bias matrix, must be on same device
        if hasattr(model_opt, 'attn_bias') and model_opt.attn_bias is not None:
            model_opt.attn_bias = model_opt.attn_bias.to(device)
        model_opt.eval()  # set to evaluation mode (disables dropout)

        # Process tiles in batches for memory efficiency
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            # Only include tiles that have S2 data
            batch = [(i, torch.from_numpy(s2_tensors[i]))
                     for i in range(start, end) if s2_tensors[i] is not None]
            if batch:
                indices, tensors = zip(*batch)
                x_opt = normalize(torch.stack(tensors).to(device))
                with torch.no_grad():  # FROZEN: no gradient computation
                    out = model_opt(x_optical=x_opt)
                # optical_GAP = Global Average Pool of all 225 patch encodings -> (batch, 768)
                for idx, emb in zip(indices, out['optical_GAP'].cpu().numpy()):
                    optical_embs[idx] = emb
        del model_opt  # free GPU memory before loading next model

    # ---- SAR-only model ----
    # Encodes 2-band S1 (VV+VH) tiles into 768-d vectors
    has_s1 = any(t is not None for t in s1_tensors)
    if has_s1:
        print("  Loading SAR-only model...")
        model_sar = croma_base(weights=CROMABase_Weights.CROMA_VIT, modalities=['sar'])
        model_sar = model_sar.to(device)
        if hasattr(model_sar, 'attn_bias') and model_sar.attn_bias is not None:
            model_sar.attn_bias = model_sar.attn_bias.to(device)
        model_sar.eval()

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = [(i, torch.from_numpy(s1_tensors[i]))
                     for i in range(start, end) if s1_tensors[i] is not None]
            if batch:
                indices, tensors = zip(*batch)
                x_sar = normalize(torch.stack(tensors).to(device))
                with torch.no_grad():
                    out = model_sar(x_sar=x_sar)
                for idx, emb in zip(indices, out['sar_GAP'].cpu().numpy()):
                    radar_embs[idx] = emb
        del model_sar

    # ---- Joint model (multimodal: S1 + S2 together) ----
    # Uses the cross-attention encoder to fuse both modalities into a single 768-d vector.
    # Only works for tiles where we have BOTH S1 and S2 (i.e., paired tiles).
    has_pairs = any(s2_tensors[i] is not None and s1_tensors[i] is not None
                    for i in range(n))
    if has_pairs:
        print("  Loading joint model...")
        model_joint = croma_base(weights=CROMABase_Weights.CROMA_VIT,
                                 modalities=['optical', 'sar'])
        model_joint = model_joint.to(device)
        if hasattr(model_joint, 'attn_bias') and model_joint.attn_bias is not None:
            model_joint.attn_bias = model_joint.attn_bias.to(device)
        model_joint.eval()

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            # Only tiles that have BOTH S2 and S1
            batch = [(i, s2_tensors[i], s1_tensors[i])
                     for i in range(start, end)
                     if s2_tensors[i] is not None and s1_tensors[i] is not None]
            if batch:
                indices, s2s, s1s = zip(*batch)
                x_opt = normalize(torch.stack(
                    [torch.from_numpy(s) for s in s2s]).to(device))
                x_sar = normalize(torch.stack(
                    [torch.from_numpy(s) for s in s1s]).to(device))
                with torch.no_grad():
                    out = model_joint(x_sar=x_sar, x_optical=x_opt)
                # joint_GAP = fused representation from cross-attention encoder
                for idx, emb in zip(indices, out['joint_GAP'].cpu().numpy()):
                    joint_embs[idx] = emb
        del model_joint

    return optical_embs, radar_embs, joint_embs


# ──────────────────────────────────────────────────────────────────────────────
# FiftyOne dataset creation
# ──────────────────────────────────────────────────────────────────────────────

def build_fiftyone_dataset(records, optical_embs, radar_embs, joint_embs,
                           requested_modes, dataset_name="croma_deforestation"):
    """Build a FiftyOne dataset for interactive visualization.

    FiftyOne lets you browse tiles visually and see their UMAP projections.
    Each tile gets: an RGB preview image, metadata (fid, evt/bef, dates),
    and the 768-d CROMA embeddings. Then UMAP projects the 768-d vectors
    to 2D so we can see if evt/bef tiles cluster separately.
    """
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name, persistent=True)

    print("Generating preview images...")
    samples = []
    for i, rec in enumerate(records):
        # Choose preview: S2 RGB if available, else S1 false-color
        if rec['s2_path']:
            tile_name = os.path.basename(rec['s2_path']).replace('.tif', '.png')
            sensor_dir = 'optical'
            img_id = rec['s2_id']
            preview_path = os.path.join(RGB_DIR, rec['fid'], rec['evt_bef'],
                                        sensor_dir, img_id, tile_name)
            make_s2_rgb(rec['s2_path'], preview_path)
        elif rec['s1_path']:
            tile_name = os.path.basename(rec['s1_path']).replace('.tif', '.png')
            sensor_dir = 'radar'
            img_id = rec['s1_id']
            preview_path = os.path.join(RGB_DIR, rec['fid'], rec['evt_bef'],
                                        sensor_dir, img_id, tile_name)
            make_s1_rgb(rec['s1_path'], preview_path)
        else:
            continue

        sample = fo.Sample(filepath=preview_path)
        sample["fid"] = rec['fid']
        sample["evt_bef"] = rec['evt_bef']
        sample["tile_idx"] = rec['tile_idx']
        sample["has_optical"] = rec['s2_path'] is not None
        sample["has_radar"] = rec['s1_path'] is not None
        sample["has_pair"] = rec['s2_path'] is not None and rec['s1_path'] is not None
        sample["sensor"] = ("paired" if sample["has_pair"]
                            else "optical" if sample["has_optical"]
                            else "radar")

        if rec['s2_id']:
            sample["s2_date"] = rec['s2_id'][:8]
        if rec['s1_id']:
            sample["s1_date"] = rec['s1_id'][17:25]

        # Store embeddings as sample fields
        if optical_embs[i] is not None:
            sample["optical_embedding"] = optical_embs[i].tolist()
        if radar_embs[i] is not None:
            sample["radar_embedding"] = radar_embs[i].tolist()
        if joint_embs[i] is not None:
            sample["joint_embedding"] = joint_embs[i].tolist()

        samples.append(sample)

    dataset.add_samples(samples)
    print(f"Dataset created: {len(dataset)} samples")

    # --- Compute UMAP for each embedding type ---
    # UMAP projects the 768-d embeddings to 2D for visualization.
    # If evt and bef tiles form separate clusters in 2D, it means the CROMA
    # encoder captures differences between them. If they overlap, it doesn't.
    for mode in requested_modes:
        field = f"{mode}_embedding"       # e.g. "optical_embedding"
        brain_key = f"croma_{mode}_umap"  # name for the UMAP result in FiftyOne

        # Only use samples that have this embedding type
        view = dataset.exists(field)
        if len(view) < 5:
            print(f"  Skipping {mode} UMAP: only {len(view)} samples")
            continue

        emb_list = [s[field] for s in view]
        emb_matrix = np.array(emb_list)  # shape: (num_samples, 768)
        print(f"  Computing {mode} UMAP ({emb_matrix.shape[0]} samples)...")

        # This stores the 2D coordinates in FiftyOne so we can visualize them
        fob.compute_visualization(
            view,
            embeddings=emb_matrix,
            brain_key=brain_key,
            method="umap",
            seed=42,
        )
        print(f"  -> brain_key: '{brain_key}'")

    return dataset


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # --- Parse command-line arguments ---
    parser = argparse.ArgumentParser(
        description='Visualize CROMA embeddings with FiftyOne')
    parser.add_argument('--fids', nargs='+', type=str, default=None,
                        help='Specific FIDs to process (e.g., 193 194)')
    parser.add_argument('--mode', choices=['optical', 'radar', 'joint', 'compare'],
                        default='compare',
                        help='Which embeddings to compute (compare = all three)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--dataset-name', type=str, default='croma_deforestation')
    parser.add_argument('--no-launch', action='store_true',
                        help='Do not launch FiftyOne app')
    args = parser.parse_args()

    # --- Choose compute device: GPU > Apple Silicon > CPU ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')   # Apple M1/M2/M3
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Which embedding types to compute
    if args.mode == 'compare':
        requested_modes = ['optical', 'radar', 'joint']
    else:
        requested_modes = [args.mode]

    # --- Find all deforestation site folders (fid_*) ---
    fid_dirs = sorted(glob.glob(os.path.join(TILES_BASE, 'fid_*')))
    if args.fids:
        fid_set = {f'fid_{f}' for f in args.fids}
        fid_dirs = [d for d in fid_dirs if os.path.basename(d) in fid_set]
    print(f"Processing {len(fid_dirs)} FIDs...")

    # --- Build list of all tiles with S1/S2 pairing ---
    all_records = []
    for fid_dir in fid_dirs:
        all_records.extend(find_paired_tiles(fid_dir))
    print(f"Found {len(all_records)} tile records")

    if not all_records:
        print("No tiles found.")
        return

    # --- Extract frozen CROMA embeddings (768-d per tile) ---
    print("Extracting embeddings...")
    optical_embs, radar_embs, joint_embs = extract_embeddings(
        all_records, device, args.batch_size)

    # --- Build FiftyOne dataset with UMAP projections ---
    print("Building FiftyOne dataset...")
    dataset = build_fiftyone_dataset(
        all_records, optical_embs, radar_embs, joint_embs,
        requested_modes, args.dataset_name)

    # --- Launch interactive browser app ---
    if not args.no_launch:
        print("\nLaunching FiftyOne app...")
        print("In the app: click '+' -> 'Embeddings' -> select a brain_key")
        print("Use 'Color by' to switch between evt_bef, fid, sensor")
        print("Lasso-select points to filter the grid below, then click thumbnails")
        session = fo.launch_app(dataset)
        input("Press Enter to close the app...")


if __name__ == '__main__':
    main()
