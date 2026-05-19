"""
Generate CROMA embeddings for one image_id (one acquisition).

Usage (CLI):
    python make_embeddings.py <image_dir>

    where <image_dir> is a folder of the form:
        .../tiles_120px_10/tiles_120px/<product>/fid_<fid>/<window>/<image_id>/
    containing tile_*.tif files.

Usage (notebook):
    from make_embeddings import embed_image, load_model, load_or_compute_stats

    model = load_model("optical")
    stats = load_or_compute_stats(tiles_root, "s2_l2a")
    embed_image(image_dir, model=model, stats=stats)

Output: one .npy per tile (shape (15, 15, 768)) under
    <project_root>/embeddings/<product>/fid_<fid>/<window>/<image_id>/tile_N.npy
"""

import argparse
import glob
import re
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
import torch
from torchgeo.models import croma_base, CROMABase_Weights


# -------- Constants --------
PATCH_SIZE = 8
EMBED_DIM = 768
TILE_SIZE = 120
GRID_SIZE = TILE_SIZE // PATCH_SIZE  # 15

S2_BAND_INDICES = list(range(12))
S1_BAND_INDICES = [0, 1]

PRODUCT_CONFIG = {
    "s2_l2a": {"modality": "optical", "bands": S2_BAND_INDICES, "min_valid": 0,    "max_valid": None},
    "s1_grd": {"modality": "sar",     "bands": S1_BAND_INDICES, "min_valid": None, "max_valid": None},
}

SCRIPT_DIR = Path(__file__).resolve().parent
EMBEDDINGS_ROOT = SCRIPT_DIR / "embeddings"


# -------- Device --------
def get_device():
    # MPS is intentionally skipped: torchgeo's CROMA leaves some internal
    # buffers (e.g. relative_position_bias) on CPU after .to(device), which
    # crashes the forward pass on Apple Silicon. Use CUDA if available, else CPU.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -------- Path parsing --------
def parse_image_dir(image_dir):
    """
    image_dir: .../tiles_120px_10/tiles_120px/<product>/fid_<fid>/<window>/<image_id>/
    Returns dict: product, fid, window, image_id, product_root, tiles_root.
    """
    p = Path(image_dir).resolve()
    image_id = p.name
    window = p.parent.name
    fid_part = p.parent.parent.name
    if not fid_part.startswith("fid_"):
        raise ValueError(f"Expected '.../fid_<N>/<window>/<image_id>', got: {image_dir}")
    fid = fid_part[len("fid_"):]
    product = p.parent.parent.parent.name
    if product not in PRODUCT_CONFIG:
        raise ValueError(f"Unknown product '{product}' (expected one of {list(PRODUCT_CONFIG)})")
    product_root = p.parent.parent.parent       # .../tiles_120px/<product>
    tiles_root = product_root.parent            # .../tiles_120px
    return {
        "product": product,
        "fid": fid,
        "window": window,
        "image_id": image_id,
        "product_root": product_root,
        "tiles_root": tiles_root,
    }


# -------- Tile IO  --------
def load_tile(path, band_indices=None):
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
    if band_indices is not None:
        data = data[band_indices]
    return data


# -------- Normalization --------
def normalize(x, mean, std):
    """CROMA recipe: clip to mean ± 2*std per channel, rescale to [0, 1].
    x: (N, C, H, W) tensor."""
    x = x.float()
    mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    std = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    min_val = mean - 2 * std
    max_val = mean + 2 * std
    x = (x - min_val) / (max_val - min_val)
    return torch.clamp(x, 0, 1)


# -------- Global stats (takes product_root as arg) --------
def compute_global_stats(product_root, band_indices, min_valid=None, max_valid=None):
    pattern = str(Path(product_root) / "fid_*" / "*" / "*" / "tile_*.tif")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No tiles matched: {pattern}")

    C = len(band_indices)
    sum_   = np.zeros(C, dtype=np.float64)
    sum_sq = np.zeros(C, dtype=np.float64)
    count  = np.zeros(C, dtype=np.int64)

    for p in paths:
        data = load_tile(p, band_indices=band_indices)
        valid = np.isfinite(data)
        if min_valid is not None:
            valid &= (data > min_valid)
        if max_valid is not None:
            valid &= (data < max_valid)
        d = np.where(valid, data, 0.0).astype(np.float64)
        sum_   += d.sum(axis=(1, 2))
        sum_sq += (d * d).sum(axis=(1, 2))
        count  += valid.sum(axis=(1, 2)).astype(np.int64)

    mean = sum_ / np.maximum(count, 1)
    var  = sum_sq / np.maximum(count, 1) - mean ** 2
    std  = np.sqrt(np.maximum(var, 0.0))
    return mean.astype(np.float32), std.astype(np.float32), len(paths)


def load_or_compute_stats(tiles_root, product):
    """Load cached stats for <tiles_root>/<product>; compute and cache if missing.
    Stats are computed over ALL tile_*.tif under tiles_root/<product>/."""
    cfg = PRODUCT_CONFIG[product]
    cache_path = Path(tiles_root) / f"stats_{product}.npz"
    if cache_path.exists():
        z = np.load(cache_path)
        return z["mean"].astype(np.float32), z["std"].astype(np.float32)

    print(f"[stats] No cache at {cache_path} -> computing over {product} folder...")
    product_root = Path(tiles_root) / product
    mean, std, n = compute_global_stats(
        product_root, cfg["bands"],
        min_valid=cfg["min_valid"], max_valid=cfg["max_valid"],
    )
    np.savez(cache_path, mean=mean, std=std, n_tiles=n)
    print(f"[stats] Cached to {cache_path}  (n_tiles={n})")
    return mean, std


# -------- Model --------
_MODEL_CACHE = {}

def load_model(modality, device=None):
    """Load CROMA. modality: 'optical' | 'sar' | 'joint'.
    'joint' loads both encoders + the joint encoder so model(x_optical=..., x_sar=...)
    can produce fused multimodal tokens. Cached per modality."""
    if modality in _MODEL_CACHE:
        return _MODEL_CACHE[modality]
    if device is None:
        device = get_device()
    if modality == "joint":
        modalities = ["sar", "optical"]
    elif modality == "optical":
        modalities = ["optical"]
    elif modality == "sar":
        modalities = ["sar"]
    else:
        raise ValueError(f"Unknown modality '{modality}' (use 'optical', 'sar', or 'joint')")
    model = croma_base(
        weights=CROMABase_Weights.CROMA_VIT,
        modalities=modalities,
        image_size=TILE_SIZE,
    ).to(device).eval()
    _MODEL_CACHE[modality] = model
    return model


# -------- Encoding  --------
def encode(model, x_normalized, modality):
    """Forward pass on a normalized batch. Returns (N, 15, 15, 768)."""
    captured = {}
    encoder = model.s2_encoder if modality == "optical" else model.s1_encoder

    def hook(module, inp, out):
        captured["tokens"] = out

    h = encoder.register_forward_hook(hook)
    with torch.no_grad():
        if modality == "optical":
            _ = model(x_optical=x_normalized)
        else:
            _ = model(x_sar=x_normalized)
    h.remove()

    out = captured["tokens"]
    if isinstance(out, tuple):
        out = out[0]
    tokens = out.cpu().numpy()
    if tokens.ndim == 2:
        tokens = tokens[None]
    expected = GRID_SIZE * GRID_SIZE
    if tokens.shape[1] == expected + 1:
        tokens = tokens[:, 1:]
    return tokens.reshape(-1, GRID_SIZE, GRID_SIZE, EMBED_DIM)


# -------- Joint encoding (S1 + S2 fused) --------
def _get_joint_encoder(model):
    """CROMA's joint/cross encoder. torchgeo's attribute name may vary across
    versions; try the common ones and give a useful error otherwise."""
    for attr in ("joint_encoder", "cross_encoder", "multimodal_encoder"):
        if hasattr(model, attr):
            return getattr(model, attr), attr
    available = [a for a in dir(model) if "encoder" in a.lower() and not a.startswith("_")]
    raise AttributeError(
        f"Joint encoder not found on CROMA model. Tried: joint_encoder, "
        f"cross_encoder, multimodal_encoder. Available '*encoder*' attrs: {available}"
    )


def encode_joint(model, x_s2_norm, x_s1_norm):
    """Forward pass with BOTH modalities. Returns (N, 15, 15, 768) joint tokens."""
    captured = {}
    encoder, _attr = _get_joint_encoder(model)

    def hook(module, inp, out):
        captured["tokens"] = out

    h = encoder.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(x_optical=x_s2_norm, x_sar=x_s1_norm)
    h.remove()

    out = captured["tokens"]
    if isinstance(out, tuple):
        out = out[0]
    tokens = out.cpu().numpy()
    if tokens.ndim == 2:
        tokens = tokens[None]
    expected = GRID_SIZE * GRID_SIZE
    if tokens.shape[1] == expected + 1:
        tokens = tokens[:, 1:]
    return tokens.reshape(-1, GRID_SIZE, GRID_SIZE, EMBED_DIM)


# -------- Main entry point --------
def embed_image(image_dir, model=None, stats=None, device=None, overwrite=False):
    """
    Embed every tile_*.tif in image_dir. Saves one .npy per tile (15, 15, 768)
    mirroring input structure under <project_root>/embeddings/.

    model: optional preloaded CROMA model (matching this image_dir's modality).
    stats: optional (mean, std) tuple; loaded from cache if None.
    overwrite: if False, skip tiles whose .npy already exists.
    """
    info = parse_image_dir(image_dir)
    product = info["product"]
    cfg = PRODUCT_CONFIG[product]
    modality = cfg["modality"]
    bands = cfg["bands"]

    tile_paths = sorted(glob.glob(str(Path(image_dir) / "tile_*.tif")))
    if not tile_paths:
        print(f"[skip] No tile_*.tif in {image_dir}")
        return []

    out_dir = (
        EMBEDDINGS_ROOT / product / f"fid_{info['fid']}"
        / info["window"] / info["image_id"]
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for tp in tile_paths:
        out_path = out_dir / (Path(tp).stem + ".npy")
        if out_path.exists() and not overwrite:
            continue
        todo.append((tp, out_path))

    if not todo:
        print(f"[done] All embeddings already exist in {out_dir}")
        return [out_dir / (Path(tp).stem + ".npy") for tp in tile_paths]

    if device is None:
        device = get_device()
    if stats is None:
        stats = load_or_compute_stats(info["tiles_root"], product)
    mean, std = stats
    if model is None:
        model = load_model(modality, device=device)

    arr = np.stack([load_tile(tp, band_indices=bands) for tp, _ in todo]).astype(np.float32)
    x = torch.from_numpy(arr)
    x = normalize(x, mean, std).to(device)
    tokens = encode(model, x, modality)  # (N, 15, 15, 768)

    written = []
    for (tp, out_path), emb in zip(todo, tokens):
        np.save(out_path, emb.astype(np.float32))
        written.append(out_path)

    print(f"[ok] {image_dir} -> {len(written)} new .npy in {out_dir}")
    return written


# -------- Joint mode: one (S2, S1) acquisition pair --------
def embed_image_joint(s2_image_dir, s1_image_dir, model=None,
                     s2_stats=None, s1_stats=None, device=None, overwrite=False):
    """
    Embed a single (S2, S1) acquisition pair through CROMA's joint encoder.
    Saves one .npy per tile (15, 15, 768) under:
        <project_root>/embeddings/joint/fid_<fid>/<window>/<s2_id>__<s1_id>/tile_N.npy

    Requirements:
      - Both image_dirs must point to the SAME fid and SAME window.
      - Tiles are matched by basename (tile_0.tif S2 <-> tile_0.tif S1).
        Only tiles present in BOTH folders are embedded.

    model: optional preloaded CROMA loaded via load_model('joint').
    s2_stats, s1_stats: optional (mean, std) tuples for each modality.
    """
    s2_info = parse_image_dir(s2_image_dir)
    s1_info = parse_image_dir(s1_image_dir)

    if s2_info["product"] != "s2_l2a":
        raise ValueError(f"Expected s2_l2a for s2_image_dir, got {s2_info['product']}")
    if s1_info["product"] != "s1_grd":
        raise ValueError(f"Expected s1_grd for s1_image_dir, got {s1_info['product']}")
    if s2_info["fid"] != s1_info["fid"]:
        raise ValueError(f"FID mismatch: S2={s2_info['fid']}  S1={s1_info['fid']}")
    if s2_info["window"] != s1_info["window"]:
        raise ValueError(f"Window mismatch: S2={s2_info['window']}  S1={s1_info['window']}")

    s2_tiles = {Path(p).name: p for p in glob.glob(str(Path(s2_image_dir) / "tile_*.tif"))}
    s1_tiles = {Path(p).name: p for p in glob.glob(str(Path(s1_image_dir) / "tile_*.tif"))}
    shared = sorted(set(s2_tiles) & set(s1_tiles))
    if not shared:
        print(f"[skip joint] No shared tile names between\n  S2: {s2_image_dir}\n  S1: {s1_image_dir}")
        return []

    out_dir = (
        EMBEDDINGS_ROOT / "joint" / f"fid_{s2_info['fid']}" / s2_info["window"]
        / f"{s2_info['image_id']}__{s1_info['image_id']}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for name in shared:
        out_path = out_dir / (Path(name).stem + ".npy")
        if out_path.exists() and not overwrite:
            continue
        todo.append((s2_tiles[name], s1_tiles[name], out_path))

    if not todo:
        print(f"[done joint] All embeddings already exist in {out_dir}")
        return [out_dir / (Path(n).stem + ".npy") for n in shared]

    if device is None:
        device = get_device()
    if s2_stats is None:
        s2_stats = load_or_compute_stats(s2_info["tiles_root"], "s2_l2a")
    if s1_stats is None:
        s1_stats = load_or_compute_stats(s1_info["tiles_root"], "s1_grd")
    if model is None:
        model = load_model("joint", device=device)

    s2_mean, s2_std = s2_stats
    s1_mean, s1_std = s1_stats

    arr_s2 = np.stack([load_tile(s2p, band_indices=S2_BAND_INDICES) for s2p, _, _ in todo]).astype(np.float32)
    arr_s1 = np.stack([load_tile(s1p, band_indices=S1_BAND_INDICES) for _, s1p, _ in todo]).astype(np.float32)

    x_s2 = normalize(torch.from_numpy(arr_s2), s2_mean, s2_std).to(device)
    x_s1 = normalize(torch.from_numpy(arr_s1), s1_mean, s1_std).to(device)

    tokens = encode_joint(model, x_s2, x_s1)  # (N, 15, 15, 768)

    written = []
    for (_, _, out_path), emb in zip(todo, tokens):
        np.save(out_path, emb.astype(np.float32))
        written.append(out_path)

    print(f"[ok joint] fid={s2_info['fid']} {s2_info['window']} -> {len(written)} new .npy in {out_dir}")
    return written


# -------- Date parsing (for the CSV pair iterator) --------
def _parse_s2_date(image_id):
    """S2 image_id starts with 'YYYYMMDDTHHMMSS_...'."""
    m = re.match(r"(\d{8})T", str(image_id))
    if not m:
        return None
    d = m.group(1)
    return date(int(d[:4]), int(d[4:6]), int(d[6:8]))


def _parse_s1_date(image_id):
    """S1 image_id contains '_YYYYMMDDTHHMMSS_'."""
    m = re.search(r"_(\d{8})T", str(image_id))
    if not m:
        return None
    d = m.group(1)
    return date(int(d[:4]), int(d[4:6]), int(d[6:8]))


# -------- CSV pair iterator (v1/v2/v3_images_s2_s1.csv: comma-separated lists) --------
def _split_ids(cell):
    """Parse a comma-separated cell into a list of stripped image_ids. Empty/NaN -> []."""
    import pandas as pd
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, str):
        s = cell.strip()
    else:
        try:
            if pd.isna(cell):
                return []
        except Exception:
            pass
        s = str(cell).strip()
    if not s or s.lower() == "nan":
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


def iter_joint_pairs_from_csv(csv_path, tiles_root, max_gap_days=7,
                              windows=("bef", "evt", "aft"),
                              require_tiles_exist=True):
    """
    Yield (fid, window, s2_image_dir, s1_image_dir, gap_days) for each unique
    joint pair found in a v1/v2/v3_images_s2_s1.csv.

    For each (fid, window), the CSV provides two comma-separated lists:
        <win>IdsS1: candidate S1 acquisitions
        <win>IdsS2: candidate S2 acquisitions

    Pairing strategy (per (fid, window) independently):
      1. Build all candidate (s2, s1) pairs whose date gap <= max_gap_days.
      2. Sort candidates by gap ascending (ties broken by S1 id, then S2 id, for determinism).
      3. Greedy assignment: pick a pair if neither side is already used in this
         (fid, window). Each s1 image_id and each s2 image_id is consumed at most once.

    csv_path: e.g. 'data_csv/v3_images_s2_s1.csv'
    tiles_root: path to .../tiles_120px/ (parent of s2_l2a/ and s1_grd/)
    require_tiles_exist: skip pairs whose image_id folder is not on disk.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    tiles_root = Path(tiles_root)

    for _, row in df.iterrows():
        fid = int(row["fid"])
        for win in windows:
            s1_col = f"{win}IdsS1"
            s2_col = f"{win}IdsS2"
            if s1_col not in df.columns or s2_col not in df.columns:
                continue
            s1_list = _split_ids(row[s1_col])
            s2_list = _split_ids(row[s2_col])
            if not s1_list or not s2_list:
                continue

            s1_dates = {sid: _parse_s1_date(sid) for sid in s1_list}
            s2_dates = {sid: _parse_s2_date(sid) for sid in s2_list}

            candidates = []
            for s1_id, d1 in s1_dates.items():
                if d1 is None:
                    continue
                for s2_id, d2 in s2_dates.items():
                    if d2 is None:
                        continue
                    g = abs((d1 - d2).days)
                    if g <= max_gap_days:
                        candidates.append((g, s1_id, s2_id))
            candidates.sort(key=lambda x: (x[0], x[1], x[2]))

            used_s1, used_s2 = set(), set()
            for gap, s1_id, s2_id in candidates:
                if s1_id in used_s1 or s2_id in used_s2:
                    continue
                s2_dir = tiles_root / "s2_l2a" / f"fid_{fid}" / win / s2_id
                s1_dir = tiles_root / "s1_grd" / f"fid_{fid}" / win / s1_id
                if require_tiles_exist and not (s2_dir.is_dir() and s1_dir.is_dir()):
                    continue
                used_s1.add(s1_id)
                used_s2.add(s2_id)
                yield (fid, win, str(s2_dir), str(s1_dir), gap)


# -------- CLI --------
def main():
    ap = argparse.ArgumentParser(description="Generate CROMA embeddings for one image_id.")
    ap.add_argument("image_dir", help="Path to a .../<product>/fid_<N>/<window>/<image_id>/ folder")
    ap.add_argument("--overwrite", action="store_true", help="Recompute even if .npy exists")
    args = ap.parse_args()

    device = get_device()
    print(f"[device] {device}")
    embed_image(args.image_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
