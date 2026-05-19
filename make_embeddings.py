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
from pathlib import Path

import numpy as np
import rasterio
import torch
from torchgeo.models import croma_base, CROMABase_Weights


# -------- Constants (same as the notebook) --------
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
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
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


# -------- Tile IO (identical to notebook) --------
def load_tile(path, band_indices=None):
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
    if band_indices is not None:
        data = data[band_indices]
    return data


# -------- Normalization (identical to notebook) --------
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


# -------- Global stats (identical body, takes product_root as arg) --------
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
    """Load CROMA. modality: 'optical' or 'sar'. Cached per modality so repeated
    calls in one process don't re-load."""
    if modality in _MODEL_CACHE:
        return _MODEL_CACHE[modality]
    if device is None:
        device = get_device()
    mod_arg = "optical" if modality == "optical" else "SAR"
    model = croma_base(
        weights=CROMABase_Weights.CROMA_VIT,
        modalities=[mod_arg],
        image_size=TILE_SIZE,
    ).to(device).eval()
    _MODEL_CACHE[modality] = model
    return model


# -------- Encoding (identical to notebook) --------
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
