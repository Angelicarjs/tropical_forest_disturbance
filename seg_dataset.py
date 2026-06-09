"""
Segmentation dataset for disturbance-type mapping.

Each sample = one tile:
  - input : CROMA joint embedding  (768, 15, 15)   <- from embeddings/joint/
  - target: per-pixel class mask    (120, 120)       <- rasterized on the fly from the shapefile

Pixels inside the polygon get the class id; everything else is 0 (background).

Only the 'evt' and 'aft' windows are used (the disturbance is visible there;
'bef' shows intact forest and would mislabel pixels).
"""

import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.features import rasterize
import geopandas as gpd

# clases
CLASS_TO_ID = {
    "Clear-cut bare soil":            1,
    "Fire scar":                      2,
    "Degradation":                    3,
    "Disorderly Selective logging":   4,
    "Clear-cut vegetation":           5,
    "Geometric Selective logging":    6,
}
NUM_CLASSES = len(CLASS_TO_ID) + 1  # +1 for background


class DisturbanceSegDataset(Dataset):
    def __init__(self, embeddings_root, tiles_root, shp_path,
                 windows=("evt", "aft")):
        """
        embeddings_root: .../embeddings  (contains joint/)
        tiles_root:      .../thesis_tiles_120px  (contains s2_l2a/)
        shp_path:        .../label_polygons.shp
        """
        self.tiles_root = Path(tiles_root)
        self.windows = set(windows)

        # --- Load polygons once, reproject to the tiles' CRS (EPSG:3857) ---
        gdf = gpd.read_file(shp_path)
        gdf["fid"] = gdf["fid"].astype(int).astype(str)
        gdf = gdf.to_crs("EPSG:3857")
        # fid -> list of (shapely_geometry, class_id)
        self.fid_polys = {}
        for _, row in gdf.iterrows():
            cid = CLASS_TO_ID.get(row["CLASSNAME"])
            if cid is None:
                continue
            self.fid_polys.setdefault(row["fid"], []).append((row.geometry, cid))

        # --- Index every embedding tile in the requested windows ---
        # path: embeddings/joint/fid_<fid>/<window>/<s2id__s1id>/tile_N.npy
        self.samples = []
        for npy in glob.glob(str(Path(embeddings_root) / "joint" / "fid_*" / "*" / "*" / "tile_*.npy")):
            p = Path(npy)
            pair = p.parent.name          # "<s2id>__<s1id>"
            window = p.parent.parent.name
            fid = p.parent.parent.parent.name.replace("fid_", "")
            if window not in self.windows:
                continue
            if fid not in self.fid_polys:
                continue                  # no label polygon -> skip
            self.samples.append({
                "npy": npy, "fid": fid, "window": window,
                "s2_id": pair.split("__")[0], "tile": p.stem,  # "tile_N"
            })

    def __len__(self):
        return len(self.samples)

    def _build_mask(self, s):
        """Rasterize this tile's polygon onto its 120x120 grid (on the fly)."""
        tif = (self.tiles_root / "s2_l2a" / f"fid_{s['fid']}"
               / s["window"] / s["s2_id"] / f"{s['tile']}.tif")
        with rasterio.open(tif) as src:
            transform = src.transform        # pixel <-> EPSG:3857 coords
            H, W = src.height, src.width      # 120, 120
        shapes = self.fid_polys[s["fid"]]     # [(geom, class_id), ...]
        mask = rasterize(
            shapes, out_shape=(H, W), transform=transform,
            fill=0, dtype="int32", all_touched=False,
        )
        return mask.astype(np.int64)

    def __getitem__(self, i):
        s = self.samples[i]
        emb = np.load(s["npy"])               # (15, 15, 768)
        emb = np.transpose(emb, (2, 0, 1))    # (768, 15, 15) -> channels first
        x = torch.from_numpy(emb).float()
        y = torch.from_numpy(self._build_mask(s)).long()  # (120, 120)
        return x, y


if __name__ == "__main__":
    # Smoke test — adjust paths to your machine
    import os
    EMB = os.path.expanduser("~/thesis_tiles_120px") if False else "embeddings"
    ds = DisturbanceSegDataset(
        embeddings_root="embeddings",
        tiles_root=os.path.expanduser("~/thesis_tiles_120px"),
        shp_path="data_shp/label_polygons.shp",
    )
    print(f"samples (evt+aft): {len(ds)}")
    x, y = ds[0]
    print(f"input  x: {tuple(x.shape)}  dtype={x.dtype}")
    print(f"target y: {tuple(y.shape)}  dtype={y.dtype}")
    print(f"clases presentes en la máscara[0]: {sorted(torch.unique(y).tolist())}")
