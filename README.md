# Intra-Year Detection of Tropical Forest Disturbances Using Multimodal Foundation Model Embeddings

Code for the Master thesis of the Copernicus Master in Digital Earth (CDE).

Sentinel-2 (optical) and Sentinel-1 (SAR) acquisitions around 389 DETER-B disturbance polygons in
Rondônia for 2023 are encoded with the frozen **CROMA** encoder in its optical, radar and joint
modes, and the resulting 768-dimensional token embeddings are studied along two axes:

- **RQ1 (qualitative)** — how the temporal trajectory of a token embedding represents a
  disturbance, and how far that signal survives cloud contamination.
- **RQ2 (quantitative)** — how well per-token classifiers separate six disturbance types from
  intact forest, and on which date a near real time rule would report the event, against the DETER
  reference date.


## Layout

```
.                            shared library, imported by every stage
├── obs_date.py              date of an observation (a joint pair takes its later acquisition)
├── seg_dataset.py           dataset, class ids, label rasterization, tile alignment
├── token_pipeline.py        RQ2 harness: split, balancing, random forest, sensitivity curve
├── log_reg.py               logistic regression probe, a plug-in for token_pipeline.signal()
├── make_embeddings.py       CROMA encoding, normalization statistics, S1/S2 pairing
├── eval_plots.py            shared plotting for the classifier evaluation
├── time_series_pipeline.py  RQ1 engine: per (fid, tile, token) temporal analysis
├── cluster_setup.sh         creates the `gee_tiles` environment, authenticates Earth Engine
│
├── data/                    data_shp (389 polygons) | data_csv (filter1:v1/filter2:v2/filter3:v3) | data_img | splits
├── preprocessing/           code (GEE download, tiling, QC) | analysis (figures) | logs
├── embedding_extraction/    code (embed_all.py) | logs
├── qualitative/             RQ1: code (notebooks) | results/figures_rq1
├── classification_models/   RQ2 classification: code | results | logs
└── near_real_time/          RQ2 detection: code | results
```

Tiles, `embeddings/`, `joint_embe_forest/` and the fitted `*.joblib` are gitignored. Everything
else, including the result tables and figures of every reported run, is versioned.

**Two rules, or nothing runs.** Run every command **from the repository root**, since all data
paths are relative; and keep the root on `sys.path`, since the shared library lives there. The
`.sh` launchers carry `export PYTHONPATH="$PWD"`; by hand use `PYTHONPATH=. python ...`.

## Environments

`gee_tiles` covers the download stage only and is created by `bash cluster_setup.sh`
(earthengine-api, shapely, pyproj). `croma_viz` covers everything downstream and no script creates
it:

```bash
conda create -n croma_viz python=3.11 -y && conda activate croma_viz
pip install torch torchgeo rasterio geopandas numpy pandas scikit-learn \
            matplotlib joblib tqdm umap-learn scienceplots
```

## Data

389 DETER-B polygons for 2023, six disturbance classes (clear-cut bare soil 271, fire scar 44,
degradation 40, disorderly selective logging 20, clear-cut vegetation 10, geometric selective
logging 4), plus a forest class sampled from PRODES. `VIEW_DATE` is a detection date, not an
occurrence date.

Three temporal windows around `VIEW_DATE`: *before* (1 January to −8 d), *event* (±7 d), *after*
(+8 to +28 d).

Three cloud filters, one CSV each: Filter1(v1) = SCL + QA60, Filter2(v2) = SCL + Cloud Score+ (≥ 0.50), Filter3(v3) = v2 plus
a 10 px dilation. 

`data/splits/` holds a partition drawn once with seed 42, stratified by type and cut at polygon
level: 175 trainval, 171 test. That is 346 of the 389 polygons; the 43 excluded have no Sentinel-2
acquisition admitted by v3 (15), acquisitions only in the before window (14), or lost every
Sentinel-1 tile at the edge of its GRD frame (14). Only the first two causes are atmospheric.

## Pipeline

**1. Download and tiling** (`gee_tiles`). Sentinel-2 L2A (12 bands, no B10) and Sentinel-1 GRD
(VV, VH), EPSG:3857 at 10 m, 120 × 120 px tiles.

```bash
python preprocessing/code/tile_pipeline.py --workers 6 --resume
```

**2. CROMA encoding** (`croma_viz`, GPU). Frozen `croma_base` from TorchGeo; one `(15, 15, 768)`
array per tile. Resumable: tiles already on disk are skipped.

```bash
python embedding_extraction/code/embed_all.py \
    --tiles-root ~/thesis_tiles_120px --csv-dir data/data_csv --modality both
```

The tiles are read from `$HOME/thesis_tiles_120px`; set `THESIS_TILES` to override.

S1 to S2 pairing lives in `make_embeddings.iter_joint_pairs_from_csv`: closest scene in either
direction, at most 7 days, greedy and one to one.

**3. RQ1.** Run `qualitative/code/rq1_temporal_analysis.ipynb` from the repository root. Worked
example FID 188, tile 0, token (3, 6), v3: cosine similarity matrix between dates, most variable
dimension, and PC1 to PC3 on axes fitted once on a reference image, for optical, SAR and joint.
The synthetic cloud analysis is in `preprocessing/analysis/cloud_eva.ipynb`.

**4. RQ2, classification.** One token = one sample, labeled by majority vote over its 8 × 8 pixel
block, event and after windows only; training capped at 2,500 tokens per class.

```bash
R=classification_models/results
python classification_models/code/run_eval.py --results-root $R/results_0 \
    --align-with joint s2_l2a --fids 83 389 25 3
python classification_models/code/run_eval.py --embed-kind s2_l2a \
    --forest-root embeddings/s2_l2a_forest --results-root $R/results_optical \
    --align-with joint s2_l2a --fids 83 389 25 3
python classification_models/code/run_eval_binary.py --results-root $R/results_binary \
    --align-with joint s2_l2a
```

Naming a `--results-root` is what the four `run_eval*.sh` launchers do, and it is what reproduces
the reported runs in place. Left to their defaults the scripts write to `results_new` and
`results_binary_new` instead, so an accidental run never overwrites the results of the
manuscript.

`--align-with` is required for any comparison between modalities: joint needs both sensors and
optical only Sentinel-2, so without it the two cover different tiles.

**5. RQ2, near real time.** Scores per acquisition, then a threshold fixed on trainval, then
timeliness on test.

```bash
./near_real_time/code/run_scores.sh                                  # trainval
SPLIT=data/splits/split_test_fids.txt TAG=test ./near_real_time/code/run_scores.sh
# nrt_threshold.ipynb picks tau: 5 % FA, 10 % FA and Youden, per mode and per model
./near_real_time/code/run_timing.sh                                  # six runs
```

## Results

`classification_models/results/results_0` is the reported joint run (`results_optical*`,
`results_binary*` for the variants; `results_1..6` are earlier runs kept for traceability).
`near_real_time/results/` holds the per-acquisition scores, the ROC curves and the timeliness
tables at each operating point. `qualitative/results/figures_rq1/` holds the RQ1 figures.

## Citation

> Moreno Rojas, A. M. (2026). *Intra-Year Detection of Tropical Forest Disturbances Using
> Multimodal Foundation Model Embeddings.* Master thesis, Copernicus Master in Digital Earth.

CROMA is by Fuller, Millard and Green (2023). Labels come from DETER-B and the forest mask from
PRODES, both INPE.
