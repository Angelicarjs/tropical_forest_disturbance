import csv

from tile_pipeline import parse_and_merge_csvs

GEO = '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]}'
FIELDNAMES = ["fid", ".geo",
              "evtIdsS2", "befIdsS2", "aftIdsS2",
              "evtIdsS1", "befIdsS1", "aftIdsS1"]


def _row(fid, **overrides):
    base = {f: "" for f in FIELDNAMES}
    base["fid"] = fid
    base[".geo"] = GEO
    base.update(overrides)
    return base


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_dedup_same_image_across_versions(tmp_path):
    rows = [_row("1", evtIdsS2="IMG_A")]
    _write(tmp_path / "v1_images_s2_s1.csv", rows)
    _write(tmp_path / "v2_images_s2_s1.csv", rows)
    _write(tmp_path / "v3_images_s2_s1.csv", rows)

    records = parse_and_merge_csvs(str(tmp_path))
    assert len(records) == 1
    assert records[0]["images"][("s2", "evt")] == ["IMG_A"]


def test_union_of_different_images_across_versions(tmp_path):
    _write(tmp_path / "v1_images_s2_s1.csv", [_row("1", evtIdsS2="IMG_A")])
    _write(tmp_path / "v2_images_s2_s1.csv", [_row("1", evtIdsS2="IMG_B")])
    _write(tmp_path / "v3_images_s2_s1.csv", [_row("1", evtIdsS2="IMG_C")])

    records = parse_and_merge_csvs(str(tmp_path))
    assert records[0]["images"][("s2", "evt")] == ["IMG_A", "IMG_B", "IMG_C"]


def test_multiple_ids_in_one_cell_split_on_comma(tmp_path):
    _write(tmp_path / "v1_images_s2_s1.csv",
           [_row("1", befIdsS1="ID_X, ID_Y, ID_Z")])
    _write(tmp_path / "v2_images_s2_s1.csv", [])
    _write(tmp_path / "v3_images_s2_s1.csv", [])

    records = parse_and_merge_csvs(str(tmp_path))
    assert records[0]["images"][("s1", "bef")] == ["ID_X", "ID_Y", "ID_Z"]


def test_first_only_keeps_one_per_window(tmp_path):
    _write(tmp_path / "v1_images_s2_s1.csv",
           [_row("1", evtIdsS2="A,B,C", evtIdsS1="X,Y")])
    _write(tmp_path / "v2_images_s2_s1.csv", [])
    _write(tmp_path / "v3_images_s2_s1.csv", [])

    records = parse_and_merge_csvs(str(tmp_path), first_only=True)
    assert len(records[0]["images"][("s2", "evt")]) == 1
    assert len(records[0]["images"][("s1", "evt")]) == 1


def test_missing_csv_does_not_crash(tmp_path):
    # Only v1 exists; v2 and v3 missing → still works
    _write(tmp_path / "v1_images_s2_s1.csv", [_row("1", evtIdsS2="IMG_A")])

    records = parse_and_merge_csvs(str(tmp_path))
    assert len(records) == 1
    assert records[0]["images"][("s2", "evt")] == ["IMG_A"]
