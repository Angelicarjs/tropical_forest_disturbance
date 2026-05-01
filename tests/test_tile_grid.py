from shapely.geometry import box

from tile_pipeline import create_tile_grid


def test_polygon_inside_one_tile_returns_one():
    # 600x600 m polygon centered well inside a single 1200 m tile
    poly = box(300, 300, 900, 900)
    assert len(create_tile_grid(poly, 1200)) == 1


def test_polygon_straddling_corner_returns_four():
    # 600x600 m polygon crossing the (1200, 1200) tile corner in both axes
    poly = box(900, 900, 1500, 1500)
    assert len(create_tile_grid(poly, 1200)) == 4


def test_grid_alignment_artifact_fid_101_pattern():
    """Same polygon, three tile sizes, mismatched grid origins.

    Polygon at [2100, 2300]² lies inside one 1200 m tile and one 2560 m
    tile, but straddles the 2240 m grid line in BOTH axes — so it gets
    split into 4 TerraFM tiles. This is the FID 101 case.
    """
    poly = box(2100, 2100, 2300, 2300)
    assert len(create_tile_grid(poly, 1200)) == 1   # CROMA
    assert len(create_tile_grid(poly, 2240)) == 4   # TerraFM
    assert len(create_tile_grid(poly, 2560)) == 1   # Clay


def test_tiles_cover_polygon_area():
    poly = box(0, 0, 3000, 3000)
    tiles = create_tile_grid(poly, 1200)
    assert sum(t.area for t in tiles) >= poly.area
