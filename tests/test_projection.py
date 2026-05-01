from shapely.geometry import Point, Polygon

from tile_pipeline import project_to_3857


def test_origin_maps_to_origin():
    p = project_to_3857(Point(0, 0))
    assert abs(p.x) < 1e-6
    assert abs(p.y) < 1e-6


def test_amazon_lon_lat_to_meters():
    # (-60, -5) → x ≈ -6_678_170 m, y ≈ -557_417 m in EPSG:3857
    p = project_to_3857(Point(-60, -5))
    assert abs(p.x - (-6_678_170)) < 100
    assert abs(p.y - (-557_417)) < 100


def test_polygon_geometry_type_preserved():
    poly = Polygon([(-60, -5), (-59, -5), (-59, -4), (-60, -4)])
    proj = project_to_3857(poly)
    assert proj.geom_type == "Polygon"
    assert proj.is_valid
