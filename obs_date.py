"""Date of an observation, shared by every analysis so they cannot disagree.

A joint observation only exists once both acquisitions have been made, so it is
dated by the later of the two. The pairing picks the Sentinel-1 scene closest in
time to the Sentinel-2 one in either direction, so in a large share of the pairs
the radar image is the later one; dating a pair by its Sentinel-2 acquisition
alone would credit the system with information it did not yet have.

Unimodal observations carry a single date, taken from their own image id.

Directory names:
    S2      20230430T142719_20230430T142714_T20LMR
    S1      S1A_IW_GRDH_1SDV_20230415T094936_20230415T095001_048105_05C876_7D01
    joint   <s2 id>__<s1 id>
"""
import re
from pathlib import Path

_S2_DATE = re.compile(r"^(\d{8})T")      # S2 ids start with the date
_S1_DATE = re.compile(r"_(\d{8})T")      # S1 ids carry it after a separator


def image_id(path) -> str:
    """The acquisition directory of a tile_N.{tif,npy} path."""
    return Path(path).parent.name


def obs_date(name) -> str:
    """YYYYMMDD of the observation. Accepts an image id or a path to a tile.

    Dates are returned as strings, which sort chronologically, so they can be
    used directly as a sort key.
    """
    name = str(name)
    if "/" in name or "\\" in name:
        name = image_id(name)

    if "__" in name:                     # joint: the later of the two
        s2_part, s1_part = name.split("__", 1)
        d2 = _S2_DATE.match(s2_part)
        d1 = _S1_DATE.search(s1_part)
        if d2 and d1:
            return max(d2.group(1), d1.group(1))
        return (d2 or d1).group(1)

    m = _S2_DATE.match(name) or _S1_DATE.search(name)
    if m is None:
        raise ValueError(f"no date in {name!r}")
    return m.group(1)
