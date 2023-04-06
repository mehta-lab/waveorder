from typing import Dict, List, Tuple, Union

import zarr
from iohub import read_micromanager
from napari_ome_zarr._reader import napari_get_reader as fallback_reader


def napari_get_reader(path):
    if isinstance(path, str):
        if ".zarr" in path:
            with zarr.open(path) as root:
                if "plate" in root.attrs:
                    return hcs_zarr_reader
                else:
                    return fallback_reader(path)
        else:
            return ome_tif_reader
    else:
        return None


def hcs_zarr_reader(
    path: Union[str, List[str]]
) -> List[Tuple[zarr.Array, Dict]]:
    reader = read_micromanager(path)
    results = list()

    zs = zarr.open(path, "r")
    names = []

    dict_ = zs.attrs.asdict()
    wells = dict_["plate"]["wells"]
    for well in wells:
        path = well["path"]
        well_dict = zs[path].attrs.asdict()
        for name in well_dict["well"]["images"]:
            names.append(name["path"])
    for pos in range(reader.get_num_positions()):
        meta = dict()
        name = names[pos]
        meta["name"] = name
        results.append((reader.get_zarr(pos), meta))
    return results


def ome_tif_reader(
    path: Union[str, List[str]]
) -> List[Tuple[zarr.Array, Dict]]:
    reader = read_micromanager(path)
    results = list()

    npos = reader.get_num_positions()
    for pos in range(npos):
        meta = dict()
        if npos == 1:
            meta["name"] = "Pos000_000"
        else:
            meta["name"] = reader.stage_positions[pos]["Label"][2:]
        results.append((reader.get_zarr(pos), meta))

    return results
