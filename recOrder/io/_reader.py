from waveorder.io.reader import WaveorderReader
import zarr
from typing import Tuple, List, Dict, Union

def napari_get_reader(path):
    if isinstance(path, str):
        if '.zarr' in path:
            return ome_zarr_reader
        else:
            return ome_tif_reader
    else:
        return None


def ome_zarr_reader(path: Union[str, List[str]]) -> List[Tuple[zarr.Array, Dict]]:
    reader = WaveorderReader(path)
    results = list()

    zs = zarr.open(path, 'r')
    names = []

    dict_ = zs.attrs.asdict()
    wells = dict_['plate']['wells']
    for well in wells:
        path = well['path']
        well_dict = zs[path].attrs.asdict()
        for name in well_dict['well']['images']:
            names.append(name['path'])
    for pos in range(reader.get_num_positions()):
        meta = dict()
        name = names[pos]
        meta['name'] = name
        results.append((reader.get_zarr(pos), meta))

    return results

def ome_tif_reader(path: Union[str, List[str]]) -> List[Tuple[zarr.Array, Dict]]:
    reader = WaveorderReader(path)
    results = list()

    npos = reader.get_num_positions()
    for pos in range(npos):
        meta = dict()
        if npos == 1:
            meta['name'] = 'Pos000_000'
        else:
            meta['name'] = reader.stage_positions[pos]['Label'][2:]
        results.append((reader.get_zarr(pos), meta))

    return results