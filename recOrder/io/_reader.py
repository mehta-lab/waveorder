from waveorder.io.reader import WaveorderReader
import zarr
from typing import Tuple, List, Dict, Union

def napari_get_reader(path):
    if isinstance(path, str) and '.zarr' in path:
        return ome_zarr_reader
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