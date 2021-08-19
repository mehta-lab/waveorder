from pycromanager import Bridge
import os
import zarr
from tqdm import tqdm
import json
import numpy as np
from numcodecs import Blosc
import shutil

class ZarrConverter:

    def __init__(self, save_directory, save_name=None):
        self.version = 'ZC 0.0'
        self.save_directory = save_directory
        self.save_name = save_name

        self._connect_and_setup_mm()

        self.data_name = self.summary_metadata.getPrefix()
        self.dtype = self.get_dtype()
        self.p = self.data_provider.getMaxIndices().getP()
        self.t = self.data_provider.getMaxIndices().getT()
        self.c = self.data_provider.getMaxIndices().getC()
        self.z = self.data_provider.getMaxIndices().getZ()
        self.y = self.data_provider.getAnyImage().getHeight()
        self.x = self.data_provider.getAnyImage().getWidth()
        self.dim = (self.p, self.t, self.c, self.z, self.y, self.x)
        print(f'Found Dataset {self.data_name} w/ dimensions (P, T, C, Z, Y, X): {self.dim}')

        self.CoordBuilder = self.data_provider.getAnyImage().getCoords().copyBuilder()

        self.metadata = dict()

        self.temp_directory = os.path.join(os.path.expanduser('~'),'recOrder_temp')
        self.stats_path = os.path.join(self.save_directory, self.save_name+'_Statistics.txt')
        self.stats_file = open(self.stats_path, 'w')
        self.temp_path = None
        self.java_path = None
        if not os.path.exists(self.temp_directory): os.mkdir(self.temp_directory)

    def _gen_coordset(self):
        return [(p,t,c,z) for p in range(self.p) for t in range(self.t) for c in range(self.t) for z in range(self.z)]

    def _connect_and_setup_mm(self):
        try:
            self.bridge = Bridge(convert_camel_case=False)
            self.mmc = self.bridge.get_core()
            self.mm = self.bridge.get_studio()

            data_viewers = self.mm.getDisplayManager().getAllDataViewers()
            if data_viewers.size != 0:
                raise ValueError(f'Detected {data_viewers.size()} data viewers \
                Make sure the only dataviewer opened is your desired dataset')
            else:
                self.data_viewer = self.mm.getDisplayManager().getAllDataViewers().get(0)
                self.data_provider = self.data_viewer.getDataProvider()
                self.summary_metadata = self.data_provider.getSummaryMetadata()
        except:
            raise ValueError('Please make sure MM is running and the data is opened')


    def _generate_summary_metadata(self):

        self.temp_path = os.path.join(self.temp_directory, 'meta.json')
        self.java_path = self.bridge.construct_java_object('java.io.File', args=[temp_path])

        PropertyMap = self.summary_metadata.toPropertyMap()
        PropertyMap.saveJSON(self.java_path, True, False)

        f = open(self.temp_path)
        dict_ = json.load(f)
        f.close()

        self.metadata['Summary'] = dict_

    def _generate_plane_metadata(self, image):

        PropertyMap = image.getMetadata().toPropertyMap()
        PropertyMap.saveJSON(self.java_path, True, False)

        f = open(self.temp_path)
        image_metadata = json.load(f)
        f.close()

        return image_metadata

    def _get_dtype(self):

        return str(self.data_provider.getAnyImage().getRawPixels().dtype)

    def _save_image_stats(self, image, coord):

        mean = np.mean(image)
        median = np.median(image)
        std = np.std(image)
        self.stats_file.write(f'Coord: {coord}, Mean: {mean}, Median: {median}, Std: {std}\n')

    def get_image_object(self, coord):
        self.CoordBuilder.p(coord[0])
        self.CoordBuilder.t(coord[1])
        self.CoordBuilder.c(coord[2])
        self.CoordBuilder.z(coord[3])
        mm_coord = self.CoordBuilder.build()

        return self.data_provider.getImage(mm_coord)

    def setup_zarr(self):

        src = os.path.join(self.save_directory, self.save_name if self.save_name else self.data_name)

        self.zarr_store = zarr.open(src)
        self.array = self.zarr_store.create('array',
                                            shape=(self.p if self.p != 0 else 1,
                                                   self.t if self.t != 0 else 1,
                                                   self.c if self.c != 0 else 1,
                                                   self.z if self.z != 0 else 1,
                                                   self.y,
                                                   self.x),
                                            chunk_size=(1, 1, 1, 1, self.y, self.x),
                                            compressor=Blosc('zstd', clevel=3, shuffle=Blosc.BITSHUFFLE),
                                            dtype=self.dtype,
                                            read_only=True)

    def run_conversion(self):

        coords = self.gen_coordset()
        bar_format = 'Status: |{bar}|{n_fmt}/{total_fmt} (Time Remaining: {remaining} s), {rate_fmt}{postfix}]'

        for coord in tqdm(coords, bar_format=bar_format):
            
            img = self.get_image_object(coord)
            
            self.metadata['ImagePlaneMetadata'][f'{coord}'] = self._generate_plane_metadata(img)
            img_raw = img.getRawPixels().reshape(self.y, self.x)
            self.array[coord[0], coord[1], coord[2], coord[3]] = img_raw
            self._save_image_stats(img_raw)

        self.zarr_store.attrs.put(self.metadata)
        self.stats_file.close()
        shutil.rmtree(self.temp_directory)


    def run_random_img_test(self, n_rounds = 10):

        for i in range(n_rounds):
            image_object = self.data_provider.getAnyImage()
            coord_object = image_object.getCoords()

            coord = (coord_object.getP(), coord_object.getT(), coord_object.getC(), coord_object.getZ())
            img_raw = image_object.getRawPixels.reshape(self.x, self.y)
            img_saved = self.array[coord[0], coord[1], coord[2], coord[3]]

            if img_raw != img_saved:

                print(f'coordinate {coord} does not match raw data.  Conversion Failed. DO NOT DELETE ORIGINAL DATA')
                break













































