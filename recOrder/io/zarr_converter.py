from pycromanager import Bridge
import os
import zarr
from tqdm import tqdm
import json
import numpy as np
from numcodecs import Blosc
import shutil
from scripts.md5_check_sum import gen_stats_file, md5

class ZarrConverter:

    def __init__(self, save_directory, save_name=None):

        # Attempt MM Connections
        self._connect_and_setup_mm()

        # Init File IO Properties
        self.version = 'recOrder Converter version=0.0'
        self.save_directory = save_directory
        self.data_name = self.summary_metadata.getPrefix()
        self.save_name = self.data_name if not save_name else save_name
        self.array = None
        self.zarr_store = None
        self.temp_directory = os.path.join(os.path.expanduser('~'), 'recOrder_temp')
        self.stats_path = os.path.join(self.save_directory, self.save_name + '_Statistics.txt')
        self.stats_file = open(self.stats_path, 'w')
        self.temp_path = None
        self.java_path = None
        if not os.path.exists(self.temp_directory): os.mkdir(self.temp_directory)

        # Generate Data Specific Properties
        self.coords = None
        self.dtype = self._get_dtype()
        self.p = self.data_provider.getMaxIndices().getP()+1
        self.t = self.data_provider.getMaxIndices().getT()+1
        self.c = self.data_provider.getMaxIndices().getC()+1
        self.z = self.data_provider.getMaxIndices().getZ()+1
        self.y = self.data_provider.getAnyImage().getHeight()
        self.x = self.data_provider.getAnyImage().getWidth()
        self.dim = (self.p, self.t, self.c, self.z, self.y, self.x)
        print(f'Found Dataset {self.data_name} w/ dimensions (P, T, C, Z, Y, X): {self.dim}')

        # Initialize Coordinate Builder
        self.CoordBuilder = self.data_provider.getAnyImage().getCoords().copyBuilder()

        # Initialize Metadata Dictionary
        self.metadata = dict()

    def _gen_coordset(self):
        """
        generates a coordinate set for (p, t, c, z).

        Returns
        -------
        list(tuples) w/ dimensions [N_images]

        """

        return [(p, t, c, z) for p in range(self.p) for t in range(self.t) for c in range(self.c) for z in range(self.z)]

    def _connect_and_setup_mm(self):
        """
        Attempts MM connection and checks to make sure only one dataset is opened.
        If failed, prompts user to open MM, close other datasets.
        If Successful, get the necessary data providers.

        Returns
        -------

        """
        try:
            self.bridge = Bridge(convert_camel_case=False)
            self.mmc = self.bridge.get_core()
            self.mm = self.bridge.get_studio()

            data_viewers = self.mm.getDisplayManager().getAllDataViewers()
        except:
            raise ValueError('Please make sure MM is running and the data is opened')

        if data_viewers.size() != 1:
            raise ValueError(f'Detected {data_viewers.size()} data viewers \
            Make sure the only dataviewer opened is your desired dataset')
        else:
            self.data_viewer = self.mm.getDisplayManager().getAllDataViewers().get(0)
            self.data_provider = self.data_viewer.getDataProvider()
            self.summary_metadata = self.data_provider.getSummaryMetadata()


    def _generate_summary_metadata(self):
        """
        generates the summary metadata by saving the existing java PropertyMap as JSON into a temp directory,
        loads the JSON, and then convert into python dictionary.  This is the most straightforward way to grab all
        of the metadata due to poor pycromanager API / Java layer interaction.

        Returns
        -------

        """

        self.temp_path = os.path.join(self.temp_directory, 'meta.json')
        self.java_path = self.bridge.construct_java_object('java.io.File', args=[self.temp_path])
        PropertyMap = self.summary_metadata.toPropertyMap()
        PropertyMap.saveJSON(self.java_path, True, False)

        f = open(self.temp_path)
        dict_ = json.load(f)
        f.close()

        self.metadata['Summary'] = dict_
        self.metadata['ImagePlaneMetadata'] = dict()

    def _generate_plane_metadata(self, image):
        """
        generates the img plane metadata by saving the existing java PropertyMap as JSON into a temp directory,
        loads the JSON, and then convert into python dictionary.  This is the most straightforward way to grab all
        of the metadata due to poor pycromanager API / Java layer interaction.

        This image-plane data houses information of the config when the image was acquired.

        Parameters
        ----------
        image:          (pycromanager-object) MM Image Object at specific coordinate which houses data/metadata.

        Returns
        -------
        image_metadata:     (dict) Dictionary of the image-plane metadata

        """

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
        """
        Uses the coordinate builder to construct MM compatible coordinates, which get passed to the data provider
        in order to grab a specific MM image object.

        Parameters
        ----------
        coord:      (tuple) Coordinates of dimension (P, T, C, Z)

        Returns
        -------
        image_object:   (pycromanager-object) MM Image object at coordinate (P, T, C, Z)

        """
        self.CoordBuilder.p(coord[0])
        self.CoordBuilder.t(coord[1])
        self.CoordBuilder.c(coord[2])
        self.CoordBuilder.z(coord[3])
        mm_coord = self.CoordBuilder.build()

        return self.data_provider.getImage(mm_coord)

    def setup_zarr(self):
        """
        Initiates the zarr store.  Will create a zarr store with user-specified name or original name of data
        if not provided.  Store will contain a group called 'array' with contains an array of original
        data dtype of dimensions (P, T, C, Z, Y, X).

        Current compressor is Blosc zstd w/ bitshuffle (high compression, faster compression)

        Returns
        -------

        """

        src = os.path.join(self.save_directory, self.save_name if self.save_name else self.data_name)

        if not src.endswith('.zarr'):
            src += '.zarr'

        self.zarr_store = zarr.open(src)
        self.array = self.zarr_store.create('array',
                                            shape=(self.p if self.p != 0 else 1,
                                                   self.t if self.t != 0 else 1,
                                                   self.c if self.c != 0 else 1,
                                                   self.z if self.z != 0 else 1,
                                                   self.y,
                                                   self.x),
                                            chunks=(1, 1, 1, 1, self.y, self.x),
                                            compressor=Blosc('zstd', clevel=3, shuffle=Blosc.BITSHUFFLE),
                                            dtype=self.dtype)

    def run_conversion(self):
        """
        Runs the data conversion and performs a random image check to make sure conversion did not
        alter any data values.


        Returns
        -------

        """

        # Run setup
        print('Running Conversion...')
        self._generate_summary_metadata()
        self.coords = self._gen_coordset()
        self.setup_zarr()

        #Format bar for CLI display
        bar_format = 'Status: |{bar}|{n_fmt}/{total_fmt} (Time Remaining: {remaining}), {rate_fmt}{postfix}]'

        # Run through every coordinate and convert image + grab image metadata, statistics
        for coord in tqdm(self.coords, bar_format=bar_format):
            
            img = self.get_image_object(coord)
            
            self.metadata['ImagePlaneMetadata'][f'{coord}'] = self._generate_plane_metadata(img)
            img_raw = img.getRawPixels().reshape(self.y, self.x)
            self.array[coord[0], coord[1], coord[2], coord[3]] = img_raw

            # Statistics file can be used later for MD5 check sum
            self._save_image_stats(img_raw, coord)

        # Put metadata into zarr store and cleanup
        self.zarr_store.attrs.put(self.metadata)
        self.stats_file.close()
        shutil.rmtree(self.temp_directory)

        # Run Tests
        print('Running Tests...')
        total_images = self.p * self.t * self.c * self.z
        self.run_random_img_test(total_images//4) # test 25% of total images
        self.run_md5_check_sum_test()

    def run_md5_check_sum_test(self):

        zarr_path = os.path.join(self.save_directory, self.save_name)
        if not zarr_path.endswith('.zarr'): zarr_path += '.zarr'
        zarr_stats_path = gen_stats_file(zarr_path, self.save_directory)

        raw_md5 = md5(self.stats_path)
        converted_md5 = md5(zarr_stats_path)

        if raw_md5 != converted_md5:
            print('MD5 check sum failed.  Potential Error in Conversion')
        else:
            print('MD5 check sum passed. Conversion successful')


    def run_random_img_test(self, n_rounds=1):
        """
        Grab random image and check against saved zarr image.  If MSE between raw image and converted
        image != 0, conversion failed.

        Parameters
        ----------
        n_rounds:   (int) number of random images to check

        Returns
        -------

        """

        choices = np.arange(0, len(self.coords), dtype='int')
        failed = False
        for i in range(n_rounds):

            rand_int = np.random.choice(choices, replace=False)
            coord = self.coords[rand_int]
            image_object = self.get_image_object(coord)

            img_raw = image_object.getRawPixels().reshape(self.x, self.y)
            img_saved = self.array[coord[0], coord[1], coord[2], coord[3]]

            mse = ((img_raw - img_saved)**2).mean(axis=None)
            if mse != 0:
                failed = True

        if failed:
            print(f'Images do not match. Conversion Failed. DO NOT DELETE ORIGINAL DATA')
        else:
            print(f'Random Image Check Passed. Conversion Successful')














































