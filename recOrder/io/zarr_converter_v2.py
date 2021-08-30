from pycromanager import Bridge
import os
import zarr
from tqdm import tqdm
import json
import numpy as np
import tifffile as tiff
import shutil
from waveorder.io.writer import WaveorderWriter
from recOrder.preproc.pre_processing import get_autocontrast_limits

class ZarrConverter:

    def __init__(self, data_directory, save_directory, save_name=None):

        # Attempt MM Connections
        self._connect_and_setup_mm()

        # Init File IO Properties
        self.version = 'recOrder Converter version=0.1'
        self.data_directory = data_directory
        self.save_directory = save_directory
        self.data_name = self.summary_metadata.getPrefix()
        self.save_name = self.data_name if not save_name else save_name
        self.array = None
        self.zarr_store = None
        self.temp_directory = os.path.join(os.path.expanduser('~'), 'recOrder_temp')
        self.temp_path = None
        self.java_path = None
        if not os.path.exists(self.temp_directory):
            os.mkdir(self.temp_directory)

        # Generate Data Specific Properties
        self.coords = None
        self.dim_order = None
        self.p_dim = None
        self.t_dim = None
        self.c_dim = None
        self.z_dim = None
        self.dtype = self._get_dtype()
        self.p = self.data_provider.getMaxIndices().getP()+1
        self.t = self.data_provider.getMaxIndices().getT()+1
        self.c = self.data_provider.getMaxIndices().getC()+1
        self.z = self.data_provider.getMaxIndices().getZ()+1
        self.y = self.data_provider.getAnyImage().getHeight()
        self.x = self.data_provider.getAnyImage().getWidth()
        self.dim = (self.p, self.t, self.c, self.z, self.y, self.x)
        self.focus_z = self.z // 2
        print(f'Found Dataset {self.data_name} w/ dimensions (P, T, C, Z, Y, X): {self.dim}')

        # Initialize Coordinate Builder
        self.CoordBuilder = self.data_provider.getAnyImage().getCoords().copyBuilder()

        # Initialize Metadata Dictionary
        self.metadata = dict()
        self.metadata['recOrder_Converter_Version'] = self.version

        # Initialize writer
        self.writer = WaveorderWriter(self.save_directory, datatype='raw', silence=True)
        self.writer.create_zarr_root(self.save_name)

    def _gen_coordset(self):
        """
        generates a coordinate set in the dimensional order to which the data was acquired.
        This is important for keeping track of where we are in the tiff file during conversion

        Returns
        -------
        list(tuples) w/ length [N_images]

        """

        # 4 possible dimensions: p, c, t, z
        n_dim = 4
        hashmap = {'position': self.p,
                   'time': self.t,
                   'channel': self.c,
                   'z': self.z}

        self.dim_order = self.metadata['Summary']['map']['AxisOrder']['array']

        dims = []
        for i in range(n_dim):
            if i < len(self.dim_order):
                dims.append(hashmap[self.dim_order[i]])
            else:
                dims.append(1)

        # Reverse the dimension order for easier calling later
        self.dim_order.reverse()

        # return array of coordinate tuples with innermost dimension being the first dim acquired
        return [(dim3, dim2, dim1, dim0) for dim3 in range(dims[3]) for dim2 in range(dims[2])
                for dim1 in range(dims[1]) for dim0 in range(dims[0])]


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
        """
        gets the datatype from the raw data array

        Returns
        -------

        """

        dt = self.data_provider.getAnyImage().getRawPixels().dtype

        return dt.name

    def _preform_image_check(self, tiff_image, coord):
        """
        checks to make sure the memory mapped image matches the saved zarr image to ensure
        a successful conversion.

        Parameters
        ----------
        tiff_image:     (nd-array) memory mapped array
        coord:          (tuple) coordinate of the image location

        Returns
        -------
        True/False:     (bool) True if arrays are equal, false otherwise

        """

        zarr_array = self.writer.store[self.writer.get_current_group()]['raw_data']['array']
        zarr_img = zarr_array[coord[self.dim_order.index('time')],
                              coord[self.dim_order.index('channel')],
                              coord[self.dim_order.index('z')]]

        return np.array_equal(zarr_img, tiff_image)

    def _get_channel_names(self):
        """
        gets the chan names from the summary metadata (in order in which they were acquired)

        Returns
        -------

        """

        chan_names = self.metadata['Summary']['map']['ChNames']['array']

        return chan_names

    def check_file_update_page(self, last_file, current_file, current_page):
        """
        function to check whether or not the tiff page # should be incremented.  If it detects that the file
        has changed then reset the counter back to 0.

        Parameters
        ----------
        last_file:          (str) filename of the last file looked at
        current_file:       (str) filename of the current file
        current_page:       (int) current tiff page #

        Returns
        -------
        current page:       (int) updated page number

        """

        if last_file != current_file or not last_file:
            current_page = 0
        else:
            current_page += 1

        return current_page

    def get_image_array(self, data_file, current_page):
        """
        Grabs the image array through memory mapping.  We must first find the byte offset which is located in the
        tiff page tag.  We then use that to quickly grab the bytes corresponding to the desired image.

        Parameters
        ----------
        data_file:          (str) path of the data-file to look at
        current_page:       (int) current tiff page

        Returns
        -------
        array:              (nd-array) image array of shape (Y, X)

        """

        file = os.path.join(self.data_directory, data_file)
        tf = tiff.TiffFile(file)

        # get byte offset from tiff tag metadata
        byte_offset = self.get_byte_offset(tf, current_page)

        array = np.memmap(file, dtype=self.dtype, mode='r', offset=byte_offset, shape=(self.y, self.x))

        return array

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
        self.CoordBuilder.p(coord[self.dim_order.index('position')])
        self.CoordBuilder.t(coord[self.dim_order.index('time')])
        self.CoordBuilder.c(coord[self.dim_order.index('channel')])
        self.CoordBuilder.z(coord[self.dim_order.index('z')])
        mm_coord = self.CoordBuilder.build()

        return self.data_provider.getImage(mm_coord)


    def get_channel_clims(self):
        """
        generate contrast limits for each channel.  Grabs the middle image of the stack to compute contrast limits
        Default clim is to ignore 1% of pixels on either end

        Returns
        -------
        clims:      [list]: list of tuples corresponding to the (min, max) contrast limits

        """

        clims = []
        for chan in range(self.c):
            img = self.get_image_object((0, 0, chan, self.focus_z))
            clims.append(get_autocontrast_limits(img.getRawPixels().reshape(self.y, self.x)))

        return clims

    def get_byte_offset(self, tiff_file, page):
        """
        Gets the byte offset from the tiff tag metadata

        Parameters
        ----------
        tiff_file:          (Tiff-File object) Opened tiff file
        page:               (int) Page to look at for the tag

        Returns
        -------
        byte offset:        (int) byte offset for the image array

        """

        for tag in tiff_file.pages[page].tags.values():
            if 'StripOffset' in tag.name:
                return tag.value[0]
            else:
                continue

    def init_zarr_structure(self):
        """
        Initiates the zarr store.  Will create a zarr store with user-specified name or original name of data
        if not provided.  Store will contain a group called 'array' with contains an array of original
        data dtype of dimensions (T, C, Z, Y, X).  Appends OME-zarr metadata with clims,chan_names

        Current compressor is Blosc zstd w/ bitshuffle (high compression, faster compression)

        Returns
        -------

        """

        clims = self.get_channel_clims()
        chan_names = self._get_channel_names()

        for pos in range(self.p):
            self.writer.create_position(pos)
            self.writer.init_array(data_shape=(self.t if self.t != 0 else 1,
                                               self.c if self.c != 0 else 1,
                                               self.z if self.z != 0 else 1,
                                               self.y,
                                               self.x),
                                   chunk_size=(1, 1, 1, self.y, self.x),
                                   chan_names=chan_names,
                                   clims=clims,
                                   dtype=self.dtype)

    def run_conversion(self):
        """
        Runs the data conversion through memory mapping and performs an image check to make sure conversion did not
        alter any data values.

        Returns
        -------

        """

        # Run setup
        print('Running Conversion...')
        self._generate_summary_metadata()
        self.coords = self._gen_coordset()
        self.init_zarr_structure()
        self.writer.open_position(0)
        current_pos = 0
        current_page = 0
        last_file = None
        p_dim = self.dim_order.index('position')

        #Format bar for CLI display
        bar_format = 'Status: |{bar}|{n_fmt}/{total_fmt} (Time Remaining: {remaining}), {rate_fmt}{postfix}]'

        # Run through every coordinate and convert image + grab image metadata, statistics
        for coord in tqdm(self.coords, bar_format=bar_format):

            # Open the new position if the position index has changed
            if current_pos != coord[p_dim]:
                self.writer.open_position(coord[p_dim])
                current_pos = coord[p_dim]

            # get the image object
            img = self.get_image_object(coord)

            # Get the metadata
            self.metadata['ImagePlaneMetadata'][f'{coord}'] = self._generate_plane_metadata(img)
            data_file = self.metadata['ImagePlaneMetadata'][f'{coord}']['map']['FileName']['scalar']

            # get the memory mapped image
            img_raw = self.get_image_array(data_file, current_page)

            # Write the data
            self.writer.write(img_raw, coord[self.dim_order.index('time')],
                              coord[self.dim_order.index('channel')],
                              coord[self.dim_order.index('z')])

            # Perform image check
            if not self._preform_image_check(img_raw, coord):
                raise ValueError('Converted zarr image does not match the raw data. Conversion Failed')

            # Update current file and page
            current_page = self.check_file_update_page(last_file, data_file, current_page)
            last_file = data_file

        # Put metadata into zarr store and cleanup
        self.writer.store.attrs.put(self.metadata)
        shutil.rmtree(self.temp_directory)

    def run_random_img_test(self, n_images=1):
        """
        Grab random image and check against saved zarr image.  If MSE between raw image and converted
        image != 0, conversion failed.

        Parameters
        ----------
        n_images:   (int) number of random images to check

        Returns
        -------

        """

        choices = np.arange(0, len(self.coords), dtype='int')
        failed = False
        for i in range(n_images):

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














































