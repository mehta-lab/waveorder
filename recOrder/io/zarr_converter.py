import os
from tqdm import tqdm
import numpy as np
import tifffile as tiff
from waveorder.io.writer import WaveorderWriter
from recOrder.preproc.pre_processing import get_autocontrast_limits
import glob
from pathlib import Path


#TODO: All data HCS with grid some determined size
#TODO: Add position slider for our datasets
#TODO: Add catch for incomplete datasets (datasets stopped early)
class ZarrConverter:

    def __init__(self, input, output, append_position_names=False):
    def __init__(self, input, output, append_position_names=False):

        # Add Initial Checks
        if len(glob.glob(os.path.join(input, '*.ome.tif'))) == 0:
            raise ValueError('Specific input contains no ome.tif files, please specify a valid input directory')
        if not output.endswith('.zarr'):
            raise ValueError('Please specify .zarr at the end of your output')

        # Init File IO Properties
        self.version = 'recOrder Converter version=0.2'
        self.data_directory = input
        self.save_directory = os.path.dirname(output)
        self.files = glob.glob(os.path.join(self.data_directory, '*.ome.tif'))
        self.summary_metadata = self._generate_summary_metadata()
        self.save_name = os.path.basename(output)
        self.append_position_names = append_position_names
        self.array = None
        self.zarr_store = None

        if not os.path.exists(self.save_directory):
            os.mkdir(self.save_directory)

        # Generate Data Specific Properties
        self.coords = None
        self.coord_map = dict()
        self.pos_names = []
        self.dim_order = None
        self.p_dim = None
        self.t_dim = None
        self.c_dim = None
        self.z_dim = None
        self.dtype = self._get_dtype()
        self.p = self.summary_metadata['IntendedDimensions']['position']
        self.t = self.summary_metadata['IntendedDimensions']['time']
        self.c = self.summary_metadata['IntendedDimensions']['channel']
        self.z = self.summary_metadata['IntendedDimensions']['z']
        self.y = self.summary_metadata['Height']
        self.x = self.summary_metadata['Width']
        self.dim = (self.p, self.t, self.c, self.z, self.y, self.x)
        self.focus_z = self.z // 2
        self.prefix_list = []
        print(f'Found Dataset {self.save_name} w/ dimensions (P, T, C, Z, Y, X): {self.dim}')

        # Initialize Metadata Dictionary
        self.metadata = dict()
        self.metadata['recOrder_Converter_Version'] = self.version
        self.metadata['Summary'] = self.summary_metadata
        self.metadata['ImagePlaneMetadata'] = dict()

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

        self.dim_order = self.summary_metadata['AxisOrder']

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

    def _gather_index_maps(self):
        """
        Will return a dictionary of {coord: (filepath, page)} of length(N_Images) to later query

        Returns
        -------

        """

        self.p_dim = self.dim_order.index('position')
        self.t_dim = self.dim_order.index('time')
        self.c_dim = self.dim_order.index('channel')
        self.z_dim = self.dim_order.index('z')

        for file in self.files:
            tf = tiff.TiffFile(file)
            meta = tf.micromanager_metadata['IndexMap']

            for page in range(len(meta['Channel'])):
                coord = [0, 0, 0, 0]
                coord[self.p_dim] = meta['Position'][page]
                coord[self.t_dim] = meta['Frame'][page]
                coord[self.c_dim] = meta['Channel'][page]
                coord[self.z_dim] = meta['Slice'][page]

                self.coord_map[tuple(coord)] = (file, page)


    def _generate_summary_metadata(self):
        """
        generates the summary metadata by opening any file and loading the micromanager_metadata

        Returns
        -------
        summary_metadata:       (dict) MM Summary Metadata

        """

        tf = tiff.TiffFile(self.files[0])
        return tf.micromanager_metadata['Summary']

    def _generate_plane_metadata(self, tiff_file, page):
        """
        generates the img plane metadata by saving the MicroManagerMetadata written in the tiff tags.

        This image-plane data houses information of the config when the image was acquired.

        Parameters
        ----------
        tiff_file:          (TiffFile Object) Opened TiffFile Object
        page:               (int) Page corresponding to the desired image plane

        Returns
        -------
        image_metadata:     (dict) Dictionary of the image-plane metadata

        """

        for tag in tiff_file.pages[page].tags.values():
            if tag.name == 'MicroManagerMetadata':
                return tag.value
            else:
                continue

    def _get_dtype(self):
        """
        gets the datatype from any image plane metadata

        Returns
        -------

        """

        tf = tiff.TiffFile(self.files[0])

        return tf.pages[0].dtype

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

        chan_names = self.metadata['Summary']['ChNames']

        return chan_names

    def _get_position_names(self):
        """
        Append a list of pos_names in ascending order (order in which they were acquired)

        Returns
        -------

        """

        for p in range(self.p):
            if self.p > 1:
                name = self.metadata['Summary']['StagePositions'][p]['Label']
            else:
                name = ''
            self.pos_names.append(name)

    def check_file_changed(self, last_file, current_file):
        """
        function to check whether or not the tiff file has changed.

        Parameters
        ----------
        last_file:          (str) filename of the last file looked at
        current_file:       (str) filename of the current file

        Returns
        -------
        True/False:       (bool) updated page number

        """

        if last_file != current_file or not last_file:
            return True
        else:
            return False

    def get_image_array(self, coord, opened_tiff):
        """
        Grabs the image array through memory mapping.  We must first find the byte offset which is located in the
        tiff page tag.  We then use that to quickly grab the bytes corresponding to the desired image.

        Parameters
        ----------
        coord:              (tuple) coordinate map entry containing file / page info
        opened_tiff:        (TiffFile Object) current opened tiffile

        Returns
        -------
        array:              (nd-array) image array of shape (Y, X)

        """
        file = coord[0]
        page = coord[1]

        # get byte offset from tiff tag metadata
        byte_offset = self.get_byte_offset(opened_tiff, page)

        array = np.memmap(file, dtype=self.dtype, mode='r', offset=byte_offset, shape=(self.y, self.x))

        return array

    def get_channel_clims(self, pos):
        """
        generate contrast limits for each channel.  Grabs the middle image of the stack to compute contrast limits
        Default clim is to ignore 1% of pixels on either end

        Returns
        -------
        clims:      [list]: list of tuples corresponding to the (min, max) contrast limits

        """

        clims = []

        coord = list(self.coords[0])
        for chan in range(self.c):

            coord[self.p_dim] = pos
            coord[self.c_dim] = chan
            coord[self.z_dim] = self.focus_z

            fname = self.coord_map[tuple(coord)][0]
            tf = tiff.TiffFile(fname)

            img = self.get_image_array(self.coord_map[tuple(coord)], tf)
            clims.append(get_autocontrast_limits(img))

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

        Current compressor is Blosc zstd w/ bitshuffle (~1.5x compression, faster compared to best 1.6x compressor)

        Returns
        -------

        """


        chan_names = self._get_channel_names()
        self._get_position_names()

        for pos in range(self.p):

            clims = self.get_channel_clims(pos)
            prefix = self.pos_names[pos] if self.append_position_names else None
            self.writer.create_position(pos, prefix=prefix)
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
        print('Setting up zarr')
        self._generate_summary_metadata()
        self.coords = self._gen_coordset()
        self._gather_index_maps()
        self.init_zarr_structure()
        self.writer.open_position(0, prefix=self.prefix_list[0] if self.append_position_names else None)
        last_file = None
        current_pos = 0

        #Format bar for CLI display
        bar_format = 'Status: |{bar}|{n_fmt}/{total_fmt} (Time Remaining: {remaining}), {rate_fmt}{postfix}]'

        # Run through every coordinate and convert image + grab image metadata, statistics
        # loop is done in order in which the images were acquired
        print('Converting Images...')
        for coord in tqdm(self.coords, bar_format=bar_format):

            # re-order coordinates into zarr format
            coord_reorder = (coord[self.p_dim],
                             coord[self.t_dim],
                             coord[self.c_dim],
                             coord[self.z_dim])

            # Only load tiff file if it has changed from previous run
            current_file = self.coord_map[coord][0]
            if self.check_file_changed(last_file, current_file):
                tf = tiff.TiffFile(current_file)
                last_file = current_file

            # Get the metadata
            page = self.coord_map[coord][1]
            self.metadata['ImagePlaneMetadata'][f'{coord_reorder}'] = self._generate_plane_metadata(tf, page)

            # get the memory mapped image
            img_raw = self.get_image_array(self.coord_map[coord], tf)

            # Open the new position if the position index has changed
            if current_pos != coord[self.p_dim]:

                prefix = self.pos_names[coord[self.p_dim]] if self.append_position_names else None
                self.writer.open_position(coord[self.p_dim], prefix=prefix)
                current_pos = coord[self.p_dim]

            # Write the data
            self.writer.write(img_raw, coord[self.t_dim], coord[self.c_dim], coord[self.z_dim])

            # Perform image check
            if not self._preform_image_check(img_raw, coord):
                raise ValueError('Converted zarr image does not match the raw data. Conversion Failed')

        # Put metadata into zarr store and cleanup
        self.writer.store.attrs.put(self.metadata)














































