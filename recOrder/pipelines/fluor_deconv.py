from recOrder.pipelines.base import PipelineInterface
from recOrder.compute.fluorescence_deconvolution import initialize_fluorescence_reconstructor, \
    deconvolve_fluorescence_2D, deconvolve_fluorescence_3D, calculate_background


class FluorescenceDeconvolution(PipelineInterface):

    def __init__(self, config, data, writer, mode, num_t):

        """
        Parameters
        ----------
        config:     (Object) initialized ConfigReader object
        data:       (Object) initialized WaveorderReader object (data should be extracted already)
        writer:     (Object) initialiazed WaveorderWriter object
        num_t:      (int) number of timepoints being analyzed

        """

        # Dataset Parameters
        self.config = config
        self.data = data
        self.writer = writer
        self.mode = mode

        # Dimension Parameters
        self.t = num_t
        self.output_channels = self.config.output_channels
        self._check_output_channels(self.output_channels)

        # Metadata
        self.chan_names = self.data.channel_names

        if isinstance(self.config.fluor_channels, int):
            self.fluor_idxs = [self.config.fluorescence_channel_indices]
        else:
            self.fluor_idxs = self.config.fluorescence_channel_indices

        self.chan_names = []
        for i in self.fluor_idxs:
            self.chan_names.append(self.data.channel_names[i])

        # Writer Parameters
        self.img_dim = (len(self.fluor_idxs), self.data.slices, self.data.height, self.data.width)
        self.data_shape = (self.t, len(self.output_channels), self.data.slices, self.data.height, self.data.width)
        self.chunk_size = (1, 1, 1, self.data.height, self.data.width)

        # check to make sure parameters match data
        self._check_parameters(self.fluor_idxs, self.config.wavelength)

        self.map = {self.fluor_idxs[i]: i for i in range(len(self.fluor_idxs))}

        # Initialize Reconstructor
        self.reconstructor = initialize_fluorescence_reconstructor(img_dim=self.img_dim[1:],
                                                                   wavelength_nm=self.config.wavelength,
                                                                   pixel_size_um=self.config.pixel_size,
                                                                   z_step_um=self.data.z_step_size,
                                                                   NA_obj=self.config.NA_objective,
                                                                   mode=self.mode,
                                                                   n_obj_media=self.config.n_objective_media,
                                                                   pad_z=self.config.pad_z,
                                                                   use_gpu=self.config.use_gpu,
                                                                   gpu_id=self.config.gpu_id)

    def _check_output_channels(self, output_channels):

        idx_set = set()
        for idx in self.config.fluorescence_channel_indices:
            idx_set.add(idx)
        for idx in self.config.postprocessing.registration_channel_idx:
            idx_set.add(idx)

        if len(idx_set) != len(output_channels):
            raise ValueError('length of output channels does not equal the number of unique deconvolutions+registrations')

    def _check_parameters(self, fluor_idxs, wavelengths):

        if isinstance(fluor_idxs, list) and isinstance(wavelengths, list):
            if len(fluor_idxs) != len(wavelengths):
                raise ValueError('Config Error: number of fluorescent channels does not match number of wavelengths')
        elif isinstance(fluor_idxs, int) and isinstance(wavelengths, int):
            pass
        elif isinstance(fluor_idxs, int) and isinstance(wavelengths, float):
            pass
        else:
            raise ValueError('Config Error: number of fluorescent channels does not match number of wavelengths')

    def deconvolve_volume(self, data):

        bg_levels = []
        for volume in data:
            bg_levels.append(calculate_background(volume))

        if self.mode == '3D':
            deconvolved_data = deconvolve_fluorescence_3D(data,
                                                          self.reconstructor,
                                                          bg_level=bg_levels,
                                                          reg=self.config.reg)
        elif self.mode == '2D':
            deconvolved_data = deconvolve_fluorescence_2D(data,
                                                          self.reconstructor,
                                                          bg_level=bg_levels,
                                                          reg=self.config.reg)
        else:
            raise ValueError('reconstruction mode not understood')

        return deconvolved_data

    def write_data(self, p, t, pt_data, stokes, birefringence, deconvolve2D, deconvolve3D, registered_stacks):
        """
                This function will iteratively write the data into its proper position, time, channel, z index.
                If any fluorescence channel is specificed in the config, it will be written in the order in which it appears
                in the data.  Dimensions differ between data type to make compute easier with waveOrder backend.

                Parameters
                ----------
                p:                  (int) Index of the p position to write
                t:                  (int) Index of the t position to write
                pt_data:            (nd-array) raw data nd-array at p,t index with dimensions (C, Z, Y, X)
                stokes:             None
                birefringence:      None
                deconvolve2D:       (nd-array) None or nd-array w/ dimensions (N_wavelength, Y, X)
                deconvolve3D:       (nd-array) None or nd-array w/ dimensions (N_wavelength, Z, Y, X)
                registered_stacks:  (nd-array) None or nd-array w/ dimensions (C, Z, Y, X)

                Returns
                -------
                Writes a zarr array to to given save directory.

                """

        z = 0 if self.mode == '2D' else None

        if len(registered_stacks) > len(self.output_channels):
            raise IndexError('Registered stacks exceeds length of output channels')
        elif len(registered_stacks) == len(self.output_channels):
            for chan in range(len(self.output_channels)):
                self.writer.write(registered_stacks[chan], p=p, t=t, c=chan, z=z)

        # handles the case where we want to register both processed data + raw_data.
        # must be provided in the exact order, with processed data first,
        # in both pre_processing and processing output_channels
        else:
            mapping = dict((v, k) for k, v in self.map.items())
            reg_count = 0
            for chan in range(len(self.output_channels)):

                # check to see if the processing channel_idx is also in the registration index, if so, use
                # the registered stack (which should be the deconvolved + registered stack)
                if mapping[chan] in self.config.postprocessing.registration_channel_idx:
                    self.writer.write(registered_stacks[reg_count], p=p, t=t, c=chan, z=z)
                    reg_count += 1
                else:
                    if self.mode == '3D':
                        self.writer.write(deconvolve3D[chan], p=p, t=t, c=chan, z=z)
                    elif self.mode == '2D':
                        self.writer.write(deconvolve2D[chan], p=p, t=t, c=chan, z=z)
                    else:
                        raise ValueError('reconstruct mode during write not understood.')

    def reconstruct_stokes_volume(self, data):
        return data

    def reconstruct_birefringence_volume(self, stokes):
        return None

