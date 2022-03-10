from recOrder.pipelines.base import PipelineInterface
from recOrder.compute.fluorescence_compute import initialize_fluorescence_reconstructor, \
    deconvolve_fluorescence_2D, deconvolve_fluorescence_3D, calculate_background
import numpy as np
from recOrder.io.utils import MockEmitter


class FluorescenceDeconvolution(PipelineInterface):

    def __init__(self, config, data, writer, mode, num_t, emitter=MockEmitter()):

        """
        Parameters
        ----------
        config:     (Object) initialized ConfigReader object
        data:       (Object) initialized WaveorderReader object (data should be extracted already)
        writer:     (Object) initialiazed WaveorderWriter object
        mode:       (str) '2D' or '3D'
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
        self.dimension_emitter = emitter

        # check to make sure parameters match data and each other
        self._check_parameters()

        # Metadata

        self.chan_names = []
        for i in self.fluor_idxs:
            self.chan_names.append(self.data.channel_names[i])


        # Writer Parameters
        self.slices = 1 if self.mode == '2D' else self.data.slices
        self.img_dim = (self.data.height, self.data.width, self.data.slices)
        self.data_shape = (self.t, len(self.output_channels), self.slices, self.data.height, self.data.width)
        self.chunk_size = (1, 1, 1, self.data.height, self.data.width)

        self.map = {self.fluor_idxs[i]: i for i in range(len(self.fluor_idxs))}

        # Initialize Reconstructor
        self.reconstructor = initialize_fluorescence_reconstructor(img_dim=self.img_dim,
                                                                   wavelength_nm=self.config.wavelength,
                                                                   pixel_size_um=self.config.pixel_size,
                                                                   z_step_um=self.data.z_step_size,
                                                                   NA_obj=self.config.NA_objective,
                                                                   magnification=self.config.magnification,
                                                                   mode=self.mode,
                                                                   n_obj_media=self.config.n_objective_media,
                                                                   pad_z=self.config.pad_z,
                                                                   use_gpu=self.config.use_gpu,
                                                                   gpu_id=self.config.gpu_id)

    def _check_output_channels(self, output_channels):

        idx_set = set()
        for idx in self.config.fluorescence_channel_indices:
            idx_set.add(idx)
        if self.config.postprocessing.registration_use:
            for idx in self.config.postprocessing.registration_channel_idx:
                idx_set.add(idx)

        if len(idx_set) != len(output_channels):
            raise ValueError('length of output channels does not equal the number of unique deconvolutions+registrations')

    def _check_parameters(self):

        if self.mode == '2D':
            if self.config.focus_zidx is None:
                raise ValueError('Config Error: focus_zidx must be specified for 2D deconvolution')

            #TODO: potential error here if the user is doing registration of raw data + deconvolution
            #TODO: could cause length issues or mis-indexing
            if isinstance(self.config.fluorescence_channel_indices, int):
                self.focus_slice = [self.config.focus_zidx]*len(self.output_channels)
            else:
                if len(self.config.focus_zidx) != len(self.output_channels):
                    raise ValueError('Config Error: focus_zidx list must match length of output channels')
                else:
                    self.focus_slice = self.config.focus_zidx

        if isinstance(self.config.fluorescence_channel_indices, int):
            self.fluor_idxs = [self.config.fluorescence_channel_indices]
        else:
            self.fluor_idxs = self.config.fluorescence_channel_indices

        if isinstance(self.config.reg, int) or isinstance(self.config.reg, float):
            self.reg = [self.config.reg]*len(self.fluor_idxs)
        else:
            if len(self.config.reg) != len(self.fluor_idxs):
                raise ValueError('Config Error: reg must be a list the same length as fluor_channels')
            else:
                self.reg = self.config.reg

        if isinstance(self.config.fluorescence_background, int) or isinstance(self.config.reg, float):
            self.background = [self.config.fluorescence_background]*len(self.fluor_idxs)
        else:
            if len(self.config.fluorescence_background) != len(self.fluor_idxs):
                raise ValueError('Config Error: background values must be a list the same length as fluor_channels')
            else:
                self.background = self.config.fluorescence_background

        if isinstance(self.config.wavelength, int) or isinstance(self.config.wavelength, float):
            if len(self.fluor_idxs) != 1:
                raise ValueError('Config Error: Wavelengths must be a list if processing more than 1 fluor_channel')
            else:
                self.wavelength = [self.config.wavelength]
        else:
            if len(self.fluor_idxs) != len(self.config.wavelength):
                raise ValueError('Config Error: Wavelengths must be a list the same length as fluor_channels')
            else:
                self.wavelength = self.config.wavelength

    def deconvolve_volume(self, data):
        """
        takes an array of raw_data volumes and deconvolves them.

        Parameters
        ----------
        data:           (nd-array) raw_data of dimensions (C, Z, Y, X) w/ C=N_fluor

        Returns
        -------
        deconvolved2D/3D:   (nd-array) deconvolved data of dimensions (N_fluor, Z, Y, X) or (N_fluor, Y, X)
                                        if N_fluor == 1 then returns either (Z, Y, X) or (Y, X)
        """

        deconvolved3D = None
        deconvolved2D = None

        #todo: move transposing to compute function
        if self.mode == '3D':
            if not self.config.fluorescence_background:
                bg_levels = calculate_background(data[:, self.data.slices // 2])
            else:
                bg_levels = self.config.fluorescence_background
            deconvolved3D = deconvolve_fluorescence_3D(data,
                                                       self.reconstructor,
                                                       bg_level=bg_levels,
                                                       reg=self.reg)

        elif self.mode == '2D':
            if not self.config.fluorescence_background:
                bg_levels = calculate_background(data)
            else:
                bg_levels = self.config.fluorescence_background

            deconvolved2D = deconvolve_fluorescence_2D(data,
                                                       self.reconstructor,
                                                       bg_level=bg_levels,
                                                       reg=self.reg)

        else:
            raise ValueError('reconstruction mode not understood')

        return deconvolved2D, deconvolved3D


    def write_data(self, p, t, pt_data, stokes, birefringence, deconvolve2D, deconvolve3D, modified_fluor):
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
        modified_fluor:  (nd-array) None or nd-array w/ dimensions (C, Z, Y, X)

        Returns
        -------
        Writes a zarr array to to given save directory.

        """

        z = 0 if self.mode == '2D' else None
        deconvolve3D = deconvolve3D[np.newaxis, :, :, :] if deconvolve3D.ndim == 3 else deconvolve3D
        deconvolve2D = deconvolve2D[np.newaxis, :, :] if deconvolve2D.ndim == 2 else deconvolve2D

        if modified_fluor is None:
            for chan in range(len(self.output_channels)):
                if self.mode == '2D':
                    self.writer.write(deconvolve2D[chan], p=p, t=t, c=chan, z=z)
                elif self.mode == '3D':
                    self.writer.write(deconvolve3D[chan], p=p, t=t, c=chan, z=z)

                self.dimension_emitter.emit((p, t, chan))

        elif len(modified_fluor) > len(self.output_channels):
            raise IndexError('Registered stacks exceeds length of output channels')

        elif len(modified_fluor) == len(self.output_channels):
            for chan in range(len(self.output_channels)):
                self.writer.write(modified_fluor[chan], p=p, t=t, c=chan, z=z)

        # handles the case where we have registered both processed data + raw_data.
        # must be provided in the exact order, with processed data first,
        # in both post_processing and processing output_channels
        else:
            mapping = dict((v, k) for k, v in self.map.items())
            reg_count = 0
            for chan in range(len(self.output_channels)):

                # check to see if the processing channel_idx is also in the registration index, if so, use
                # the registered stack (which should be the deconvolved + registered stack)
                if mapping[chan] in self.config.postprocessing.registration_channel_idx:
                    self.writer.write(modified_fluor[reg_count], p=p, t=t, c=chan, z=z)
                    reg_count += 1
                else:
                    if self.mode == '3D':
                        self.writer.write(deconvolve3D[chan], p=p, t=t, c=chan, z=z)
                    elif self.mode == '2D':
                        self.writer.write(deconvolve2D[chan], p=p, t=t, c=chan, z=z)
                    else:
                        raise ValueError('reconstruct mode during write not understood.')

                self.dimension_emitter.emit((p, t, chan))

    def reconstruct_stokes_volume(self, data):
        """
        gathers raw data channels that are going to be deconvolved.

        Parameters
        ----------
        data:               (nd-array) raw data of dimensions (C, Z, Y, X)

        Returns
        -------
        collected_data:     (nd-array) collected raw data of dimensions (C, Z, Y, X) or (C, Y, X)
        """

        collected_data = []

        for val, idx in enumerate(self.fluor_idxs):
            if self.mode == '2D':
                collected_data.append(data[val, self.focus_slice[idx]])
            else:
                collected_data.append(data[val])

        return np.asarray(collected_data)

    def reconstruct_birefringence_volume(self, stokes):
        return None

