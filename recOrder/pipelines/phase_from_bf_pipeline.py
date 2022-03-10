from recOrder.io.config_reader import ConfigReader
from waveorder.io.reader import WaveorderReader
from waveorder.io.writer import WaveorderWriter
from recOrder.io.utils import MockEmitter
from recOrder.compute.qlipp_compute import reconstruct_phase2D, reconstruct_phase3D, initialize_reconstructor
import numpy as np
from recOrder.pipelines.base import PipelineInterface

class PhaseFromBF(PipelineInterface):

    def __init__(self, config: ConfigReader, data: WaveorderReader, writer: WaveorderWriter, num_t: int, emitter=MockEmitter()):

        # Dataset Parameters
        self.config = config
        self.data = data
        self.writer = writer

        # Emitter
        self.dimension_emitter = emitter

        # Dimension Parameters
        self.t = num_t
        self.output_channels = self.config.output_channels
        self._check_output_channels(self.output_channels)
        self.mode = '2D' if 'Phase2D' in self.output_channels else '3D'
        self.bf_chan_idx = self.config.brightfield_channel_index
        self.fluor_idxs = []

        # Assume any other channel in the data is fluorescence
        for i in range(self.data.channels):
            if i != self.bf_chan_idx:
                self.fluor_idxs.append(i)

        self.slices = self.data.slices
        self.focus_slice = None

        if self.mode == '2D':
            self.slices = 1
            self.focus_slice = self.config.focus_zidx

        # Set image dimensions / writer parameters
        self.img_dim = (self.data.height, self.data.width, self.data.slices)
        self.data_shape = (self.t, len(self.output_channels), self.slices, self.img_dim[0], self.img_dim[1])
        self.chunk_size = (1, 1, 1, self.data_shape[-2], self.data_shape[-1])

        # Initialize Reconstructor
        self.reconstructor = initialize_reconstructor(pipeline='PhaseFromBF',
                                                      image_dim=(self.img_dim[0], self.img_dim[1]),
                                                      wavelength_nm=self.config.wavelength,
                                                      NA_obj=self.config.NA_objective,
                                                      NA_illu=self.config.NA_condenser,
                                                      n_obj_media=self.config.n_objective_media,
                                                      mag=self.config.magnification,
                                                      n_slices=self.data.slices,
                                                      z_step_um=self.data.z_step_size,
                                                      pad_z=self.config.pad_z,
                                                      pixel_size_um=self.config.pixel_size,
                                                      mode=self.mode,
                                                      use_gpu=self.config.use_gpu,
                                                      gpu_id=self.config.gpu_id)


    def _check_output_channels(self, output_channels):
        """
        Function to make sure that the correct output channels are specified.
        Does not allow for both 2D and 3D simultaneous reconstruction due to saving formats.

        Parameters
        ----------
        output_channels:        (list) List of str output channels

        Returns
        -------

        """

        for channel in output_channels:
            if 'Phase3D' in channel:
                continue
            elif 'Phase2D' in channel:
                continue
            elif 'Phase3D' in channel and 'Phase2D' in channel:
                raise KeyError('Simultaneous 2D and 3D phase reconstruction not supported')
            else:
                continue

    def reconstruct_stokes_volume(self, data):
        """
        dummy function that reshapes the data to the traditional "stokes volume" output.
        Used by the pipeline_manager and feeds into phase reconstruction

        Parameters
        ----------
        data:           (nd-array) raw data of dimensions (C, Z, Y, X)

        Returns
        -------
        data:           (nd-array) brightfield data of dimensions (Y, X, Z)

        """
        # return np.transpose(data[self.bf_chan_idx], (-2, -1, -3))
        return data[self.bf_chan_idx]

    def reconstruct_birefringence_volume(self, data):
        """
        dummy function to skip in pipeline_manager.
        No birefringence compute occuring in this pipeline

        """

        return data

    def deconvolve_volume(self, bf_data):
        """
        This method reconstructs a phase volume or 2D phase image given stokes stack

        Parameters
        ----------
        bf_data:             (nd-array) Brightfield stack of (Y, X, Z)

        Returns
        -------
        phase3D:            (nd-array) 3D phase stack of (Z, Y, X)

        or

        phase2D:            (nd-array) 2D phase image of (Y, X)

        """
        phase2D = None
        phase3D = None

        if 'Phase3D' in self.output_channels:
            phase3D = reconstruct_phase3D(bf_data, self.reconstructor, method=self.config.phase_denoiser_3D,
                                          reg_re=self.config.Tik_reg_ph_3D, rho=self.config.rho_3D,
                                          lambda_re=self.config.TV_reg_ph_3D, itr=self.config.itr_3D)

        if 'Phase2D' in self.output_channels:
            phase2D = reconstruct_phase2D(bf_data, self.reconstructor, method=self.config.phase_denoiser_2D,
                                          reg_p=self.config.Tik_reg_ph_2D, rho=self.config.rho_2D,
                                          lambda_p=self.config.TV_reg_ph_2D, itr=self.config.itr_2D)

        return phase2D, phase3D

    def write_data(self, p, t, pt_data, stokes, birefringence, phase2D, phase3D, modified_fluor):
        """
        This function will iteratively write the data into its proper position, time, channel, z index.
        If any fluorescence channel is specificed in the config, it will be written in the order in which it appears
        in the data.  Dimensions differ between data type to make compute easier with waveOrder backend.

        Parameters
        ----------
        p:                  (int) Index of the p position to write
        t:                  (int) Index of the t position to write
        pt_data:            (nd-array) raw data nd-array at p,t index with dimensions (C, Z, Y, X)
        stokes:             (nd-array) None or nd-array w/ dimensions (Z, C, Y, X)
        birefringence:      (nd-array) None or nd-array w/ dimensions (C, Z, Y, X)
        phase2D:            (nd-array) None or nd-array w/ dimensions (Y, X)
        phase3D:            (nd-array) None or nd-array w/ dimensions (Z, Y, X)
        modified_fluor:  (nd-array) None or nd-array w/ dimensions (C, Z, Y, X)

        Returns
        -------
        Writes a zarr array to to given save directory.

        """
        z = 0 if self.mode == '2D' else None
        slice_ = self.focus_slice if self.mode == '2D' else slice(None)
        fluor_idx = 0

        for chan in range(len(self.output_channels)):
            if 'Phase3D' in self.output_channels[chan]:
                self.writer.write(phase3D, p=p, t=t, c=chan, z=z)
            elif 'Phase2D' in self.output_channels[chan]:
                self.writer.write(phase2D, p=p, t=t, c=chan, z=z)

            # Assume any other output channel in config is fluorescence
            else:
                if self.config.postprocessing.registration_use or self.config.postprocessing.deconvolution_use:
                    self.writer.write(modified_fluor[fluor_idx][slice_], p=p, t=t, c=chan, z=z)
                    fluor_idx += 1
                else:
                    self.writer.write(pt_data[self.fluor_idxs[fluor_idx], slice_], p=p, t=t, c=chan, z=z)
                    fluor_idx += 1

            self.dimension_emitter.emit((p, t, chan))