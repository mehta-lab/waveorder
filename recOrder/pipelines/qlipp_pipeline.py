from recOrder.io.config_reader import ConfigReader
from waveorder.io.reader import WaveorderReader
from waveorder.io.writer import WaveorderWriter
from recOrder.io import MetadataReader
from recOrder.io.utils import load_bg, MockEmitter, rec_bkg_to_wo_bkg
from recOrder.compute.qlipp_compute import reconstruct_qlipp_birefringence, reconstruct_qlipp_stokes, \
    reconstruct_phase2D, reconstruct_phase3D, initialize_reconstructor
import numpy as np
from recOrder.pipelines.base import PipelineInterface


class QLIPP(PipelineInterface):

    """
    This class contains methods to reconstruct an entire dataset alongside pre/post-processing
    """

    def __init__(self, config: ConfigReader, data: WaveorderReader, writer: WaveorderWriter, mode: str, num_t: int, emitter=MockEmitter()):
        """
        Parameters
        ----------
        config:     (Object) initialized ConfigReader object
        data:       (Object) initialized WaveorderReader object (data should be extracted already)
        writer:     (Object) initialiazed WaveorderWriter object
        mode:       (str) mode of operation, can be '2D', '3D', or 'stokes'
        num_t:      (int) number of timepoints being analyzed
        """

        # Dataset Parameters
        self.config = config
        self.data = data
        self.writer = writer
        self.mode = mode

        # Emitter
        self.dimension_emitter = emitter

        # Dimension Parameters
        self.t = num_t
        self.output_channels = self.config.output_channels
        self._check_output_channels(self.output_channels)

        if self.data.channels < 4:
            raise ValueError(f'Number of Channels is {data.channels}, cannot be less than 4')

        self.slices = self.data.slices
        self.focus_slice = None
        if self.mode == '2D':
            self.slices = 1
            self.focus_slice = self.config.focus_zidx

        self.img_dim = (self.data.height, self.data.width, self.data.slices)

        # Metadata
        self.chan_names = self.data.channel_names
        self.bg_path = self.config.background if self.config.background else None
        if self.config.calibration_metadata:
            self.calib_meta = MetadataReader(self.config.calibration_metadata)
            self.calib_scheme = self.calib_meta.Calibration_scheme
            self.bg_roi = self.calib_meta.ROI # TODO: remove for 1.0.0
        else:
            self.calib_meta = None
            self.calib_scheme = '4-State'
            self.bg_roi = None # TODO: remove for 1.0.0

        # identify the image indicies corresponding to each polarization orientation
        self.s0_idx, self.s1_idx, \
        self.s2_idx, self.s3_idx, \
        self.s4_idx, self.fluor_idxs = self.parse_channel_idx(self.data.channel_names)

        # Writer Parameters
        self.data_shape = (self.t, len(self.output_channels), self.slices, self.img_dim[0], self.img_dim[1])
        self.chunk_size = (1, 1, 1, self.data_shape[-2], self.data_shape[-1])

        wo_background_correction = rec_bkg_to_wo_bkg(self.config.background_correction)

        # Initialize Reconstructor
        if self.no_phase:
            self.reconstructor = initialize_reconstructor(pipeline='birefringence',
                                                          image_dim=(self.img_dim[0], self.img_dim[1]),
                                                          wavelength_nm=self.config.wavelength,
                                                          swing=self.calib_meta.Swing,
                                                          calibration_scheme=self.calib_scheme,
                                                          n_slices=self.data.slices,
                                                          pad_z=self.config.pad_z,
                                                          bg_correction=wo_background_correction,
                                                          mode=self.mode,
                                                          use_gpu=self.config.use_gpu,
                                                          gpu_id=self.config.gpu_id)

        else:
            self.reconstructor = initialize_reconstructor(pipeline='QLIPP',
                                                          image_dim=(self.img_dim[0], self.img_dim[1]),
                                                          wavelength_nm=self.config.wavelength,
                                                          swing=self.calib_meta.Swing,
                                                          calibration_scheme=self.calib_scheme,
                                                          NA_obj=self.config.NA_objective,
                                                          NA_illu=self.config.NA_condenser,
                                                          n_obj_media=self.config.n_objective_media,
                                                          mag=self.config.magnification,
                                                          n_slices=self.data.slices,
                                                          z_step_um=self.data.z_step_size,
                                                          pad_z=self.config.pad_z,
                                                          pixel_size_um=self.config.pixel_size,
                                                          bg_correction=wo_background_correction,
                                                          mode=self.mode,
                                                          use_gpu=self.config.use_gpu,
                                                          gpu_id=self.config.gpu_id)

        # Prepare background corrections for waveorder
        if self.config.background_correction in ['global', 'local_fit+']:
            bg_data = load_bg(self.bg_path, self.img_dim[0], self.img_dim[1], self.bg_roi) # TODO: remove ROI for 1.0.0
            self.bg_stokes = self.reconstructor.Stokes_recon(bg_data)
            self.bg_stokes = self.reconstructor.Stokes_transform(self.bg_stokes)
        elif self.config.background_correction == 'local_fit':
            self.bg_stokes = np.zeros((5, self.img_dim[0], self.img_dim[1]))
            self.bg_stokes[0, ...] = 1  # Set background to "identity" Stokes parameters.
        else:
            self.bg_stokes = None

    def _check_output_channels(self, output_channels):
        self.no_birefringence = True
        self.no_phase = True
        for channel in output_channels:
            if 'Retardance' in channel or 'Orientation' in channel or 'Brightfield' in channel:
                self.no_birefringence = False
            if 'Phase3D' in channel or 'Phase2D' in channel:
                self.no_phase = False
            else:
                continue


    def reconstruct_stokes_volume(self, data):
        """
        This method reconstructs a stokes volume from raw data

        Parameters
        ----------
        data:           (nd-array) raw data volume at certain position, time.
                                  dimensions must be (C, Z, Y, X)

        Returns
        -------
        stokes:         (nd-array) stokes volume of dimensions (C, Z, Y, X) w/ C=5
                                    where C is the stokes channels (S0..S3 + DOP)

        """

        if self.calib_scheme == '4-State':
            LF_array = np.zeros([4, self.data.slices, self.data.height, self.data.width])

            LF_array[0] = data[self.s0_idx]
            LF_array[1] = data[self.s1_idx]
            LF_array[2] = data[self.s2_idx]
            LF_array[3] = data[self.s3_idx]

        elif self.calib_scheme == '5-State':
            LF_array = np.zeros([5, self.data.slices, self.data.height, self.data.width])
            LF_array[0] = data[self.s0_idx]
            LF_array[1] = data[self.s1_idx]
            LF_array[2] = data[self.s2_idx]
            LF_array[3] = data[self.s3_idx]
            LF_array[4] = data[self.s4_idx]

        else:
            raise NotImplementedError(f"calibration scheme {self.calib_scheme} not implemented")

        stokes = reconstruct_qlipp_stokes(LF_array, self.reconstructor, self.bg_stokes)

        return stokes

    def deconvolve_volume(self, stokes):
        """
        This method reconstructs a phase volume or 2D phase image given stokes stack

        Parameters
        ----------
        stokes:             (nd-array) stokes stack of (C, Y, X, Z) where C = stokes channel

        Returns
        -------
        phase3D:            (nd-array) 3D phase stack of (Z, Y, X)

        or

        phase2D:            (nd-array) 2D phase image of (Y, X)

        """
        phase2D = None
        phase3D = None

        if 'Phase3D' in self.output_channels:
            phase3D = reconstruct_phase3D(stokes[0], self.reconstructor, method=self.config.phase_denoiser_3D,
                                          reg_re=self.config.Tik_reg_ph_3D, rho=self.config.rho_3D,
                                          lambda_re=self.config.TV_reg_ph_3D, itr=self.config.itr_3D)

        if 'Phase2D' in self.output_channels:
            phase2D = reconstruct_phase2D(stokes[0], self.reconstructor, method=self.config.phase_denoiser_2D,
                                          reg_p=self.config.Tik_reg_ph_2D, rho=self.config.rho_2D,
                                          lambda_p=self.config.TV_reg_ph_2D, itr=self.config.itr_2D)

        return phase2D, phase3D

    def reconstruct_birefringence_volume(self, stokes):
        """
        This method reconstructs birefringence (Ret, Ori, BF, Pol)
        for given stokes

        Parameters
        ----------
        stokes:             (nd-array) stokes stack of (C, Y, X, Z) or (C, Y, X) where C = stokes channel

        Returns
        -------
        birefringence:       (nd-array) birefringence stack of (C, Z, Y, X) or (C, Y, X)
                                        where C = Retardance, Orientation, BF, Polarization

        """

        if self.no_birefringence:
            return None
        else:
            return reconstruct_qlipp_birefringence(
                                stokes[:, slice(None) if self.slices != 1 else self.focus_slice, :, :],
                                self.reconstructor)

    # todo: think about better way to write fluor/registered data?
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
        # stokes = np.transpose(stokes, (-1, -4, -3, -2)) if len(stokes.shape) == 4 else stokes
        fluor_idx = 0

        for chan in range(len(self.output_channels)):
            if 'Retardance' in self.output_channels[chan]:
                ret = birefringence[0] / (2 * np.pi) * self.config.wavelength
                self.writer.write(ret, p=p, t=t, c=chan, z=z)
            elif 'Orientation' in self.output_channels[chan]:
                self.writer.write(birefringence[1], p=p, t=t, c=chan, z=z)
            elif 'Brightfield' in self.output_channels[chan]:
                self.writer.write(birefringence[2], p=p, t=t, c=chan, z=z)
            elif 'Phase3D' in self.output_channels[chan]:
                self.writer.write(phase3D, p=p, t=t, c=chan, z=z)
            elif 'Phase2D' in self.output_channels[chan]:
                self.writer.write(phase2D, p=p, t=t, c=chan, z=z)
            elif 'S0' in self.output_channels[chan]:
                self.writer.write(stokes[0, slice_, :, :], p=p, t=t, c=chan, z=z)
            elif 'S1' in self.output_channels[chan]:
                self.writer.write(stokes[1, slice_, :, :], p=p, t=t, c=chan, z=z)
            elif 'S2' in self.output_channels[chan]:
                self.writer.write(stokes[2, slice_, :, :], p=p, t=t, c=chan, z=z)
            elif 'S3' in self.output_channels[chan]:
                self.writer.write(stokes[3, slice_, :, :], p=p, t=t, c=chan, z=z)

            # Assume any other output channel in config is fluorescence
            else:
                if self.config.postprocessing.registration_use or self.config.postprocessing.deconvolution_use:
                    self.writer.write(modified_fluor[fluor_idx][slice_], p=p, t=t, c=chan, z=z)
                    fluor_idx += 1
                else:
                    self.writer.write(pt_data[self.fluor_idxs[fluor_idx], slice_], p=p, t=t, c=chan, z=z)
                    fluor_idx += 1

            self.dimension_emitter.emit((p, t, chan))

    def parse_channel_idx(self, channel_list):
        """
        Parses the metadata to find which channel each state resides in.  Useful if the acquisition
        does not have the pol states adjacent to one another.

        Parameters
        ----------
        channel_list:       (list) List of strings corresponding to the channel names

        Returns
        -------
        s0_idx, s1_idx, s2_idx, s3_idx, s4_idx, fluor_idx:      (int) Index corresponding to where each state
                                                                        sits in the channel order

        """
        fluor_idx = []
        s0_idx = None
        s1_idx = None
        s2_idx = None
        s3_idx = None
        s4_idx = None
        if 'PolScope_Plugin_Version' in self.calib_meta.json_metadata['Summary']:
            open_pol = True
        else:
            open_pol = False

        for channel in range(len(channel_list)):
            if 'State0' in channel_list[channel]:
                s0_idx = channel
            elif 'State1' in channel_list[channel]:
                s1_idx = channel
            elif 'State2' in channel_list[channel]:
                s2_idx = channel
            elif 'State3' in channel_list[channel]:
                s3_idx = channel
            elif 'State4' in channel_list[channel]:
                s4_idx = channel
            else:
                fluor_idx.append(channel)

        if open_pol:
            s1_idx, s2_idx, s3_idx, s4_idx = s4_idx, s3_idx, s1_idx, s2_idx

        return s0_idx, s1_idx, s2_idx, s3_idx, s4_idx, fluor_idx


