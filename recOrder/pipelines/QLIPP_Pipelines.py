from recOrder.io.config_reader import ConfigReader
from waveorder.io.reader import MicromanagerReader
from waveorder.io.writer import WaveorderWriter
from recOrder.io.utils import load_bg
from recOrder.compute.QLIPP_compute import *
import json
import numpy as np
from recOrder.pipelines.Pipeline_ABC import Pipeline_Builder



class qlipp_pipeline(Pipeline_Builder):

    """
    This class contains methods to reconstruct an entire dataset alongside pre/post-processing
    """

    def __init__(self, config: ConfigReader, data: MicromanagerReader, save_dir: str, name: str, mode: str):
        """
        Parameters
        ----------
        config:     (Object) initialized ConfigReader object
        data:       (Object) initialized MicromanagerReader object (data should be extracted already)
        save_dir:   (str) save directory
        name:       (str) name of the sample to pass for naming of folders, etc.
        mode:       (str) mode of operation, can be '2D', '3D', or 'stokes'
        """

        # Dataset Parameters
        self.config = config
        self.data = data
        self.name = name
        self.mode = mode
        self.save_dir = save_dir

        # Dimension Parameters
        self.t = data.frames if self.config.timepoints == ['all'] else NotImplementedError
        self.channels = self.config.output_channels
        if self.data.channels < 4:
            raise ValueError(f'Number of Channels is {data.channels}, cannot be less than 4')

        self.slices = self.data.slices
        self.focus_slice = None
        if self.mode == '2D':
            self.slices = 1
            self.focus_slice = self.config.focus_zidx

        # Metadata
        self.chan_names = self.data.channel_names
        self.LF_indices = (self.parse_channel_idx(self.chan_names))
        self.calib_meta = json.load(open(self.config.calibration_metadata))
        self.bg_path = self.config.background
        self.bg_roi = self.calib_meta['Summary']['ROI Used (x ,y, width, height)']
        self.bg_correction = self.config.background_correction
        self.img_dim = (self.data.height, self.data.width, self.data.slices)
        self.s0_idx, self.s1_idx, self.s2_idx, self.s3_idx, self.fluor_idxs = self.parse_channel_idx(self.chan_names)

        # Writer Parameters
        self.data_shape = (self.t, len(self.channels), self.slices, self.img_dim[0], self.img_dim[1])
        self.chunk_size = (1, 1, 1, self.img_dim[0], self.img_dim[1])
        self.writer = WaveorderWriter(self.save_dir, 'physical' if mode != 'stokes' else 'stokes')
        self.writer.create_zarr_root(f'{self.name}.zarr')
        self.writer.store.attrs.put(self.config.config)

        #TODO: read step size from metadata

        # Initialize Reconstructor
        self.reconstructor = initialize_reconstructor((self.img_dim[0], self.img_dim[1]), self.config.wavelength,
                                                 self.calib_meta['Summary']['~ Swing (fraction)'],
                                                 len(self.calib_meta['Summary']['ChNames']),
                                                 self.config.NA_objective, self.config.NA_condenser,
                                                 self.config.magnification, self.data.slices, self.config.z_step,
                                                 self.config.pad_z, self.config.pixel_size,
                                                 self.config.background_correction, self.config.n_objective_media,
                                                 self.mode, self.config.use_gpu, self.config.gpu_id)

        # Compute BG stokes if necessary
        if self.bg_correction != None:
            bg_data = load_bg(self.bg_path, self.img_dim[0], self.img_dim[1], self.bg_roi)
            self.bg_stokes = self.reconstructor.Stokes_recon(bg_data)
            self.bg_stokes = self.reconstructor.Stokes_transform(self.bg_stokes)

    def reconstruct_stokes_volume(self, data):
        """
        This method reconstructs a stokes volume from raw data

        Parameters
        ----------
        data:           (nd-array) raw data volume at certain position, time.
                                  dimensions must be (C, Z, Y, X)

        Returns
        -------
        stokes:         (nd-array) stokes volume of dimensions (Z, 5, Y, X)
                                    where C is the stokes channels (S0..S3 + DOP)

        """


        LF_array = np.zeros([4, self.data.slices, self.data.height, self.data.width])

        LF_array[0] = data[self.LF_indices[0]]
        LF_array[1] = data[self.LF_indices[1]]
        LF_array[2] = data[self.LF_indices[2]]
        LF_array[3] = data[self.LF_indices[3]]

        stokes = reconstruct_QLIPP_stokes(LF_array, self.reconstructor, self.bg_stokes)

        return stokes

    def reconstruct_phase_volume(self, stokes):
        """
        This method reconstructs a phase volume or 2D phase image given stokes stack

        Parameters
        ----------
        stokes:             (nd-array) stokes stack of (Z, C, Y, X) where C = stokes channel

        Returns
        -------
        phase3D:            (nd-array) 3D phase stack of (Z, Y, X)

        or

        phase2D:            (nd-array) 2D phase image of (Y, X)

        """

        if 'Phase3D' in self.channels:
            phase3D = self.reconstructor.Phase_recon_3D(np.transpose(stokes[:, 0], (1, 2, 0)),
                                                   method=self.config.phase_denoiser_3D,
                                                   reg_re=self.config.Tik_reg_ph_3D, rho=self.config.rho_3D,
                                                   lambda_re=self.config.TV_reg_ph_3D, itr=self.config.itr_3D,
                                                   verbose=False)

            return np.transpose(phase3D, (2, 0, 1))

        if 'Phase2D' in self.channels:
            _, phase2D = self.reconstructor.Phase_recon(np.transpose(stokes[:, 0], (1, 2, 0)),
                                                        method=self.config.phase_denoiser_2D,
                                                        reg_u=self.config.Tik_reg_abs_2D,
                                                        reg_p=self.config.Tik_reg_ph_2D,
                                                        rho=self.config.rho_2D, lambda_u=self.config.TV_reg_abs_2D,
                                                        lambda_p=self.config.TV_reg_ph_2D, itr=self.config.itr_2D,
                                                        verbose=False)

            return phase2D

        else:
            return None


    def reconstruct_birefringence_volume(self, stokes):
        """
        This method reconstructs birefringence (Ret, Ori, BF, Pol)
        for given stokes

        Parameters
        ----------
        stokes:             (nd-array) stokes stack of (Z, C, Y, X) where C = stokes channel

        Returns
        -------
        birefringence:       (nd-array) birefringence stack of (C, Z, Y, X)
                                        where C = Retardance, Orientation, BF, Polarization

        """

        birefringence = reconstruct_QLIPP_birefringence(stokes[slice(None) if self.slices != 1 else self.focus_slice],
                                                        self.reconstructor)

        return birefringence

    def write_data(self, pt, pt_data, stokes, birefringence, phase, registered_stacks):

        t = pt[1]
        fluor_idx = 0

        for chan in range(len(self.channels)):
            if 'Retardance' in self.channels[chan]:
                ret = birefringence[0] / (2 * np.pi) * self.config.wavelength
                self.writer.write(ret, t=t, c=chan)
            elif 'Orientation' in self.channels[chan]:
                self.writer.write(birefringence[1], t=t, c=chan)
            elif 'Brightfield' in self.channels[chan]:
                self.writer.write(birefringence[2], t=t, c=chan)
            elif 'Phase3D' in self.channels[chan]:
                self.writer.write(phase, t=t, c=chan)
            elif 'S0' in self.channels[chan]:
                self.writer.write(stokes[:, 0], t=t, c=chan)
            elif 'S1' in self.channels[chan]:
                self.writer.write(stokes[:, 1], t=t, c=chan)
            elif 'S2' in self.channels[chan]:
                self.writer.write(stokes[:, 2], t=t, c=chan)
            elif 'S3' in self.channels[chan]:
                self.writer.write(stokes[:, 3], t=t, c=chan)
            else:
                if self.config.postprocessing.registration_use:
                    print('writing registered data')
                    self.writer.write(registered_stacks[fluor_idx], t=t, c=chan)
                    fluor_idx += 1
                else:
                    print('not writing registered data')
                    self.writer.write(pt_data[self.fluor_idxs[fluor_idx]], t=t, c=chan)
                    fluor_idx += 1

    #TODO: name fluor channels based off metadata name?
    #TODO: ADD CHECKS FOR OUTPUT CHANNELS AND NUMBER OF FLUOR
    def parse_channel_idx(self, channel_list):
        fluor_idx = []
        for channel in range(len(channel_list)):
            if 'State0' in channel_list[channel]:
                s0_idx = channel
            elif 'State1' in channel_list[channel]:
                s1_idx = channel
            elif 'State2' in channel_list[channel]:
                s2_idx = channel
            elif 'State3' in channel_list[channel]:
                s3_idx = channel
            else:
                fluor_idx.append(channel)

        return s0_idx, s1_idx, s2_idx, s3_idx, fluor_idx


