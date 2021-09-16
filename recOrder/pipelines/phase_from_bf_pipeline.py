from recOrder.io.config_reader import ConfigReader
from waveorder.io.reader import MicromanagerReader
from waveorder.io.writer import WaveorderWriter
from recOrder.io.utils import load_bg
from recOrder.compute.qlipp_compute import reconstruct_qlipp_birefringence, reconstruct_qlipp_stokes, \
    reconstruct_qlipp_phase2D, reconstruct_qlipp_phase3D, initialize_reconstructor
import json
import numpy as np
from recOrder.pipelines.pipeline_interface import PipelineInterface

class PhaseFromBF(PipelineInterface):

    def __init__(self, config: ConfigReader, data: MicromanagerReader, save_dir: str, name: str,
                 num_t: int, use_hcs: bool):

        # Dataset Parameters
        self.config = config
        self.data = data
        self.name = name
        self.save_dir = save_dir
        self.use_hcs = use_hcs

        # Dimension Parameters
        self.t = num_t
        self.output_channels = self.config.output_channels
        self._check_output_channels(self.output_channels)
        self.mode = '2D' if 'Phase2D' in self.output_channels else '3D'
        self.bf_chan_idx = self.config.BF_chan_idx

        self.slices = self.data.slices
        self.focus_slice = None

        if self.mode == '2D':
            self.slices = 1
            self.focus_slice = self.config.focus_zidx

        self.img_dim = (self.data.height, self.data.width, self.data.slices)

        # Writer Parameters
        self._file_writer = None
        self.data_shape = (self.t, len(self.output_channels), self.slices, self.img_dim[0], self.img_dim[1])
        self.chunk_size = (1, 1, 1, self.img_dim[0], self.img_dim[1])

        if self.use_hcs:
            hcs_meta = self.data.hcs_meta
        else:
            hcs_meta = None

        self.writer = WaveorderWriter(self.save_dir, hcs=self.use_hcs, hcs_meta=hcs_meta, verbose=True)
        self.writer.create_zarr_root(f'{self.name}.zarr')
        existing_meta = self.writer.store.attrs.asdict().copy()
        existing_meta['Config'] = self.config.yaml_dict
        self.writer.store.attrs.put(existing_meta)

        # Initialize Reconstructor
        self.reconstructor = initialize_reconstructor((self.img_dim[0], self.img_dim[1]), self.config.wavelength,
                                                      0, 1, False, self.config.NA_objective, self.config.NA_condenser,
                                                      self.config.magnification, self.data.slices,
                                                      self.data.z_step_size, self.config.pad_z, self.config.pixel_size,
                                                      self.config.background_correction, self.config.n_objective_media,
                                                      self.mode, self.config.use_gpu, self.config.gpu_id)


    def _check_output_channels(self, output_channels):

        for channel in output_channels:
            if 'Phase3D' in channel:
                continue
            elif 'Phase2D' in channel:
                continue
            elif 'Phase3D' in channel and 'Phase2D' in channel:
                raise KeyError('Simultaneous 2D and 3D phase reconstruction not supported')
            else:
                raise KeyError(f'Output channel "{channel}" not permitted')

    def reconstruct_phase_volume(self, stokes):
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
            phase3D = reconstruct_qlipp_phase3D(stokes[0], self.reconstructor, method=self.config.phase_denoiser_3D,
                                                reg_re=self.config.Tik_reg_ph_3D, rho=self.config.rho_3D,
                                                lambda_re=self.config.TV_reg_ph_3D, itr=self.config.itr_3D)

        if 'Phase2D' in self.output_channels:
            phase2D = reconstruct_qlipp_phase2D(stokes[0], self.reconstructor, method=self.config.phase_denoiser_2D,
                                                reg_p=self.config.Tik_reg_ph_2D, rho=self.config.rho_2D,
                                                lambda_p=self.config.TV_reg_ph_2D, itr=self.config.itr_2D)

        return phase2D, phase3D

    # todo: think about better way to write fluor/registered data?
    def write_data(self, pt, pt_data, stokes, birefringence, phase2D, phase3D, registered_stacks):
        """
        This function will iteratively write the data into its proper position, time, channel, z index.
        If any fluorescence channel is specificed in the config, it will be written in the order in which it appears
        in the data.  Dimensions differ between data type to make compute easier with waveOrder backend.

        Parameters
        ----------
        pt:                 (tuple) tuple containing position and time indicies.
        pt_data:            (nd-array) raw data nd-array at p,t index with dimensions (C, Z, Y, X)
        stokes:             (nd-array) None or nd-array w/ dimensions (Z, C, Y, X)
        birefringence:      (nd-array) None or nd-array w/ dimensions (C, Z, Y, X)
        phase2D:            (nd-array) None or nd-array w/ dimensions (Y, X)
        phase3D:            (nd-array) None or nd-array w/ dimensions (Z, Y, X)
        registered_stacks:  (nd-array) None or nd-array w/ dimensions (C, Z, Y, X)

        Returns
        -------
        Writes a zarr array to to given save directory.

        """

        t = pt[1]
        z = 0 if self.mode == '2D' else None
        slice_ = self.focus_slice if self.mode == '2D' else slice(None)
        stokes = np.transpose(stokes, (3, 0, 1, 2)) if len(stokes.shape) == 4 else stokes
        fluor_idx = 0

        for chan in range(len(self.output_channels)):
            if 'Retardance' in self.output_channels[chan]:
                ret = birefringence[0] / (2 * np.pi) * self.config.wavelength
                self.writer.write(ret, t=t, c=chan, z=z)
            elif 'Orientation' in self.output_channels[chan]:
                self.writer.write(birefringence[1], t=t, c=chan, z=z)
            elif 'Brightfield' in self.output_channels[chan]:
                self.writer.write(birefringence[2], t=t, c=chan, z=z)
            elif 'Phase3D' in self.output_channels[chan]:
                self.writer.write(phase3D, t=t, c=chan, z=z)
            elif 'Phase2D' in self.output_channels:
                self.writer.write(phase2D, t=t, c=chan, z=z)
            elif 'S0' in self.output_channels[chan]:
                self.writer.write(stokes[slice_, 0, :, :], t=t, c=chan, z=z)
            elif 'S1' in self.output_channels[chan]:
                self.writer.write(stokes[slice_, 1, :, :], t=t, c=chan, z=z)
            elif 'S2' in self.output_channels[chan]:
                self.writer.write(stokes[slice_, 2, :, :], t=t, c=chan, z=z)
            elif 'S3' in self.output_channels[chan]:
                self.writer.write(stokes[slice_, 3, :, :], t=t, c=chan, z=z)

            # Assume any other output channel in config is fluorescence
            else:
                if self.config.postprocessing.registration_use:
                    self.writer.write(registered_stacks[fluor_idx], t=t, c=chan, z=z)
                    fluor_idx += 1
                else:
                    self.writer.write(pt_data[self.fluor_idxs[fluor_idx]], t=t, c=chan, z=z)
                    fluor_idx += 1

    def reconstruct_stokes_volume(self, data):
        return data[self.bf_chan_idx]

    def reconstruct_birefringence_volume(self, data):
        return data

    #TODO: Finish up dummy functions so pipeline runs, think about how to feed data into reconstruct stokes

