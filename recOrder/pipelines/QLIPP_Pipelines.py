from recOrder.io.config_reader import ConfigReader
from waveorder.io.reader import MicromanagerReader
from waveorder.io.writer import WaveorderWriter
from recOrder.io.utils import load_bg
from recOrder.compute.QLIPP_compute import initialize_reconstructor, reconstruct_QLIPP_birefringence
import json
import numpy as np
import time

class qlipp_pipeline_constructor:

    def __init__(self, config: ConfigReader, data: MicromanagerReader, sample: str):

        if config.default == 'QLIPP_3D':
            self.reconstructor = qlipp_3D_pipeline(config, data, sample)
        elif config.default == 'None':
            pass
            # self.reconstructor == 'Custom'

    def run_reconstruction(self):
        self.reconstructor.reconstruct()


class qlipp_3D_pipeline:

    def __init__(self, config, data: MicromanagerReader, sample: str):
        self.config = config
        self.data = data
        self.calib_meta = json.load(open(self.config.calibration_metadata))
        self.sample = sample

        #TODO: Parse positions if not 'all', parse timepoints if not 'all'
        self.pos = data.get_num_positions() if self.config.positions == 'all' else self.config.positions
        self.t = data.frames if self.config.timepoints == 'all' else None

        self.channels = self.config.output_channels
        self.chan_names = self.data.channel_names
        self.bg_path = self.config.background
        self.bg_roi = self.config.background_ROI
        self.bg_correction = self.config.background_correction
        self.img_dim = (self.data.height, self.data.width, self.data.slices)

        # self.bg_data = load_bg(self.bg_path, self.img_dim[0], self.img_dim[1], self.bg_roi)
        # self.reconstructor = initialize_reconstructor(self.img_dim, self.config.wavelength, self.config.swing)

        if self.data.channels < 4:
            raise ValueError(f'Number of Channels is {data.channels}, cannot be less than 4')

    def reconstruct(self):

        bg_data = load_bg(self.bg_path, self.img_dim[0], self.img_dim[1], self.bg_roi)

        #TODO: read step size from metadata
        reconstructor = initialize_reconstructor((self.img_dim[0], self.img_dim[1]), self.config.wavelength,
                                                 self.calib_meta['Summary']['~ Swing (fraction)'],
                                                 len(self.calib_meta['Summary']['ChNames']),
                                                 self.config.NA_objective, self.config.NA_condenser,
                                                 self.config.magnification, self.img_dim[2], self.config.z_step,
                                                 self.config.pad_z, self.config.pixel_size,
                                                 self.config.background_correction, self.config.n_objective_media,
                                                 self.config.use_gpu, self.config.gpu_id)

        #TODO: Add check to make sure that State0..4 are the first 4 channels
        bg_stokes = reconstructor.Stokes_recon(bg_data)
        bg_stokes = reconstructor.Stokes_transform(bg_stokes)

        data_shape = (1, len(self.channels), self.img_dim[2], self.img_dim[0], self.img_dim[1])
        chunk_size = (1, 1, 1, self.img_dim[0], self.img_dim[1])

        writer = WaveorderWriter(self.config.processed_dir, 'physical')
        writer.create_zarr_root(f'{self.sample}.zarr')


        start_time = time.time()
        print(f'Beginning Reconstruction...')
        #TODO: write fluorescence data from remaining channels, need to get their c_idx
        for pos in range(self.pos):

            writer.create_position(pos)
            writer.init_array(data_shape, chunk_size, self.channels)

            if pos != 0:
                pos_tot_time = (pos_end_time-pos_start_time)/60
                total_time = pos_tot_time*self.pos
                remaining_time = total_time - pos*pos_tot_time
                print(f'Estimated Time Remaining: {np.round(remaining_time,0)} min')

            pos_start_time = time.time()
            for t in range(self.t):

                position = self.data.get_array(pos)
                recon_data = reconstruct_QLIPP_birefringence(position[t], reconstructor, bg_stokes)

                print(f'Reconstructing Position {pos}, Time {t}')
                time_start_time = time.time()

                if 'Phase3D' in self.channels:
                    phase3D = reconstructor.Phase_recon_3D(np.transpose(recon_data[2], (1, 2, 0)),
                                                           method=self.config.phase_denoiser_3D,
                                                           reg_re=self.config.Tik_ref_ph_3D, rho=self.config.rho_3D,
                                                           lambda_re=self.config.TV_ref_ph_3D, itr=self.config.itr_3D,
                                                           verbose=False)

                for chan in range(len(self.channels)):
                    if 'Retardance' in self.channels[chan]:
                        ret = recon_data[0] / (2 * np.pi) * self.config.wavelength
                        writer.write(ret, t=t, c=chan)

                    elif 'Orientation' in self.channels[chan]:
                        writer.write(recon_data[1], t=t, c=chan)
                    elif 'Brightfield' in self.channels[chan]:
                        writer.write(recon_data[2], t=t, c=chan)
                    elif 'Phase3D' in self.channels[chan]:
                        writer.write(np.transpose(phase3D, (2,0,1)), t=t, c=chan)
                    else:
                        #TODO: Add writing fluorescence
                        raise NotImplementedError(f'{self.channels[chan]} not available to write yet')
                time_end_time = time.time()
                print(f'Finished Reconstructing Position {pos}, Time {t} ({(time_end_time-time_start_time)/60} min)')

            pos_end_time = time.time()

    def add_denoising(self):
        pass


