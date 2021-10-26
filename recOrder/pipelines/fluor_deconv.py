from recOrder.pipelines.pipeline_interface import PipelineInterface


class FluorescenceDeconvolution(PipelineInterface):

    def __init__(self, config, data, writer, num_t):

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

        # Dimension Parameters
        self.t = num_t
        self.output_channels = self.config.output_channels
        self._check_output_channels(self.output_channels)

        if self.data.channels < 4:
            raise ValueError(f'Number of Channels is {data.channels}, cannot be less than 4')

        self.slices = self.data.slices

        self.img_dim = (self.data.height, self.data.width, self.data.slices)

        # Metadata
        self.chan_names = self.data.channel_names

        # identify the image indicies corresponding to each polarization orientation
        self.s0_idx, self.s1_idx, \
        self.s2_idx, self.s3_idx, \
        self.s4_idx, self.fluor_idxs = self.parse_channel_idx(self.data.channel_names)

        # Writer Parameters
        self.data_shape = (self.t, len(self.output_channels), self.slices, self.img_dim[0], self.img_dim[1])
        self.chunk_size = (1, 1, 1, self.data_shape[-2], self.data_shape[-1])

        # Initialize Reconstructor
        self.reconstructor = initialize_reconstructor(pipeline='QLIPP',
                                                      image_dim=(self.img_dim[0], self.img_dim[1]),
                                                      wavelength_nm=self.config.wavelength,
                                                      swing=self.calib_meta['Summary']['Swing (fraction)'],
                                                      calibration_scheme=self.calib_scheme,
                                                      NA_obj=self.config.NA_objective,
                                                      NA_illu=self.config.NA_condenser,
                                                      n_obj_media=self.config.n_objective_media,
                                                      mag=self.config.magnification,
                                                      n_slices=self.data.slices,
                                                      z_step_um=self.data.z_step_size,
                                                      pad_z=self.config.pad_z,
                                                      pixel_size_um=self.config.pixel_size,
                                                      bg_correction=self.config.background_correction,
                                                      mode=self.mode,
                                                      use_gpu=self.config.use_gpu,
                                                      gpu_id=self.config.gpu_id)

        # Compute BG stokes if necessary
        if self.config.background_correction != None:
            bg_data = load_bg(self.bg_path, self.img_dim[0], self.img_dim[1], self.bg_roi)
            self.bg_stokes = self.reconstructor.Stokes_recon(bg_data)
            self.bg_stokes = self.reconstructor.Stokes_transform(self.bg_stokes)