

class SignalManager:
    """
    manages signal connections between certain GUI elements and their corresponding functions

    """
    # todo: perhaps list function names and connections in a dictionary, similar to how configfile works
    #   then do assignment using setattr/getattr

    def __init__(self, module_type, module, funcs_module):
        if module_type == "offline":

            # OFFLINE RECONSTRUCTION TAB SIGNALS
            module.qbutton_browse_config_file.clicked[bool].connect(funcs_module.set_config_load_path)
            # module.le_path_to_config_savepath.editingFinished.connect(self._handle_save_config_path_changed)
            # module.qbutton_savepath_config_file.clicked[bool].connect(self.set_config_save_path)
            module.qbutton_loadconfig.clicked[bool].connect(funcs_module.load_configuration_file)
            module.qbutton_load_default_config.clicked[bool].connect(funcs_module.load_default_config)
            module.qbutton_save_config.clicked[bool].connect(funcs_module.save_configuration_file)

            module.qbutton_runReconstruction.clicked[bool].connect(funcs_module.run_reconstruction)
            module.qbutton_stopReconstruction.clicked[bool].connect(funcs_module.stop_reconstruction)

        elif module_type == "acquisition":
            # ONLINE RECON TAB SIGNALS
            module.qbutton_connect_to_mm.clicked[bool].connect(funcs_module._handle_mm2_connection)
            module.qButton_set_data_connection.clicked[bool].connect(funcs_module._handle_data_connection)
            # self.cb_connection_type.currentIndexChanged[int].connect(self._handle_data_connection)

            module.qbutton_snap_and_correct.clicked[bool].connect(funcs_module._handle_snap)
            module.qbutton_stop_monitor.clicked[bool].connect(funcs_module.stop_monitor)

            module.cb_background_method.currentIndexChanged[int].connect(funcs_module._handle_bg_method)
            module.qbutton_browse_bg_file.clicked[bool].connect(funcs_module.set_bg_file_path)

            module.le_transmission_scale.textChanged[str].connect(funcs_module.set_scales)
            module.le_retardance_scale.textChanged[str].connect(funcs_module.set_scales)
            module.le_orientation_scale.textChanged[str].connect(funcs_module.set_scales)
            module.le_polarization_scale.textChanged[str].connect(funcs_module.set_scales)

        elif module_type == "calibration":
            # CALIBRATION TAB SIGNALS
            module.qbutton_calibrate_lc.clicked[bool].connect(funcs_module.calibrate)
            module.le_swing.textChanged[str].connect(funcs_module.set_swing)
            module.le_wavelength.textChanged[str].connect(funcs_module.set_wavelength)

            module.qbutton_set_state0.clicked.connect(funcs_module._handle_set_state)
            module.qbutton_set_state1.clicked.connect(funcs_module._handle_set_state)
            module.qbutton_set_state2.clicked.connect(funcs_module._handle_set_state)
            module.qbutton_set_state3.clicked.connect(funcs_module._handle_set_state)
            module.qbutton_set_state4.clicked.connect(funcs_module._handle_set_state)
            module.qbutton_set_current.clicked.connect(funcs_module._handle_set_state)

        elif module_type == 'combined':
            pass

        else:
            raise NotImplementedError()
