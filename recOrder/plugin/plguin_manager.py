
# each of these dictionaries contains mappings between pyqt widget names, action type, and connecting function name
# {key: value} = {pyqt_widget_name : [action_type, function_name]}

OFFLINE = \
    {
        'qbutton_browse_config_file':   ['clicked', 'set_config_load_path'],
        'qbutton_loadconfig':           ['clicked', 'load_configuration_file'],
        'qbutton_load_default_config':  ['clicked', 'load_default_config'],
        'qbutton_save_config':          ['clicked', 'save_configuration_file'],
        'qbutton_runReconstruction':    ['clicked', 'run_reconstruction'],
        'qbutton_stopReconstruction':   ['clicked', 'stop_reconstruction'],
    }


class SignalManager:
    """
    manages signal connections between certain GUI elements and their corresponding functions

    """
    # todo: perhaps list function names and connections in a dictionary, similar to how configfile works
    #   then do assignment using setattr/getattr

    def __init__(self, module_type, module, funcs_module):
        if module_type == "offline":

            # OFFLINE RECONSTRUCTION TAB SIGNALS
            for widget in OFFLINE.items():
                m = getattr(module, widget[0])
                m_action = getattr(m, widget[1][0])
                m_action.connect(getattr(funcs_module, widget[1][1]))

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
