from recOrder.plugin.qtdesigner.recOrder_calibration_v4 import Ui_Form
from recOrder.plugin.calibration.calibration_module import CalibrationFunctions

# each of these dictionaries contains mappings between pyqt widget names, action type, and connecting function name
# {key: value} = {pyqt_widget_name : [action_type, function_name]}

# OFFLINE = \
#     {
#         'qbutton_browse_config_file':   ['clicked', 'set_config_load_path'],
#         'qbutton_loadconfig':           ['clicked', 'load_configuration_file'],
#         'qbutton_load_default_config':  ['clicked', 'load_default_config'],
#         'qbutton_save_config':          ['clicked', 'save_configuration_file'],
#         'qbutton_runReconstruction':    ['clicked', 'run_reconstruction'],
#         'qbutton_stopReconstruction':   ['clicked', 'stop_reconstruction'],
#     }

CALIBRATION_RECEIVERS = {

    # Connect to Micromanager
    'qbutton_mm_connect':           ['clicked', 'connect_to_mm'],
    # 'le_mm_status':                 ['mm_status_changed', 'handle_mm_status'],

    # Calibration Parameters
    # 'qbutton_browse':               ['clicked', 'browse_dir_path'],
    # 'chb_use_roi':                  ['clicked', 'set_calib_roi'],
    # 'le_directory':                 ['currentTextChanged', 'set_dir_path'],
    # 'le_swing':                     ['currentTextChanged', 'set_swing'],
    # 'le_wavelength':                ['currentTextChanged', 'set_wavelength'],
    # 'cb_calib_scheme':              ['currentIndexChanged', 'set_calib_scheme'],
    #
    # # Run Calibration
    # 'qbutton_calibrate':            ['clicked', 'run_calibration']
    # 'progress_bar':                 ['progress_value_changed', 'handle_progress_update'],
    # 'le_extinction':                ['extinction_value_changed', 'handle_extinction_update'],

    # Calibration Feedback
}

CALIBRATION_EMITTERS = {

    # Connect to Micromanager
    'le_mm_status':                 ['mm_status_changed', 'handle_mm_status_update'],

    # Calibration Parameters

    # Run Calibration
    # 'progress_bar':                 ['progress_value_changed', 'handle_progress_update'],
    # 'le_extinction':                ['extinction_value_changed', 'handle_extinction_update']

    # Calibration Feedback
}


class SignalManager:
    """
    manages signal connections between certain GUI elements and their corresponding functions

    """
    # todo: perhaps list function names and connections in a dictionary, similar to how configfile works
    #   then do assignment using setattr/getattr

    def __init__(self, module: Ui_Form, funcs_module: CalibrationFunctions):


        module.qbutton_mm_connect.clicked[bool].connect(funcs_module.connect_to_mm)
        funcs_module.mm_status_changed.connect(funcs_module._handle_mm_status_update)

        # elif module_type == "calibration":
        #     # CALIBRATION TAB SIGNALS
        #     module.qbutton_calibrate_lc.clicked[bool].connect(funcs_module.calibrate)
        #     module.le_swing.textChanged[str].connect(funcs_module.set_swing)
        #     module.le_wavelength.textChanged[str].connect(funcs_module.set_wavelength)
        #
        #     module.qbutton_set_state0.clicked.connect(funcs_module._handle_set_state)
        #     module.qbutton_set_state1.clicked.connect(funcs_module._handle_set_state)
        #     module.qbutton_set_state2.clicked.connect(funcs_module._handle_set_state)
        #     module.qbutton_set_state3.clicked.connect(funcs_module._handle_set_state)
        #     module.qbutton_set_state4.clicked.connect(funcs_module._handle_set_state)
        #     module.qbutton_set_current.clicked.connect(funcs_module._handle_set_state)
        #
        # elif module_type == 'combined':
        #     pass

