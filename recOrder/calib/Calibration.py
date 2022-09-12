import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tifffile as tiff
import time
from recOrder.io.core_functions import define_meadowlark_state, snap_image, set_lc_waves, set_lc_voltage, set_lc_daq, \
    set_lc_state, snap_and_average, snap_and_get_image, get_lc, define_config_state
from recOrder.calib.Optimization import BrentOptimizer, MinScalarOptimizer
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from napari.utils.notifications import show_warning
from scipy.interpolate import interp1d
from scipy.stats import linregress
from scipy.optimize import least_squares
import json
import os
import logging
import warnings
from recOrder.io.utils import MockEmitter
from datetime import datetime
from importlib_metadata import version

LC_DEVICE_NAME = 'MeadowlarkLcOpenSource'


class QLIPP_Calibration():

    def __init__(self, mmc, mm, group='Channel', lc_control_mode='MM-Retardance', interp_method='schnoor_fit',
                 wavelength=532, optimization='min_scalar', print_details=True):
        '''

        Parameters
        ----------
        mmc : object
            MicroManager core instance
        mm : object
            MicroManager Studio instance
        group : str
            Name of the MicroManager channel group used defining LC states [State0, State1, State2, ...]
        lc_control_mode : str
            Defined the control mode of the liquid crystals. One of the following:
            * MM-Retardance: The retardance of the LC is set directly through the MicroManager LC device adapter. The
            MicroManager device adapter determines the corresponding voltage which is sent to the LC.
            * MM-Voltage: The CalibrationData class in recOrder uses the LC calibration data to determine the correct
            LC voltage for a given retardance. The LC voltage is set through the MicroManager LC device adapter.
            * DAC: The CalibrationData class in recOrder uses the LC calibration data to determine the correct
            LC voltage for a given retardance. The voltage is applied to the IO port of the LC controller through the
            TriggerScope DAC outputs.
        interp_method : str
            Method of interpolating the LC retardance-to-voltage calibration curve. One of the following:
            * linear: linear interpolation of retardance as a function of voltage and wavelength
            * schnoor_fit: Schnoor fit interpolation as described in https://doi.org/10.1364/AO.408383
        wavelength : float
            Measurement wavelength
        optimization : str
            LC retardance optimization method, 'min_scalar' (default) or 'brent'
        print_details : bool
            Set verbose option
        '''

        # Micromanager API
        self.mm = mm
        self.mmc = mmc
        self.snap_manager = mm.getSnapLiveManager()

        # Meadowlark LC Device Adapter Property Names
        self.PROPERTIES = {'LCA': (LC_DEVICE_NAME, 'Retardance LC-A [in waves]'),
                           'LCB': (LC_DEVICE_NAME, 'Retardance LC-B [in waves]'),
                           'LCA-Voltage': (LC_DEVICE_NAME, 'Voltage (V) LC-A'),
                           'LCB-Voltage': (LC_DEVICE_NAME, 'Voltage (V) LC-B'),
                           'LCA-DAC': ('TS_DAC01', 'Volts'),
                           'LCB-DAC': ('TS_DAC02', 'Volts'),
                           'State0': (LC_DEVICE_NAME, 'Pal. elem. 00; enter 0 to define; 1 to activate'),
                           'State1': (LC_DEVICE_NAME, 'Pal. elem. 01; enter 0 to define; 1 to activate'),
                           'State2': (LC_DEVICE_NAME, 'Pal. elem. 02; enter 0 to define; 1 to activate'),
                           'State3': (LC_DEVICE_NAME, 'Pal. elem. 03; enter 0 to define; 1 to activate'),
                           'State4': (LC_DEVICE_NAME, 'Pal. elem. 04; enter 0 to define; 1 to activate')
                           }
        self.group = group

        # GUI Emitter
        self.intensity_emitter = MockEmitter()
        self.plot_sequence_emitter = MockEmitter()

        #Set Mode
        # TODO: make sure LC or TriggerScope are loaded in the respective modes
        allowed_modes = ['MM-Retardance', 'MM-Voltage', 'DAC']
        assert lc_control_mode in allowed_modes, f'LC control mode must be one of {allowed_modes}'
        self.mode = lc_control_mode
        self.LC_DAC_conversion = 4  # convert between the input range of LCs (0-20V) and the output range of the DAC (0-5V)

        # Initialize calibration class
        allowed_interp_methods = ['schnoor_fit', 'linear']
        assert interp_method in allowed_interp_methods,\
            f'LC calibration data interpolation method must be one of {allowed_interp_methods}'
        dir_path = mmc.getDeviceAdapterSearchPaths().get(0) # MM device adapter directory
        self.calib = CalibrationData(os.path.join(dir_path, 'mmgr_dal_MeadowlarkLC.csv'), interp_method=interp_method,
                                     wavelength=wavelength)

        # Optimizer
        if optimization == 'min_scalar':
            self.optimizer = MinScalarOptimizer(self)
        elif optimization == 'brent':
            self.optimizer = BrentOptimizer(self)
        else:
            raise ModuleNotFoundError(f'No optimizer named {optimization}')

        # User / Calculated Parameters
        self.swing = None
        self.wavelength = None
        self.lc_bound = None
        self.I_Black = None
        self.ROI = None
        self.ratio = 1.793
        self.print_details = print_details
        self.calib_scheme = '4-State'

        # LC States
        self.lca_ext = None
        self.lcb_ext = None
        self.lca_0 = None
        self.lcb_0 = None
        self.lca_45 = None
        self.lcb_45 = None
        self.lca_60 = None
        self.lcb_60 = None
        self.lca_90 = None
        self.lcb_90 = None
        self.lca_120 = None
        self.lcb_120 = None
        self.lca_135 = None
        self.lcb_135 = None

        # Calibration Outputs
        self.I_Ext = None
        self.I_Ref = None
        self.I_Elliptical = None
        self.inten = []
        self.swing0 = None
        self.swing45 = None
        self.swing60 = None
        self.swing90 = None
        self.swing120 = None
        self.swing135 = None
        self.height = None
        self.width = None
        self.directory = None
        self.inst_mat = None

        # Shutter
        self.shutter_device = self.mmc.getShutterDevice()
        self._auto_shutter_state = None
        self._shutter_state = None

    def set_dacs(self, lca_dac, lcb_dac):
        self.PROPERTIES['LCA-DAC'] = (f'TS_{lca_dac}', 'Volts')
        self.PROPERTIES['LCB-DAC'] = (f'TS_{lcb_dac}', 'Volts')

    def set_wavelength(self, wavelength):
        self.calib.set_wavelength(wavelength)
        self.wavelength = self.calib.wavelength

    def set_lc(self, retardance, LC: str):
        """
        Set LC state to given retardance in waves

        Parameters
        ----------
        retardance : float
            Retardance in waves
        LC : str
            LCA or LCB

        Returns
        -------

        """

        if self.mode == 'MM-Retardance':
            set_lc_waves(self.mmc, self.PROPERTIES[f'{LC}'], retardance)
        elif self.mode == 'MM-Voltage':
            volts = self.calib.get_voltage(retardance)
            set_lc_voltage(self.mmc, self.PROPERTIES[f'{LC}-Voltage'], volts)
        elif self.mode == 'DAC':
            volts = self.calib.get_voltage(retardance)
            dac_volts = volts / self.LC_DAC_conversion
            set_lc_daq(self.mmc, self.PROPERTIES[f'{LC}-DAC'], dac_volts)

    def get_lc(self, LC: str):
        """
        Get LC retardance in waves

        Parameters
        ----------
        LC : str
            LCA or LCB

        Returns
        -------
            LC retardance in waves
        """

        if self.mode == 'MM-Retardance':
            retardance = get_lc(self.mmc, self.PROPERTIES[f'{LC}'])
        elif self.mode == 'MM-Voltage':
            volts = get_lc(self.mmc, self.PROPERTIES[f'{LC}-Voltage'])  # returned value is in volts
            retardance = self.calib.get_retardance(volts)
        elif self.mode == 'DAC':
            dac_volts = get_lc(self.mmc, self.PROPERTIES[f'{LC}-DAC'])
            volts = dac_volts * self.LC_DAC_conversion
            retardance = self.calib.get_retardance(volts)

        return retardance

    def define_lc_state(self, state, lca_retardance, lcb_retardance):
        """
        Define of the two LCs after calibration

        Parameters
        ----------
        state: str
            Polarization stage (e.g. State0)
        lca_retardance: float
            LCA retardance in waves
        lcb_retardance: float
            LCB retardance in waves

        Returns
        -------

        """

        if self.mode == 'MM-Retardance':
            self.set_lc(lca_retardance, 'LCA')
            self.set_lc(lcb_retardance, 'LCB')
            define_meadowlark_state(self.mmc, self.PROPERTIES[state])
        elif self.mode == 'DAC':
            lca_volts = self.calib.get_voltage(lca_retardance) / self.LC_DAC_conversion
            lcb_volts = self.calib.get_voltage(lcb_retardance) / self.LC_DAC_conversion
            define_config_state(self.mmc, self.group, state,
                                [self.PROPERTIES['LCA-DAC'], self.PROPERTIES['LCB-DAC']], [lca_volts, lcb_volts])
        elif self.mode == 'MM-Voltage':
            lca_volts = self.calib.get_voltage(lca_retardance)
            lcb_volts = self.calib.get_voltage(lcb_retardance)
            define_config_state(self.mmc, self.group, state,
                                [self.PROPERTIES['LCA-Voltage'], self.PROPERTIES['LCB-Voltage']], [lca_volts, lcb_volts])

    def opt_lc(self, x, device_property, reference, normalize=False):

        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]

        self.set_lc(x, device_property)

        mean = snap_and_average(self.snap_manager)

        if normalize:
            max_ = 65335
            min_ = self.I_Black

            val = (mean - min_) / (max_ - min_)
            ref = (reference - min_) / (max_ - min_)

            logging.debug(f'LC-Value: {x}')
            logging.debug(f'F-Value:{val - ref}\n')
            return val - ref

        else:
            logging.debug(str(mean))
            self.intensity_emitter.emit(mean)
            self.inten.append(mean - reference)

            return np.abs(mean - reference)

    def opt_lc_cons(self, x, device_property, reference, mode):

        self.set_lc(x, device_property)
        swing = (self.lca_ext - x) * self.ratio

        if mode == '60':
            self.set_lc(self.lcb_ext + swing, 'LCB')

        if mode == '120':
            self.set_lc(self.lcb_ext - swing, 'LCB')

        mean = snap_and_average(self.snap_manager)
        logging.debug(str(mean))

        # append to intensity array for plotting later
        self.intensity_emitter.emit(mean)
        self.inten.append(mean - reference)

        return np.abs(mean - reference)

    def opt_lc_grid(self, a_min, a_max, b_min, b_max, step):
        """
        Exhaustive Search method

        Finds the minimum intensity value for a given
        grid of LCA,LCB values

        :param a_min: float
            Minimum value of LCA
        :param a_max: float
            Maximum value of LCA
        :param b_min: float
            Minimum value of LCB
        :param b_max: float
            Maximum value of LCB
        :param step: float
            step size of the grid between max/min values


        :return best_lca: float
            LCA value corresponding to lowest mean Intensity
        :return best_lcb: float
            LCB value corresponding to lowest mean Intensity
        :return min_int: float
            Lowest value of mean Intensity
        """

        min_int = 65536
        better_lca = -1
        better_lcb = -1

        # coarse search
        for lca in np.arange(a_min, a_max, step):
            for lcb in np.arange(b_min, b_max, step):

                self.set_lc(lca, 'LCA')
                self.set_lc(lcb, 'LCB')

                # current_int = np.mean(snap_image(calib.mmc))
                current_int = snap_and_average(self.snap_manager)
                self.intensity_emitter.emit(current_int)

                if current_int < min_int:
                    better_lca = lca
                    better_lcb = lcb
                    min_int = current_int
                    logging.debug("update (%f, %f, %f)" % (min_int, better_lca, better_lcb))

        logging.debug("coarse search done")
        logging.debug("better lca = " + str(better_lca))
        logging.debug("better lcb = " + str(better_lcb))
        logging.debug("better int = " + str(min_int))

        best_lca = better_lca
        best_lcb = better_lcb

        return best_lca, best_lcb, min_int

    # ========== Optimization wrappers =============
    # ==============================================
    def opt_Iext(self):
        self.plot_sequence_emitter.emit('Coarse')
        logging.info('Calibrating State0 (Extinction)...')
        logging.debug('Calibrating State0 (Extinction)...')

        set_lc_state(self.mmc, self.group, 'State0')
        time.sleep(2)

        # Perform exhaustive search with step 0.1 over range:
        # 0.01 < LCA < 0.5
        # 0.25 < LCB < 0.75
        step = 0.1
        logging.debug(f"================================")
        logging.debug(f"Starting first grid search, step = {step}")
        logging.debug(f"================================")

        best_lca, best_lcb, i_ext_ = self.opt_lc_grid(0.01, 0.5, 0.25, 0.75, step)

        logging.debug("grid search done")
        logging.debug("lca = " + str(best_lca))
        logging.debug("lcb = " + str(best_lcb))
        logging.debug("intensity = " + str(i_ext_))

        self.set_lc(best_lca, 'LCA')
        self.set_lc(best_lcb, 'LCB')

        logging.debug(f"================================")
        logging.debug(f"Starting fine search")
        logging.debug(f"================================")

        # Perform brent optimization around results of 2nd grid search
        # threshold not very necessary here as intensity value will
        # vary between exposure/lamp intensities
        self.plot_sequence_emitter.emit('Fine')
        lca, lcb, I_ext = self.optimizer.optimize(state='ext', lca_bound=0.1, lcb_bound=0.1,
                                                  reference=self.I_Black, thresh=1, n_iter=5)

        # Set the Extinction state to values output from optimization
        self.define_lc_state('State0', lca, lcb)

        self.lca_ext = lca
        self.lcb_ext = lcb
        self.I_Ext = I_ext

        logging.debug("fine search done")
        logging.info(f'LCA State0 (Extinction) = {lca:.3f}')
        logging.debug(f'LCA State0 (Extinction) = {lca:.5f}')
        logging.info(f'LCB State0 (Extinction) = {lcb:.3f}')
        logging.debug(f'LCB State0 (Extinction) = {lcb:.5f}')
        logging.info(f'Intensity (Extinction) = {I_ext:.0f}')
        logging.debug(f'Intensity (Extinction) = {I_ext:.3f}')

        logging.debug("--------done--------")
        logging.info("--------done--------")


    def opt_I0(self):
        """
        no optimization performed for this.  Simply apply swing and read intensity
        This is the same as "Ielliptical".  Used for both schemes.
        :return: float
            mean of image
        """

        logging.info('Calibrating State1 (I0)...')
        logging.debug('Calibrating State1 (I0)...')

        self.lca_0 = self.lca_ext - self.swing
        self.lcb_0 = self.lcb_ext
        self.set_lc(self.lca_0, 'LCA')
        self.set_lc(self.lcb_0, 'LCB')

        self.define_lc_state('State1', self.lca_0, self.lcb_0)
        intensity = snap_and_average(self.snap_manager)
        self.I_Elliptical = intensity
        self.swing0 = np.sqrt((self.lcb_0 - self.lcb_ext) ** 2 + (self.lca_0 - self.lca_ext) ** 2)

        logging.info(f'LCA State1 (I0) = {self.lca_0:.3f}')
        logging.debug(f'LCA State1 (I0) = {self.lca_0:.5f}')
        logging.info(f'LCB State1 (I0) = {self.lcb_0:.3f}')
        logging.debug(f'LCB State1 (I0) = {self.lcb_0:.5f}')
        logging.info(f'Intensity (I0) = {intensity:.0f}')
        logging.debug(f'Intensity (I0) = {intensity:.3f}')
        logging.info("--------done--------")
        logging.debug("--------done--------")

    def opt_I45(self, lca_bound, lcb_bound):
        """
        optimized relative to Ielliptical (opt_I90)
        Parameters
        ----------
        lca_bound
        lcb_bound
        Returns
        -------
        lca, lcb value at optimized state
        intensity value at optimized state
        """
        self.inten = []
        logging.info('Calibrating State2 (I45)...')
        logging.debug('Calibrating State2 (I45)...')

        self.set_lc(self.lca_ext, 'LCA')
        self.set_lc(self.lcb_ext - self.swing, 'LCB')

        self.lca_45, self.lcb_45, intensity = self.optimizer.optimize('45', lca_bound, lcb_bound,
                                                                      reference=self.I_Elliptical, n_iter=5, thresh=.01)

        self.define_lc_state('State2', self.lca_45, self.lcb_45)

        self.swing45 = np.sqrt((self.lcb_45 - self.lcb_ext) ** 2 + (self.lca_45 - self.lca_ext) ** 2)

        logging.info(f'LCA State2 (I45) = {self.lca_45:.3f}')
        logging.debug(f'LCA State2 (I45) = {self.lca_45:.5f}')
        logging.info(f'LCB State2 (I45) = {self.lcb_45:.3f}')
        logging.debug(f'LCB State2 (I45) = {self.lcb_45:.5f}')
        logging.info(f'Intensity (I45) = {intensity:.0f}')
        logging.debug(f'Intensity (I45) = {intensity:.3f}')
        logging.info("--------done--------")
        logging.debug("--------done--------")

    def opt_I60(self, lca_bound, lcb_bound):
        """
        optimized relative to Ielliptical (opt_I0_4State)
        Parameters
        ----------
        lca_bound
        lcb_bound
        Returns
        -------
        lca, lcb value at optimized state
        intensity value at optimized state
        """
        self.inten = []

        logging.info('Calibrating State2 (I60)...')
        logging.debug('Calibrating State2 (I60)...')

        # Calculate Initial Swing for initial guess to optimize around
        # Based on ratio calculated from ellpiticity/orientation of LC simulation
        swing_ell = np.sqrt((self.lca_ext - self.lca_0) ** 2 + (self.lcb_ext - self.lcb_0) ** 2)
        lca_swing = np.sqrt(swing_ell ** 2 / (1 + self.ratio ** 2))
        lcb_swing = self.ratio * lca_swing

        # Optimization
        self.set_lc(self.lca_ext + lca_swing, 'LCA')
        self.set_lc(self.lcb_ext + lcb_swing, 'LCB')

        self.lca_60, self.lcb_60, intensity = self.optimizer.optimize('60', lca_bound, lcb_bound,
                                                                      reference=self.I_Elliptical,
                                                                      n_iter=5, thresh=.01)

        self.define_lc_state('State2', self.lca_60, self.lcb_60)

        self.swing60 = np.sqrt((self.lcb_60 - self.lcb_ext) ** 2 + (self.lca_60 - self.lca_ext) ** 2)

        # Print comparison of target swing, target ratio
        # Ratio determines the orientation of the elliptical state
        # should be close to target.  Swing will vary to optimize ellipticity
        logging.debug(f'ratio: swing_LCB / swing_LCA = {(self.lcb_ext - self.lcb_60) / (self.lca_ext - self.lca_60):.4f} \
              | target ratio: {-self.ratio}')
        logging.debug(f'total swing = {self.swing60:.4f} | target = {swing_ell}')

        logging.info("LCA State2 (I60) = " + str(self.lca_60))
        logging.debug("LCA State2 (I60) = " + str(self.lca_60))
        logging.info("LCB State2 (I60) = " + str(self.lcb_60))
        logging.debug("LCB State2 (I60) = " + str(self.lcb_60))
        logging.info(f'Intensity (I60) = {intensity}')
        logging.debug(f'Intensity (I60) = {intensity}')
        logging.info("--------done--------")
        logging.debug("--------done--------")

    def opt_I90(self, lca_bound, lcb_bound):
        """
        optimized relative to Ielliptical (opt_I90)
        Parameters
        ----------
        lca_bound
        lcb_bound
        Returns
        -------
        lca, lcb value at optimized state
        intensity value at optimized state
        """
        logging.info('Calibrating State3 (I90)...')
        logging.debug('Calibrating State3 (I90)...')

        self.inten = []

        self.set_lc(self.lca_ext + self.swing, 'LCA')
        self.set_lc(self.lcb_ext, 'LCB')

        self.lca_90, self.lcb_90, intensity = self.optimizer.optimize('90', lca_bound, lcb_bound,
                                                                      reference=self.I_Elliptical,
                                                                      n_iter=5, thresh=.01)

        self.define_lc_state('State3', self.lca_90, self.lcb_90)

        self.swing90 = np.sqrt((self.lcb_90 - self.lcb_ext) ** 2 + (self.lca_90 - self.lca_ext) ** 2)

        logging.info(f'LCA State3 (I90) = {self.lca_90:.3f}')
        logging.debug(f'LCA State3 (I90) = {self.lca_90:.5f}')
        logging.info(f'LCB State3 (I90) = {self.lcb_90:.3f}')
        logging.debug(f'LCB State3 (I90) = {self.lcb_90:.5f}')
        logging.info(f'Intensity (I90) = {intensity:.0f}')
        logging.debug(f'Intensity (I90) = {intensity:.3f}')
        logging.info("--------done--------")
        logging.debug("--------done--------")

    def opt_I120(self, lca_bound, lcb_bound):
        """
        optimized relative to Ielliptical (opt_I0_4State)
        Parameters
        ----------
        lca_bound
        lcb_bound
        Returns
        -------
        lca, lcb value at optimized state
        intensity value at optimized state
        """
        logging.info('Calibrating State3 (I120)...')
        logging.debug('Calibrating State3 (I120)...')

        # Calculate Initial Swing for initial guess to optimize around
        # Based on ratio calculated from ellpiticity/orientation of LC simulation
        swing_ell = np.sqrt((self.lca_ext - self.lca_0) ** 2 + (self.lcb_ext - self.lcb_0) ** 2)
        lca_swing = np.sqrt(swing_ell ** 2 / (1 + self.ratio ** 2))
        lcb_swing = self.ratio * lca_swing

        # Brent Optimization
        self.set_lc(self.lca_ext + lca_swing, 'LCA')
        self.set_lc(self.lcb_ext - lcb_swing, 'LCB')

        self.lca_120, self.lcb_120, intensity = self.optimizer.optimize('120', lca_bound, lcb_bound,
                                                                      reference=self.I_Elliptical,
                                                                      n_iter=5, thresh=.01)

        self.define_lc_state('State3', self.lca_120, self.lcb_120)

        self.swing120 = np.sqrt((self.lcb_120 - self.lcb_ext) ** 2 + (self.lca_120 - self.lca_ext) ** 2)

        # Print comparison of target swing, target ratio
        # Ratio determines the orientation of the elliptical state
        # should be close to target.  Swing will vary to optimize ellipticity
        logging.debug(f'ratio: swing_LCB / swing_LCA = {(self.lcb_ext - self.lcb_120) / (self.lca_ext - self.lca_120):.4f}\
             | target ratio: {self.ratio}')
        logging.debug(f'total swing = {self.swing120:.4f} | target = {swing_ell}')
        logging.info("LCA State3 (I120) = " + str(self.lca_120))
        logging.debug("LCA State3 (I120) = " + str(self.lca_120))
        logging.info("LCB State3 (I120) = " + str(self.lcb_120))
        logging.debug("LCB State3 (I120) = " + str(self.lcb_120))
        logging.info(f'Intensity (I120) = {intensity}')
        logging.debug(f'Intensity (I120) = {intensity}')
        logging.info("--------done--------")
        logging.debug("--------done--------")

    def opt_I135(self, lca_bound, lcb_bound):
        """
        optimized relative to Ielliptical (opt_I0)
        Parameters
        ----------
        lca_bound
        lcb_bound
        Returns
        -------
        lca, lcb value at optimized state
        intensity value at optimized state
        """

        logging.info('Calibrating State3 (I135)...')
        logging.debug('Calibrating State3 (I135)...')
        self.inten = []

        self.set_lc(self.lca_ext, 'LCA')
        self.set_lc(self.lcb_ext + self.swing, 'LCB')

        self.lca_135, self.lcb_135, intensity = self.optimizer.optimize('135', lca_bound, lcb_bound,
                                                                      reference=self.I_Elliptical,
                                                                      n_iter=5, thresh=.01)

        self.define_lc_state('State4', self.lca_135, self.lcb_135)

        self.swing135 = np.sqrt((self.lcb_135 - self.lcb_ext) ** 2 + (self.lca_135 - self.lca_ext) ** 2)

        logging.info(f'LCA State4 (I135) = {self.lca_135:.3f}')
        logging.debug(f'LCA State4 (I135) = {self.lca_135:.5f}')
        logging.info(f'LCB State4 (I135) = {self.lcb_135:.3f}')
        logging.debug(f'LCB State4 (I135) = {self.lcb_135:.5f}')
        logging.info(f'Intensity (I135) = {intensity:.0f}')
        logging.debug(f'Intensity (I135) = {intensity:.3f}')
        logging.info("--------done--------")
        logging.debug("--------done--------")

    def open_shutter(self):
        if self.shutter_device == '': # no shutter
            input('Please manually open the shutter and press <Enter>')
        else:
            self.mmc.setShutterOpen(True)

    def reset_shutter(self):
        """
        Return autoshutter to its original state before closing

        Returns
        -------

        """
        if self.shutter_device == '': # no shutter
            input('Please reset the shutter to its original state and press <Enter>')
            logging.info("This is the end of the command-line instructions. You can return to the napari window.")
        else:
            self.mmc.setAutoShutter(self._auto_shutter_state)
            self.mmc.setShutterOpen(self._shutter_state)

    def close_shutter_and_calc_blacklevel(self):
        self._auto_shutter_state = self.mmc.getAutoShutter()
        self._shutter_state = self.mmc.getShutterOpen()

        if self.shutter_device == '':  # no shutter
            show_warning('No shutter found. Please follow the command-line instructions...')
            shutter_warning_msg = """
            recOrder could not find an automatic shutter configured through Micro-Manager.
            >>> If you would like manually enter the black level, enter an integer or float and press <Enter>
            >>> If you would like to estimate the black level, please close the shutter and press <Enter> 
            """

            in_string = input(shutter_warning_msg)
            if in_string.isdigit(): # True if positive integer
                self.I_Black = float(in_string)
                return
        else:
            self.mmc.setAutoShutter(False)
            self.mmc.setShutterOpen(False)

        n_avg = 20
        avgs = []
        for i in range(n_avg):
            mean = snap_and_average(self.snap_manager)
            self.intensity_emitter.emit(mean)
            avgs.append(mean)

        blacklevel = np.mean(avgs)
        self.I_Black = blacklevel

    def get_full_roi(self):
        # Get Image Parameters
        self.mmc.snapImage()
        self.mmc.getImage()
        self.height, self.width = self.mmc.getImageHeight(), self.mmc.getImageWidth()
        self.ROI = (0, 0, self.width, self.height)

    def check_and_get_roi(self):

        windows = self.mm.displays().getAllImageWindows()
        size = windows.size()

        boxes = []
        for i in range(size):
            win = windows.get(i).toFront()
            time.sleep(0.05)
            roi = self.mm.displays().getActiveDataViewer().getImagePlus().getRoi()
            if roi != None:
                boxes.append(roi)

        if len(boxes) == 0:
            raise ValueError('No ROI Bounding Box Found, Please Draw Bounding Box on the Preview (live) Window')

        if len(boxes) > 1:
            raise ValueError('More than one Bounding Box Found, Please Remove any box not on the preview (live) window')

        if len(boxes) == 1:
            rect = boxes[0].getBounds()
            return rect

    def display_and_check_ROI(self, rect):

        img = snap_image(self.mmc)

        print('Will Calibrate Using this ROI:')
        fig, ax = plt.subplots()

        ax.imshow(np.reshape(img, (self.height, self.width)), 'gray')
        box = patches.Rectangle((rect.x, rect.y), rect.width, rect.height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(box)
        plt.show()

        cont = input('Would You Like to Calibrate Using this ROI? (Yes/No): \t')

        if cont in ['Yes', 'Y', 'yes', 'ye', 'y', '']:
            return True

        if cont in ['No', 'N', 'no', 'n']:
            return False

        else:
            raise ValueError('Did not understand your answer, please check spelling')

    def calculate_extinction(self, swing, black_level, intensity_extinction, intensity_elliptical):
        return np.round((1 / np.sin(np.pi * swing) ** 2) * \
               (intensity_elliptical - black_level) / (intensity_extinction - black_level), 2)

    def calc_inst_matrix(self):

        if self.calib_scheme == '4-State':
            chi = self.swing
            inst_mat = np.array([[1, 0, 0, -1],
                                 [1, np.sin(2 * np.pi * chi), 0, -np.cos(2 * np.pi * chi)],
                                 [1, -0.5 * np.sin(2 * np.pi * chi),
                                  np.sqrt(3) * np.cos(np.pi * chi) * np.sin(np.pi * chi), -np.cos(2 * np.pi * chi)],
                                 [1, -0.5 * np.sin(2 * np.pi * chi), -np.sqrt(3) / 2 * np.sin(2 * np.pi * chi),
                                  -np.cos(2 * np.pi * chi)]])

            return inst_mat

        if self.calib_scheme == '5-State':
            chi = self.swing * 2 * np.pi

            inst_mat = np.array([[1, 0, 0, -1],
                                 [1, np.sin(chi), 0, -np.cos(chi)],
                                 [1, 0, np.sin(chi), -np.cos(chi)],
                                 [1, -np.sin(chi), 0, -np.cos(chi)],
                                 [1, 0, -np.sin(chi), -np.cos(chi)]])

            return inst_mat

    def write_metadata(self, notes=None, microscope_params=None):

        inst_mat = self.calc_inst_matrix()
        inst_mat = np.around(inst_mat, decimals=5).tolist()

        metadata = {'Summary': {
                        'Timestamp': str(datetime.now()),
                        'recOrder-napari version': version('recOrder-napari'),
                        'waveorder version': version('waveorder')},
                    'Calibration': {
                        'Calibration scheme': self.calib_scheme,
                        'Swing (waves)': self.swing,
                        'Wavelength (nm)': self.wavelength,
                        'Retardance to voltage interpolation method': self.calib.interp_method,
                        'LC control mode': self.mode,
                        'Black level': np.round(self.I_Black, 2),
                        'Extinction ratio': self.extinction_ratio,
                        'ROI (x, y, width, height)': self.ROI},
                    'Notes': notes,
                    'Microscope parameters': microscope_params
                    }

        if self.calib_scheme == '4-State':
            metadata['Calibration'].update({
                'Channel names': [f"State{i}" for i in range(4)],
                'LC retardance': {f'LC{i}_{j}': np.around(getattr(self, f'lc{i.lower()}_{j}'), decimals=6)
                                  for j in ['ext', '0', '60', '120']
                                  for i in ['A', 'B']},
                'LC voltage': {f'LC{i}_{j}': np.around(self.calib.get_voltage(getattr(self, f'lc{i.lower()}_{j}')), decimals=4)
                               for j in ['ext', '0', '60', '120']
                               for i in ['A', 'B']},
                'Swing_0': np.around(self.swing0, decimals=3),
                'Swing_60': np.around(self.swing60, decimals=3),
                'Swing_120': np.around(self.swing120, decimals=3),
                'Instrument matrix': inst_mat
            })

        elif self.calib_scheme == '5-State':
            metadata['Calibration'].update({
                'Channel names': [f"State{i}" for i in range(5)],
                'LC retardance': {f'LC{i}_{j}': np.around(getattr(self, f'lc{i.lower()}_{j}'), decimals=6)
                                  for j in ['ext', '0', '45', '90', '135']
                                  for i in ['A', 'B']},
                'LC voltage': {f'LC{i}_{j}': np.around(self.calib.get_voltage(getattr(self, f'lc{i.lower()}_{j}')), decimals=4)
                               for j in ['ext', '0', '45', '90', '135']
                               for i in ['A', 'B']},
                'Swing_0': np.around(self.swing0, decimals=3),
                'Swing_45': np.around(self.swing45, decimals=3),
                'Swing_90': np.around(self.swing90, decimals=3),
                'Swing_135': np.around(self.swing135, decimals=3),
                'Instrument matrix': inst_mat
            })

        if not self.meta_file.endswith('.txt'):
            self.meta_file += '.txt'

        with open(self.meta_file, 'w') as metafile:
            json.dump(metadata, metafile, indent=1)

    def _add_colorbar(self, mappable):
        last_axes = plt.gca()
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mappable, cax=cax)
        plt.sca(last_axes)
        return cbar

    def _capture_state(self, state, n_avg):
        set_lc_state(self.mmc, self.group, state)

        imgs = []
        for i in range(n_avg):
            imgs.append(snap_and_get_image(self.snap_manager))

        return np.mean(imgs, axis=0)

    def _plot_bg_images(self, imgs):

        img_names = ['Extinction', '0', '60', '120'] if len(imgs) == 4 else ['Extinction', '0', '45', '90', 135]
        fig, ax = plt.subplots(2, 2, figsize=(20, 20)) if len(imgs) == 4 else plt.subplots(3, 2, figsize=(20, 20))

        img_idx = 0
        for ax1 in range(len(ax[:, 0])):
            for ax2 in range(len(ax[0, :])):
                if img_idx < len(imgs):
                    im = ax[ax1, ax2].imshow(imgs[img_idx], 'gray')
                    ax[ax1, ax2].set_title(img_names[img_idx])
                    self._add_colorbar(im)
                else:
                    try:
                        fig.delaxes(ax[2, 1])
                    except:
                        break
        plt.show()


    def capture_bg(self, n_avg, directory):
        """"
        This function will capture an image at every state
        and save to specified directory
        This may throw errors depending on the micromanager config file--
        modify 'State_' to match to the corresponding channel preset in config
        :param: n_states (int)
            Number of states used for calibration
        :param: directory (string)
            Directory to save images
        """

        if not os.path.exists(directory):
            os.makedirs(directory)

        logging.info('Capturing Background')
        logging.debug('Capturing Bacckground State0')

        state0 = self._capture_state('State0', n_avg)
        logging.debug('Saving Bacckground State0')
        tiff.imsave(os.path.join(directory, 'State0.tif'), state0)

        logging.debug('Capturing Bacckground State1')
        state1 = self._capture_state('State1', n_avg)
        logging.debug('Saving Bacckground State1')
        tiff.imsave(os.path.join(directory, 'State1.tif'), state1)

        logging.debug('Capturing Bacckground State2')
        state2 = self._capture_state('State2', n_avg)
        logging.debug('Saving Bacckground State2')
        tiff.imsave(os.path.join(directory, 'State2.tif'), state2)

        logging.debug('Capturing Bacckground State3')
        state3 = self._capture_state('State3', n_avg)
        logging.debug('Saving Bacckground State3')
        tiff.imsave(os.path.join(directory, 'State3.tif'), state3)

        imgs = [state0, state1, state2, state3]

        if self.calib_scheme == '5-State':
            logging.debug('Capturing Bacckground State4')
            state4 = self._capture_state('State4', n_avg)
            logging.debug('Saving Bacckground State4')
            tiff.imsave(os.path.join(directory, 'State4.tif'), state4)
            imgs.append(state4)

        # self._plot_bg_images(np.asarray(imgs))

        return np.asarray(imgs)


class CalibrationData:
    """
    Interpolates LC calibration data between retardance (in waves), voltage (in mV), and wavelength (in nm)
    """

    def __init__(self, path, wavelength=532, interp_method='linear'):
        """

        Parameters
        ----------
        path : str
            path to .csv calibration data file
        wavelength : int
            usage wavelength, in nanometers
        interp_method : str
            interpolation method, either "linear" or "schnoor_fit" (https://doi.org/10.1364/AO.408383)
        """

        header, raw_data = self.read_data(path)
        self.calib_wavelengths = np.array([i[:3] for i in header[1::3]]).astype('double')

        self.wavelength = None
        self.V_min = 0
        self.V_max = 20

        if interp_method in ['linear', 'schnoor_fit']:
            self.interp_method = interp_method
        else:
            raise ValueError('Unknown interpolation method.')

        self.set_wavelength(wavelength)
        if interp_method == 'linear':
            self.interpolate_data(raw_data, self.calib_wavelengths)  # calib_wavelengths is not used, values hardcoded
        elif interp_method == 'schnoor_fit':
            self.fit_params = self.fit_data(raw_data, self.calib_wavelengths)

        self.ret_min = self.get_retardance(self.V_max)
        self.ret_max = self.get_retardance(self.V_min)

    @staticmethod
    def read_data(path):
        """
        Read raw calibration data

        Example calibration data format:

            Voltage(mv),490-A,490-B,Voltage(mv),546-A,546-B,Voltage(mv),630-A,630-B
            -,-,-,-,-,-,-,-,-
            0,490,490,0,546,546,0,630,630
            0,970.6205,924.4288,0,932.2446,891.2008,0,899.6626,857.2885
            200,970.7488,924.4422,200,932.2028,891.1546,200,899.5908,857.3078
            ...
            20000,40.5954,40.4874,20000,38.6905,39.5402,20000,35.5043,38.1445
            -,-,-,-,-,-,-,-,-

        The first row of the CSV file is a header row, structured as [Voltage (mV), XXX-A, XXX-B,
        Voltage (nm), XXX-A, XXX-B, ...] where XXX is the calibration wavelength in nanometers. For example 532-A would
        contain measurements of the retardance of LCA as a function of applied voltage at 532 nm. The second row
        contains dashes in every column. The third row contains "0" in the Voltage column and the calibration wavelength
        in the retardance columns, e.g [0, 532, 532]. The following rows contain the LC calibration data. Retardance is
        recorded in nanometers and voltage is recorded in millivolts. The last row contains dashes in every column.

        Parameters
        ----------
        path : str
            path to .csv calibration data file

        Returns
        -------
        header : list
            Calibration data file header line. Contains information on calibration wavelength
        raw_data : ndarray
            Calibration data. Voltage is in millivolts and retardance is in nanometers

        """
        with open(path, 'r') as f:
            header = f.readline().strip().split(',')

        raw_data = np.loadtxt(path, delimiter=',', comments='-', skiprows=3)
        return header, raw_data

    @staticmethod
    def schnoor_fit(V, a, b1, b2, c, d, e, wavelength):
        """

        Parameters
        ----------
        V : float
            Voltage in volts
        a, b1, b2, c, d, e : float
            Fit parameters
        wavelength : float
            Wavelength in nanometers

        Returns
        -------
        retardance : float
            Retardance in nanometers

        """
        retardance = a + (b1 + b2 / wavelength ** 2) / (1 + (V / c) ** d) ** e

        return retardance

    @staticmethod
    def schnoor_fit_inv(retardance, a, b1, b2, c, d, e, wavelength):
        """

        Parameters
        ----------
        retardance : float
            Retardance in nanometers
        a, b1, b2, c, d, e : float
            Fit parameters
        wavelength : float
            Wavelength in nanometers

        Returns
        -------
        voltage : float
            Voltage in volts

        """

        voltage = c * (((b1 + b2 / wavelength ** 2) / (retardance - a)) ** (1 / e) - 1) ** (1 / d)

        return voltage

    @staticmethod
    def _fun(x, wavelengths, xdata, ydata):
        fval = CalibrationData.schnoor_fit(xdata, *x, wavelengths)
        res = ydata - fval
        return res.flatten()

    def set_wavelength(self, wavelength):
        if len(self.calib_wavelengths) == 1 and wavelength != self.calib_wavelengths:
            raise ValueError("Calibration is not provided at this wavelength. "
                             "Wavelength dependence of LC retardance vs voltage cannot be extrapolated.")

        if wavelength < self.calib_wavelengths.min() or \
                wavelength > self.calib_wavelengths.max():
            warnings.warn("Specified wavelength is outside of the calibration range. "
                          "LC retardance vs voltage data will be extrapolated at this wavelength.")

        self.wavelength = wavelength
        if self.interp_method == 'linear':
            # Interpolation of calib beyond this range produce strange results.
            if self.wavelength < 450:
                self.wavelength = 450
                warnings.warn("Wavelength is limited to 450-720 nm for this interpolation method.")
            if self.wavelength > 720:
                self.wavelength = 720
                warnings.warn("Wavelength is limited to 450-720 nm for this interpolation method.")

    def fit_data(self, raw_data, calib_wavelengths):
        """
        Perform Schnoor fit on interpolation data

        Parameters
        ----------
        raw_data : np.array
            LC calibration data in (Voltage, LCA retardance, LCB retardance) format. Only the LCA retardance vs voltage
            curve is used.
        calib_wavelengths : 1D np.array
            Calibration wavelength for each (Voltage, LCA retardance, LCB retardance) set in the calibration data

        Returns
        -------

        """
        xdata = raw_data[:, 0::3] / 1000    # convert to volts
        ydata = raw_data[:, 1::3]           # in nanometers

        x0 = [10, 1000, 1e7, 1, 10, 0.1]
        p = least_squares(self._fun, x0, method='trf', args=(calib_wavelengths, xdata, ydata),
                          bounds=((-np.inf, 0, 0, 0, 0, 0), (np.inf,)*6),
                          x_scale=[10, 1000, 1e7, 1, 10, 0.1])

        if not p.success:
            raise RuntimeError("Schnoor fit to calibration data did not work.")

        y = ydata.flatten()
        y_hat = y - p.fun
        slope, intercept, r_value, *_ = linregress(y, y_hat)
        r_squared = r_value**2
        if r_squared < 0.999:
            warnings.warn(f'Schnoor fit has R2 value of {r_squared:.5f}, fit may not have worked well.')

        return p.x

    def interpolate_data(self, raw_data, calib_wavelengths):
        """
        Perform linear interpolation of LC calibration data

        Parameters
        ----------
        raw_data : np.array
            LC calibration data in (Voltage, LCA retardance, LCB retardance) format. Only the LCA retardance vs voltage
            curve is used.
        calib_wavelengths : 1D np.array
            Calibration wavelength for each (Voltage, LCA retardance, LCB retardance) set in the calibration data
            These values are not used in this method. Instead, the [490, 546, 630] wavelengths are hardcoded.

        Returns
        -------

        """
        # 0V to 20V step size 1 mV
        x_range = np.arange(0, np.max(raw_data[:, ::3]), 1)

        # interpolate calib - only LCA data is used
        spline490 = interp1d(raw_data[:, 0], raw_data[:, 1])
        spline546 = interp1d(raw_data[:, 3], raw_data[:, 4])
        spline630 = interp1d(raw_data[:, 6], raw_data[:, 7])

        if self.wavelength < 490:
            new_a1_y = np.interp(x_range, x_range, spline490(x_range))
            new_a2_y = np.interp(x_range, x_range, spline546(x_range))

            wavelength_new = 490 + (490 - self.wavelength)
            fact1 = np.abs(490 - wavelength_new) / (546 - 490)
            fact2 = np.abs(546 - wavelength_new) / (546 - 490)

            temp_curve = np.asarray([[i, 2 * new_a1_y[i] - (fact1 * new_a1_y[i] + fact2 * new_a2_y[i])]
                          for i in range(len(new_a1_y))])
            self.spline = interp1d(temp_curve[:, 0], temp_curve[:, 1])
            self.curve = self.spline(x_range)

        elif self.wavelength > 630:

            new_a1_y = np.interp(x_range, x_range, spline546(x_range))
            new_a2_y = np.interp(x_range, x_range, spline630(x_range))

            wavelength_new = 630 + (630 - self.wavelength)
            fact1 = np.abs(630 - wavelength_new) / (630 - 546)
            fact2 = np.abs(546 - wavelength_new) / (630 - 546)

            temp_curve = np.asarray([[i, 2 * new_a1_y[i] - (fact1 * new_a1_y[i] + fact2 * new_a2_y[i])]
                                     for i in range(len(new_a1_y))])
            self.spline = interp1d(temp_curve[:, 0], temp_curve[:, 1])
            self.curve = self.spline(x_range)


        elif 490 < self.wavelength < 546:

            new_a1_y = np.interp(x_range, x_range, spline490(x_range))
            new_a2_y = np.interp(x_range, x_range, spline546(x_range))

            fact1 = np.abs(490 - self.wavelength) / (546 - 490)
            fact2 = np.abs(546 - self.wavelength) / (546 - 490)

            temp_curve = np.asarray([[i, fact1 * new_a1_y[i] + fact2 * new_a2_y[i]] for i in range(len(new_a1_y))])
            self.spline = interp1d(temp_curve[:, 0], temp_curve[:, 1])
            self.curve = self.spline(x_range)

        elif 546 < self.wavelength < 630:

            new_a1_y = np.interp(x_range, x_range, spline546(x_range))
            new_a2_y = np.interp(x_range, x_range, spline630(x_range))

            fact1 = np.abs(546 - self.wavelength) / (630 - 546)
            fact2 = np.abs(630 - self.wavelength) / (630 - 546)

            temp_curve = np.asarray([[i, fact1 * new_a1_y[i] + fact2 * new_a2_y[i]] for i in range(len(new_a1_y))])
            self.spline = interp1d(temp_curve[:, 0], temp_curve[:, 1])
            self.curve = self.spline(x_range)

        elif self.wavelength == 490:
            self.curve = spline490(x_range)
            self.spline = spline490

        elif self.wavelength == 546:
            self.curve = spline546(x_range)
            self.spline = spline546

        elif self.wavelength == 630:
            self.curve = spline630(x_range)
            self.spline = spline630

        else:
            raise ValueError(f'Wavelength {self.wavelength} not understood')

    def get_voltage(self, retardance):
        """

        Parameters
        ----------
        retardance : float
            retardance in waves

        Returns
        -------
        voltage
            voltage in volts

        """

        retardance = np.asarray(retardance, dtype='double')
        voltage = None
        ret_nanometers = retardance*self.wavelength

        if retardance < self.ret_min:
            voltage = self.V_max
        elif retardance > self.ret_max:
            voltage = self.V_min
        else:
            if self.interp_method == 'linear':
                voltage = np.abs(self.curve - ret_nanometers).argmin() / 1000
            elif self.interp_method == 'schnoor_fit':
                voltage = self.schnoor_fit_inv(ret_nanometers, *self.fit_params, self.wavelength)

        return voltage

    def get_retardance(self, volts):
        """

        Parameters
        ----------
        volts : float
            voltage in volts

        Returns
        -------
        retardance : float
            retardance in waves

        """

        volts = np.asarray(volts, dtype='double')
        ret_nanometers = None

        if volts < self.V_min:
            volts = self.V_min
        elif volts >= self.V_max:
            if self.interp_method == 'linear':
                volts = self.V_max - 1e-3   # interpolation breaks down at upper boundary
            else:
                volts = self.V_max

        if self.interp_method == 'linear':
            ret_nanometers = self.spline(volts * 1000)
        elif self.interp_method == 'schnoor_fit':
            ret_nanometers = self.schnoor_fit(volts, *self.fit_params, self.wavelength)
        retardance = ret_nanometers / self.wavelength

        return retardance
