import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tifffile as tiff
import time
from recOrder.io.core_functions import define_lc_state, snap_image, set_lc_waves, set_lc_volts, set_lc_state, \
    snap_and_average, snap_and_get_image, get_lc_waves, get_lc_volts, define_lc_state_volts
from recOrder.calib.Optimization import BrentOptimizer, MinScalarOptimizer
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.interpolate import interp1d
import json
import os
import logging
from recOrder.io.utils import MockEmitter
from datetime import datetime


class QLIPP_Calibration():

    def __init__(self, mmc, mm, group='Channel', optimization='min_scalar', mode='retardance', print_details=True):

        # Micromanager API
        self.mm = mm
        self.mmc = mmc
        self.snap_manager = mm.getSnapLiveManager()

        # Meadowlark LC Device Adapter Property Names
        self.PROPERTIES = {'LCA': 'Retardance LC-A [in waves]',
                          'LCB': 'Retardance LC-B [in waves]',
                          'State0': 'Pal. elem. 00; enter 0 to define; 1 to activate',
                          'State1': 'Pal. elem. 01; enter 0 to define; 1 to activate',
                          'State2': 'Pal. elem. 02; enter 0 to define; 1 to activate',
                          'State3': 'Pal. elem. 03; enter 0 to define; 1 to activate',
                          'State4': 'Pal. elem. 04; enter 0 to define; 1 to activate',
                          'LCA-volt': 'TS_DAC01',
                          'LCB-volt': 'TS_DAC02'
                          }
        self.group = group

        # GUI Emitter
        self.intensity_emitter = MockEmitter()
        self.plot_sequence_emitter = MockEmitter()

        #Set Mode
        self.mode = mode
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.curves = CalibrationCurves(os.path.join(dir_path, './Meadowlark_Curves.npy')) if self.mode != 'retardance' else None

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

        # Voltage DACS
        self.lca_dac = None
        self.lcb_dac = None

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

    def set_dacs(self, lca_dac, lcb_dac):
        self.PROPERTIES['LCA-volt'] = f'TS_{lca_dac}'
        self.PROPERTIES['LCB-volt'] = f'TS_{lcb_dac}'

    def set_wavelength(self, wavelength):
        self.wavelength = wavelength

        if self.mode == 'voltage':
            self.curves.set_wavelength(wavelength)

    def set_lc(self, val, device_property):

        if self.mode == 'retardance':
            set_lc_waves(self.mmc, val, self.PROPERTIES[device_property])
        else:
            volt = self.curves.get_voltage(val)
            set_lc_volts(self.mmc, volt/4000, self.PROPERTIES[f'{device_property}-volt'])

    def get_lc(self, device_property):

        if self.mode == 'retardance':
            return get_lc_waves(self.mmc, self.PROPERTIES[device_property])
        else:
            volts = get_lc_volts(self.mmc, self.PROPERTIES[f'{device_property}-volt'])
            return self.curves.get_retardance(volts*4000)

    def define_lc_state(self, state, lca, lcb):

        if self.mode == 'retardance':
            define_lc_state(self.mmc, state, lca, lcb, self.PROPERTIES)
        else:
            define_lc_state_volts(self.mmc, self.group, state, lca, lcb, self.lca_dac, self.lcb_dac)

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
        logging.info("LCA State0 (Extinction) = " + str(lca))
        logging.debug("LCA State0 (Extinction) = " + str(lca))
        logging.info("LCB State0 (Extinction) = " + str(lcb))
        logging.debug("LCB State0 (Extinction) = " + str(lcb))
        logging.info("Intensity (Extinction) = " + str(I_ext))
        logging.debug("Intensity (Extinction) = " + str(I_ext))

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

        self.define_lc_state('State1', self.lca_ext - self.swing, self.lcb_ext)

        ref = snap_and_average(self.snap_manager)

        self.lca_0 = self.lca_ext - self.swing
        self.lcb_0 = self.lcb_ext
        self.I_Elliptical = ref
        self.swing0 = np.sqrt((self.lcb_0 - self.lcb_ext) ** 2 + (self.lca_0 - self.lca_ext) ** 2)

        logging.info("LCA State1 (I0) = " + str(self.lca_0))
        logging.debug("LCA State1 (I0) = " + str(self.lca_0))
        logging.info("LCB State1 (I0) = " + str(self.lcb_0))
        logging.debug("LCB State1 (I0) = " + str(self.lcb_0))
        logging.info(f'Intensity (I0) = {ref}')
        logging.debug(f'Intensity (I0) = {ref}')
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

        logging.info("LCA State2 (I45) = " + str(self.lca_45))
        logging.debug("LCA State2 (I45) = " + str(self.lca_45))
        logging.info("LCB State2 (I45) = " + str(self.lcb_45))
        logging.debug("LCB State2 (I45) = " + str(self.lcb_45))
        logging.info(f'Intensity (I45) = {intensity}')
        logging.debug(f'Intensity (I45) = {intensity}')
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

        logging.info("LCA State3 (I90) = " + str(self.lca_90))
        logging.debug("LCA State3 (I90) = " + str(self.lca_90))
        logging.info("LCB State3 (I90) = " + str(self.lcb_90))
        logging.debug("LCB State3 (I90) = " + str(self.lcb_90))
        logging.info(f'Intensity (I90) = {intensity}')
        logging.debug(f'Intensity (I90) = {intensity}')
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
        print('Calibrating State4 (I135)...')
        self.inten = []

        self.set_lc(self.lca_ext, 'LCA')
        self.set_lc(self.lcb_ext + self.swing, 'LCB')

        self.lca_135, self.lcb_135, intensity = self.optimizer.optimize('135', lca_bound, lcb_bound,
                                                                      reference=self.I_Elliptical,
                                                                      n_iter=5, thresh=.01)

        self.define_lc_state('State4', self.lca_135, self.lcb_135)

        self.swing135 = np.sqrt((self.lcb_135 - self.lcb_ext) ** 2 + (self.lca_135 - self.lca_ext) ** 2)

        logging.info("LCA State4 (I135) = " + str(self.lca_135))
        logging.debug("LCA State4 (I135) = " + str(self.lca_135))
        logging.info("LCB State4 (I135) = " + str(self.lcb_135))
        logging.debug("LCB State4 (I135) = " + str(self.lcb_135))
        logging.info(f'Intensity (I135) = {intensity}')
        logging.debug(f'Intensity (I135) = {intensity}')
        logging.info("--------done--------")
        logging.debug("--------done--------")

    def calc_blacklevel(self):

        auto_shutter = self.mmc.getAutoShutter()
        shutter = self.mmc.getShutterOpen()

        self.mmc.setAutoShutter(False)
        self.mmc.setShutterOpen(False)

        n_avg = 20
        avgs = []
        for i in range(n_avg):
            mean = snap_and_average(self.snap_manager)
            self.intensity_emitter.emit(mean)
            avgs.append(mean)

        blacklevel = np.mean(avgs)

        self.mmc.setAutoShutter(auto_shutter)

        if not auto_shutter:
            self.mmc.setShutterOpen(shutter)

        self.I_Black = blacklevel

        return blacklevel

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

    def run_5state_calibration(self, param):
        """
        Param is a list or tuple of:
            (swing, wavelength, lc_bounds, black level)
        """
        self.swing = param[0]
        self.wavelength = param[1]
        self.meta_file = param[2]
        use_full_FOV = param[3]

        # Get Image Parameters
        self.mmc.snapImage()
        self.mmc.getImage()
        self.height, self.width = self.mmc.getImageHeight(), self.mmc.getImageWidth()
        self.ROI = (0, 0, self.width, self.height)

        # Check if change of ROI is needed
        if use_full_FOV is False:
            rect = self.check_and_get_roi()
            cont = self.display_and_check_ROI(rect)

            if not cont:
                print('\n---------Stopping Calibration---------\n')
                return
            else:
                self.mmc.setROI(rect.x, rect.y, rect.width, rect.height)
                self.ROI = (rect.x, rect.y, rect.width, rect.height)

        # Calculate Blacklevel
        logging.debug('Calculating Blacklevel ...')
        self.I_Black = self.calc_blacklevel()
        logging.debug(f'Blacklevel: {self.I_Black}\n')

        # Set LC Wavelength:
        if self.mode == 'retardance':
            self.mmc.setProperty('MeadowlarkLcOpenSource', 'Wavelength', self.wavelength)

        self.opt_Iext()
        self.opt_I0()
        self.opt_I45(0.05, 0.05)
        self.opt_I90(0.05, 0.05)
        self.opt_I135(0.05, 0.05)

        # Calculate Extinction
        self.extinction_ratio = self.calculate_extinction()

        # Write Metadata
        self.write_metadata()

        # Return ROI to full FOV
        if use_full_FOV is False:
            self.mmc.clearROI()

        logging.info("\n=======Finished Calibration=======\n")
        logging.info(f"EXTINCTION = {self.extinction_ratio}")
        logging.debug("\n=======Finished Calibration=======\n")
        logging.debug(f"EXTINCTION = {self.extinction_ratio}")

    def run_4state_calibration(self, param):
        """
        Param is a list or tuple of:
            (swing, wavelength, lc_bounds, black level, <mode>)
            where <mode> is one of 'full','coarse','fine'
        """
        self.swing = param[0]
        self.wavelength = param[1]
        self.meta_file = param[2]
        use_full_FOV = param[3]

        # Get Image Parameters
        self.mmc.snapImage()
        self.mmc.getImage()
        self.height, self.width = self.mmc.getImageHeight(), self.mmc.getImageWidth()
        self.ROI = (0, 0, self.width, self.height)

        # Check if change of ROI is needed
        if use_full_FOV is False:
            rect = self.check_and_get_roi()
            cont = self.display_and_check_ROI(rect)

            if not cont:
                print('\n---------Stopping Calibration---------\n')
                return
            else:
                self.mmc.setROI(rect.x, rect.y, rect.width, rect.height)
                self.ROI = (rect.x, rect.y, rect.width, rect.height)

        # Calculate Blacklevel
        print('Calculating Blacklevel ...')
        self.I_Black = self.calc_blacklevel()
        print(f'Blacklevel: {self.I_Black}\n')

        # Set LC Wavelength:
        if self.mode == 'retardance':
            self.mmc.setProperty('MeadowlarkLcOpenSource', 'Wavelength', self.wavelength)

        self.opt_Iext()
        self.opt_I0()
        self.opt_I60(0.05, 0.05)
        self.opt_I120(0.05, 0.05)

        # Calculate Extinction
        self.extinction_ratio = self.calculate_extinction()

        # Write Metadata
        self.write_metadata()

        # Return ROI to full FOV
        if use_full_FOV is False:
            self.mmc.clearROI()

        print("\n=======Finished Calibration=======\n")
        print(f"EXTINCTION = {self.extinction_ratio}")

    def run_calibration(self, scheme, options):

        if scheme == '5-State':
            self.run_5state_calibration(options)

        elif scheme == '4-State Extinction':
            self.run_4state_calibration(options)

        else:
            raise ValueError('Please define the calibration scheme')

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
        inst_mat = inst_mat.tolist()

        if self.calib_scheme == '4-State':
            data = {'Summary':
                    {'Timestamp': str(datetime.now()),
                     'Acquired Using': '4-State',
                     'Swing (fraction)': self.swing,
                     'Wavelength (nm)': self.wavelength,
                     'BlackLevel': np.round(self.I_Black, 2),
                     'Extinction Ratio': self.extinction_ratio,
                     'ChNames': ["State0", "State1", "State2", "State3"],
                     '[LCA_Ext, LCB_Ext]': [self.lca_ext, self.lcb_ext],
                     '[LCA_0, LCB_0]': [self.lca_0, self.lcb_0],
                     '[LCA_60, LCB_60]': [self.lca_60, self.lcb_60],
                     '[LCA_120, LCB_120]': [self.lca_120, self.lcb_120],
                     'Swing0': self.swing0,
                     'Swing60': self.swing60,
                     'Swing120': self.swing120,
                     'ROI Used (x, y, width, height)': self.ROI,
                     'Instrument_Matrix': inst_mat},
                    'Notes': notes,
                    'Microscope Parameters': microscope_params
                    }

        elif self.calib_scheme == '5-State':
            data = {'Summary':
                    {'Timestamp': str(datetime.now()),
                     'Acquired Using': '5-State',
                     'Swing (fraction)': self.swing,
                     'Wavelength (nm)': self.wavelength,
                     'BlackLevel': np.round(self.I_Black, 2),
                     'Extinction Ratio': self.extinction_ratio,
                     'ChNames': ["State0", "State1", "State2", "State3", "State4"],
                     '[LCA_Ext, LCB_Ext]': [self.lca_ext, self.lcb_ext],
                     '[LCA_0, LCB_0]': [self.lca_0, self.lcb_0],
                     '[LCA_45, LCB_45]': [self.lca_45, self.lcb_45],
                     '[LCA_90, LCB_90]': [self.lca_90, self.lcb_90],
                     '[LCA_135, LCB_135]': [self.lca_135, self.lcb_135],
                     'Swing0': self.swing0,
                     'Swing45': self.swing45,
                     'Swing90': self.swing90,
                     'Swing135': self.swing135,
                     'ROI Used (x, y, width, height)': self.ROI,
                     'Instrument_Matrix': inst_mat},
                    'Notes': notes,
                    'Microscope Parameters': microscope_params
                    }

        if not self.meta_file.endswith('.txt'):
            self.meta_file += '.txt'

        with open(self.meta_file, 'w') as metafile:
            json.dump(data, metafile, indent=1)

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


class CalibrationCurves:

    def __init__(self, path, wavelength = None):

        self.raw_curves = np.load(path)

        # 0V to 20V step size 1 mV
        self.x_range = np.arange(0, 20000, 1)

        # interpolate curves
        self.spline490 = interp1d(self.raw_curves[0, 0], self.raw_curves[0, 1])
        self.spline546 = interp1d(self.raw_curves[1, 0], self.raw_curves[1, 1])
        self.spline630 = interp1d(self.raw_curves[2, 0], self.raw_curves[2, 1])

        self.wavelength = wavelength
        self.curve = None

    def set_wavelength(self, wavelength):
        self.wavelength = wavelength

        # Interpolation of curves beyond this range produce strange results.
        if self.wavelength < 450:
            self.wavelength = 450
        if self.wavelength > 720:
            self.wavelength = 720

        self.create_wavelength_curve()

    def create_wavelength_curve(self):

        if self.wavelength < 490:
            new_a1_y = np.interp(self.x_range, self.x_range, self.spline490(self.x_range))
            new_a2_y = np.interp(self.x_range, self.x_range, self.spline546(self.x_range))

            wavelength_new = 490 + (490 - self.wavelength)
            fact1 = np.abs(490 - wavelength_new) / (546 - 490)
            fact2 = np.abs(546 - wavelength_new) / (546 - 490)

            temp_curve = np.asarray([[i, 2 * new_a1_y[i] - (fact1 * new_a1_y[i] + fact2 * new_a2_y[i])]
                          for i in range(len(new_a1_y))])
            self.spline = interp1d(temp_curve[:, 0], temp_curve[:, 1])
            self.curve = self.spline(self.x_range)

        elif self.wavelength > 630:

            new_a1_y = np.interp(self.x_range, self.x_range, self.spline546(self.x_range))
            new_a2_y = np.interp(self.x_range, self.x_range, self.spline630(self.x_range))

            wavelength_new = 630 + (630 - self.wavelength)
            fact1 = np.abs(630 - wavelength_new) / (630 - 546)
            fact2 = np.abs(546 - wavelength_new) / (630 - 546)

            temp_curve = np.asarray([[i, 2 * new_a1_y[i] - (fact1 * new_a1_y[i] + fact2 * new_a2_y[i])]
                                     for i in range(len(new_a1_y))])
            self.spline = interp1d(temp_curve[:, 0], temp_curve[:, 1])
            self.curve = self.spline(self.x_range)


        elif 490 < self.wavelength < 546:

            new_a1_y = np.interp(self.x_range, self.x_range, self.spline490(self.x_range))
            new_a2_y = np.interp(self.x_range, self.x_range, self.spline546(self.x_range))

            fact1 = np.abs(490 - self.wavelength) / (546 - 490)
            fact2 = np.abs(546 - self.wavelength) / (546 - 490)

            temp_curve = np.asarray([[i, fact1 * new_a1_y[i] + fact2 * new_a2_y[i]] for i in range(len(new_a1_y))])
            self.spline = interp1d(temp_curve[:, 0], temp_curve[:, 1])
            self.curve = self.spline(self.x_range)

        elif 546 < self.wavelength < 630:

            new_a1_y = np.interp(self.x_range, self.x_range, self.spline546(self.x_range))
            new_a2_y = np.interp(self.x_range, self.x_range, self.spline630(self.x_range))

            fact1 = np.abs(546 - self.wavelength) / (630 - 546)
            fact2 = np.abs(630 - self.wavelength) / (630 - 546)

            temp_curve = np.asarray([[i, fact1 * new_a1_y[i] + fact2 * new_a2_y[i]] for i in range(len(new_a1_y))])
            self.spline = interp1d(temp_curve[:, 0], temp_curve[:, 1])
            self.curve = self.spline(self.x_range)

        elif self.wavelength == 490:
            self.curve = self.spline490(self.x_range)
            self.spline = self.spline490

        elif self.wavelength == 546:
            self.curve = self.spline546(self.x_range)
            self.spline = self.spline546

        elif self.wavelength == 630:
            self.curve = self.spline630(self.x_range)
            self.spline = self.spline630

        else:
            raise ValueError(f'Wavelength {self.wavelength} not understood')

    def get_voltage(self, retardance):

        ret_abs = retardance*self.wavelength

        # Since x-step is 1mV starting at 0, returned index = voltage (mV)
        index = np.abs(self.curve - ret_abs).argmin()

        return index

    def get_retardance(self, volt):
        return self.spline(volt) / self.wavelength