import json
import os
from natsort import natsorted


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)

    return data


def get_last_metadata_file(path):
    last_metadata_file = natsorted([file for file in os.listdir(path) if file.startswith('calibration_metadata')])[-1]
    return os.path.join(path, last_metadata_file)


class MetadataReader:
    """
    Calibration metadata reader class. Helps load metadata from different metadata formats and naming conventions
    """

    def __init__(self, path: str):
        """

        Parameters
        ----------
        path: full path to calibration metadata
        """
        self.metadata_path = path
        self.json_metadata = load_json(self.metadata_path)

        self.Timestamp = self.get_summary_calibration_attr('Timestamp')
        self.recOrder_napari_verion = self.get_summary_calibration_attr('recOrder-napari version')
        self.waveorder_version = self.get_summary_calibration_attr('waveorder version')
        self.Calibration_scheme = self.get_calibration_scheme()
        self.Swing = self.get_swing()
        self.Wavelength = self.get_summary_calibration_attr('Wavelength (nm)')
        self.Black_level = self.get_black_level()
        self.Extinction_ratio = self.get_extinction_ratio()
        self.ROI = tuple(self.get_roi()) # JSON does not preserve tuples
        self.Channel_names = self.get_channel_names()
        self.LCA_retardance = self.get_lc_retardance('LCA')
        self.LCB_retardance = self.get_lc_retardance('LCB')
        self.LCA_voltage = self.get_lc_voltage('LCA')
        self.LCB_voltage = self.get_lc_voltage('LCB')
        self.Swing_measured = self.get_swing_measured()
        self.Notes = self.get_notes()
        self.Microscope_parameters = self.get_microscope_parameters()

    def get_summary_calibration_attr(self, attr):
        try:
            val = self.json_metadata['Summary'][attr]
        except KeyError:
            try:
                val = self.json_metadata['Calibration'][attr]
            except KeyError:
                val = None
        return val

    def get_cal_states(self):
        if self.Calibration_scheme == '4-State':
            states = ['ext', '0', '60', '120']
        elif self.Calibration_scheme == '5-State':
            states = ['ext', '0', '45', '90', '135']
        return states

    def get_lc_retardance(self, lc):
        """

        Parameters
        ----------
        lc: 'LCA' or 'LCB'

        Returns
        -------

        """
        states = self.get_cal_states()

        val = None
        try:
            val = [self.json_metadata['Calibration']['LC retardance'][f'{lc}_{state}'] for state in states]
        except KeyError:
            states[0] = 'Ext'
            if lc == 'LCA':
                val = [self.json_metadata['Summary'][f'[LCA_{state}, LCB_{state}]'][0] for state in states]
            elif lc == 'LCB':
                val = [self.json_metadata['Summary'][f'[LCA_{state}, LCB_{state}]'][1] for state in states]

        return val

    def get_lc_voltage(self, lc):
        """

        Parameters
        ----------
        lc: 'LCA' or 'LCB'

        Returns
        -------

        """
        states = self.get_cal_states()

        val = None
        if 'Calibration' in self.json_metadata:
            lc_voltage = self.json_metadata['Calibration']['LC voltage']
            if lc_voltage:
                val = [self.json_metadata['Calibration']['LC voltage'][f'{lc}_{state}'] for state in states]

        return val

    def get_swing(self):
        try:
            val = self.json_metadata['Calibration']['Swing (waves)']
        except KeyError:
            val = self.json_metadata['Summary']['Swing (fraction)']
        return val

    def get_swing_measured(self):
        states = self.get_cal_states()
        try:
            val = [self.json_metadata['Calibration'][f'Swing_{state}'] for state in states[1:]]
        except KeyError:
            val = [self.json_metadata['Summary'][f'Swing{state}'] for state in states[1:]]

        return val

    def get_calibration_scheme(self):
        try:
            val = self.json_metadata['Calibration']['Calibration scheme']
        except KeyError:
            val = self.json_metadata['Summary']['Acquired Using']
        return val

    def get_black_level(self):
        try:
            val = self.json_metadata['Calibration']['Black level']
        except KeyError:
            val = self.json_metadata['Summary']['BlackLevel']
        return val

    def get_extinction_ratio(self):
        try:
            val = self.json_metadata['Calibration']['Extinction ratio']
        except KeyError:
            val = self.json_metadata['Summary']['Extinction Ratio']
        return val

    def get_roi(self):
        try:
            val = self.json_metadata['Calibration']['ROI (x, y, width, height)']
        except KeyError:
            val = self.json_metadata['Summary']['ROI Used (x, y, width, height)']
        return val

    def get_channel_names(self):
        try:
            val = self.json_metadata['Calibration']['Channel names']
        except KeyError:
            val = self.json_metadata['Summary']['ChNames']
        return val

    def get_microscope_parameters(self):
        try:
            val = self.json_metadata['Microscope parameters']
        except KeyError:
            try:
                val = self.json_metadata['Microscope Parameters']
            except KeyError:
                val = None
        return val

    def get_notes(self):
        try:
            val = self.json_metadata['Notes']
        except KeyError:
            val = None
        return val

