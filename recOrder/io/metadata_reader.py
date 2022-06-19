import json


def load_json(path):
    with open(path) as f:
        data = json.load(f)

    return data


class MetadataReader:

    def __init__(self, path: str):
        self.metadata_path = path
        self.json_metadata = load_json(self.metadata_path)

        self.Timestamp = self.get_json_attr('Timestamp')
        self.recOrder_napari_verion = self.get_json_attr('recOrder-napari version')
        self.waveorder_version = self.get_json_attr('waveorder version')
        self.Calibration_scheme = self.get_json_attr('Calibration scheme')
        self.Swing = self.get_json_attr('Swing (waves)')
        self.Wavelength = self.get_json_attr('Wavelength (nm)')
        self.Black_level = self.get_json_attr('Black level')
        self.Extinction_ratio = self.get_json_attr('Extinction ratio')
        self.ROI = self.get_json_attr('ROI (x, y, width, height)')
        self.Channel_names = self.get_json_attr('Channel names')
        self.LCA_retardane = self.get_lc_retardance('LCA')
        self.LCB_retardane = self.get_lc_retardance('LCB')
        self.LCA_voltage = self.get_lc_voltage('LCA')
        self.LCB_voltage = self.get_lc_voltage('LCB')
        self.Swing = self.get_swing()
        self.Notes = self.json_metadata['Notes']
        self.Microscope_parameters = self.json_metadata['Microscope parameters']

    def get_json_attr(self, attr):
        try:
            val = self.json_metadata['Summary'][attr]
        except KeyError:
            val = self.json_metadata['Calibration'][attr]
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
            if lc == 'LCA':
                val = [self.json_metadata['Summary'][f'LCA_{state}, LCB_{state}'][0] for state in states]
            elif lc == 'LCB':
                val = [self.json_metadata['Summary'][f'LCA_{state}, LCB_{state}'][1] for state in states]

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

        lc_voltage = self.json_metadata['Calibration']['LC voltage']
        if lc_voltage:
            val = [lc_voltage[f'{lc}_{state}'] for state in states]
        else:
            val = None

        return val

    def get_swing(self):
        states = self.get_cal_states()

        val = [self.json_metadata['Calibration'][f'Swing_{state}'] for state in states[1:]]

        return val

