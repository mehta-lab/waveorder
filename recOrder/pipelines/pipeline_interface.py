
"""
This is an informal interface.  If the codebase grows and we wish to have more rigid enforcement, we can implement
formal interfaces:
https://realpython.com/python-interface/#formal-interfaces
"""


class PipelineInterface:
    """
    An interface for all pipeline classes

    """

    def reconstruct_stokes_volume(self, data):
        pass

    def reconstruct_phase_volume(self, stokes):
        pass

    def reconstruct_birefringence_volume(self, stokes):
        pass

    def write_data(self, pt, pt_data, stokes, birefringence, phase2D, phase3D, registered_stacks):
        pass

    def writer(self):
        pass
