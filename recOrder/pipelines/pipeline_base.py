

class Pipeline_Structure:
    """
    Builder class that all pipelines must inherit and follow
    """
    writer = None

    def reconstruct_stokes_volume(self, data):
        pass

    def reconstruct_phase_volume(self, stokes):
        pass

    def reconstruct_birefringence_volume(self, stokes):
        pass

    def write_data(self, pt, pt_data, stokes, birefringence, phase2D, phase3D, registered_stacks):
        pass