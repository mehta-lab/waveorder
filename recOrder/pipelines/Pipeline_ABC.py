

class Pipeline_Builder:
    """
    Builder class that all pipelines must inherit and follow
    """

    def reconstruct_stokes_volume(self):
        pass

    def reconstruct_phase_volume(self):
        pass

    def reconstruct_birefringence_volume(self):
        pass

    def write_data(self):
        pass