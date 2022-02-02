from abc import ABC, abstractmethod


class PipelineInterface(ABC):
    """
    An interface for all pipeline classes

    """

    @abstractmethod
    def reconstruct_stokes_volume(self, data):
        pass

    @abstractmethod
    def deconvolve_volume(self, stokes):
        pass

    @abstractmethod
    def reconstruct_birefringence_volume(self, stokes):
        pass

    @abstractmethod
    def write_data(self, p, t, pt_data, stokes, birefringence, deconvolve2D, deconvolve3D, modified_fluor):
        pass