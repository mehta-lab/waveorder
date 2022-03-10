from PyQt5.QtCore import pyqtSignal
from recOrder.pipelines.pipeline_manager import PipelineManager
from napari.qt.threading import WorkerBaseSignals, WorkerBase
import time
import os

class ReconstructionSignals(WorkerBaseSignals):
    """
    Custom Signals class that includes napari native signals
    """

    dimension_emitter = pyqtSignal(tuple)
    store_emitter = pyqtSignal(str)
    aborted = pyqtSignal()


class ReconstructionWorker(WorkerBase):

    def __init__(self, calib_window, config):
        super().__init__(SignalsClass=ReconstructionSignals)

        self.calib_window = calib_window
        self.config = config
        self.manager = None

    def _check_abort(self):
        if self.abort_requested:
            self.aborted.emit()
            raise TimeoutError('Stop Requested')

    def work(self):

        self.manager = PipelineManager(self.config, emitter=self.dimension_emitter)

        store_path = os.path.join(self.manager.config.save_dir, self.manager.config.data_save_name)
        store_path = store_path + '.zarr' if not store_path.endswith('.zarr') else store_path
        self.store_emitter.emit(store_path)

        print(f'Beginning Reconstruction...')

        for pt in sorted(self.manager.pt_set):
            start_time = time.time()

            self.manager.try_init_array(pt)

            self._check_abort()

            pt_data = self.manager.data.get_zarr(pt[0])[pt[1]]  # (C, Z, Y, X) virtual

            self._check_abort()

            stokes = self.manager.pipeline.reconstruct_stokes_volume(pt_data)

            self._check_abort()

            stokes = self.manager.pre_processing(stokes)

            self._check_abort()

            birefringence = self.manager.pipeline.reconstruct_birefringence_volume(stokes)

            self._check_abort()

            # will return either phase or fluorescent deconvolved volumes
            deconvolve2D, deconvolve3D = self.manager.pipeline.deconvolve_volume(stokes)

            self._check_abort()

            birefringence, deconvolve2D, deconvolve3D, modified_fluor = self.manager.post_processing(pt_data,
                                                                                             deconvolve2D,
                                                                                             deconvolve3D,
                                                                                             birefringence)

            self._check_abort()

            self.manager.pipeline.write_data(self.manager.indices_map[pt[0]], pt[1], pt_data, stokes,
                                     birefringence, deconvolve2D, deconvolve3D, modified_fluor)

            self._check_abort()

            end_time = time.time()
            print(f'Finishing Reconstructing P = {pt[0]}, T = {pt[1]} ({(end_time - start_time) / 60:0.2f}) min')

            existing_meta = self.manager.writer.store.attrs.asdict().copy()
            existing_meta['Config'] = self.manager.config.yaml_dict
            self.manager.writer.store.attrs.put(existing_meta)

            self._check_abort()
