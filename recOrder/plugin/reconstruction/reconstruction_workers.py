from napari.qt.threading import WorkerBase, WorkerBaseSignals, thread_worker
from recOrder.pipelines.pipeline_manager import PipelineManager


# class ReconstructionSignals(WorkerBaseSignals):
#     pass
#
# class ReconstructionWorker(WorkerBase):
#
#     def __init__(self, config):
#         super().__init__()
#         self._signals = ReconstructionSignals()

@thread_worker
def reconstruct(config):

    config.save_yaml()
    manager = PipelineManager(config, overwrite=True)
    manager.run()


