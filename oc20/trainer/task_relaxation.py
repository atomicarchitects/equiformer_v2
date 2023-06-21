from ocpmodels.tasks.task import BaseTask
from ocpmodels.common.registry import registry

      
@registry.register_task("run-relaxations")
class MyRelaxationTask(BaseTask):
    def run(self):
        assert (
            self.trainer.relax_dataset is not None
        ), "Relax dataset is required for making predictions"
        assert self.config["checkpoint"]
        self.trainer.run_relaxations()