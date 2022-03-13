from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.utils import logger
from deepspeed.runtime.pipe import schedule


class PipelineEngine():

    def train_micro_batch_size_per_gpu(self):
        return self._config.train_micro_batch_size_per_gpu
    def gradient_accumulation_steps(self):
        return self._config.gradient_accumulation_steps

    def __init__(self, args, model, model_parameters, has_bool_tensors=False):

        self.module = model
        self.config = (args.deepspeed_config
                        if hasattr(args, "deepspeed_config") else None)
        self._config = DeepSpeedConfig(self.config)

        # We schedule the all-reduces, so disable it in super().backward()
        self.enable_backward_allreduce = False
        self.has_bool_tensors = has_bool_tensors

        # used to disable the pipeline all-reduce when used with 1-bit Adam/1-bit LAMB
        self.pipeline_enable_backward_allreduce = True

        # pipeline step for logging
        self.log_batch_step_id = -1

        self.micro_batch_size = self.train_micro_batch_size_per_gpu()
        self.micro_batches = self.gradient_accumulation_steps()

        # Set Grid and Communication Groups
        self.grid = self.module._grid
        if self.grid.get_global_rank() == 0:
            logger.info(f'CONFIG: micro_batches={self.micro_batches} '
                        f'micro_batch_size={self.micro_batch_size}')

        self.global_rank = self.grid.get_global_rank()

        # assert self.dp_world_size == self.grid.data_parallel_size
        # assert self.train_batch_size() == \
        #     self.micro_batch_size * self.micro_batches * self.grid.data_parallel_size

        #  Set Stage Inf
        self.num_stages = self.grid.pipe_parallel_size
        self.stage_id = self.grid.get_stage_id()
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

        self.sched = schedule.TrainSchedule(micro_batches=self.micro_batches,
                                            stages=self.num_stages,
                                            stage_id=self.stage_id)