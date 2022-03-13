import torch
import torch.fx
from module import PipelineModule
from engine import PipelineEngine

def join_layers(vision_model):
    layers = [
        *vision_model.features,
        vision_model.avgpool,
        lambda x: torch.flatten(x, 1),
        *vision_model.classifier,
    ]
    return layers

class PPGraph():
    """
    Visualize a torch model with torch.fx.Graph
    Basic usage:
        g = TorchGraph(module, 'resnet18')
        with open("a.svg", "w") as f:
            f.write(g.get__graph().create_svg())
    """
    # _INSTRUCTION_MAP = {
    #     schedule.OptimizerStep: _exec_optimizer_step,
    #     schedule.ReduceGrads: _exec_reduce_grads,
    #     schedule.ReduceTiedGrads: _exec_reduce_tied_grads,
    #     schedule.LoadMicroBatch: _exec_load_micro_batch,
    #     schedule.ForwardPass: _exec_forward_pass,
    #     schedule.BackwardPass: _exec_backward_pass,
    #     schedule.SendActivation: _exec_send_activations,
    #     schedule.RecvActivation: _exec_recv_activations,
    #     schedule.SendGrad: _exec_send_grads,
    #     schedule.RecvGrad: _exec_recv_grads,
    # }

    def __init__(self, module: torch.nn.Module, example: torch.tensor, args, part='parameters'):
    
        print(module)
        print(example)

        self.module = PipelineModule(layers=join_layers(module),
                                    loss_fn=torch.nn.CrossEntropyLoss(),
                                    num_stages=args.pipeline_parallel_size,
                                    partition_method=part,
                                    activation_checkpoint_interval=0,
                                    global_rank=0,
                                    world_size=2,
                                    local_rank=0
                                    )

        model_parameters=[p for p in module.parameters() if p.requires_grad]
        self.engine = PipelineEngine(args=args,
                                    model=self.module,
                                    model_parameters=model_parameters)
        self.sched = self.engine.sched

        for step_cmds in self.sched:
            # For each instruction in the step
            for cmd in step_cmds:
                print(cmd)