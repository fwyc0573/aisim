import json
from typing import Tuple
import torch
import torch.fx
from torch.fx import symbolic_trace, Interpreter
from torch.optim.optimizer import Optimizer
#from transformers.utils.fx import symbolic_trace as transformers_symbolic_trace
from .shape_prop import ShapeProp, TensorMetadata
from .typename import typename
from . import Node
from transformers import PreTrainedModel
import time
from .timer import Timer, make_dot
import torch.optim as optim


class TorchDatabase(torch.fx.Interpreter):
    """
    Generate a torch database with torch.fx.Interpreter
    Basic usage:
        module = torchvision.models.resnet50(pretrained=True).cuda()
        example = torch.rand(1, 3, 224, 224).cuda()
        optimizer = optim.SGD(module.parameters(), lr=0.01)

        timer = Timer(100, 'resnet50')

        g = TorchDatabase(module, example, 'resnet50', 100)
        print(g._forward_database)
        print(g._backward_database)
        print(g._optimizer_database)
    """
    def __init__(self, module: torch.nn.Module, example: torch.tensor, name: str, timer: Timer, optimizer: Optimizer):
        self.module = module
        self.example = example
        self.name = name
        self.timer = timer
        self.optimizer = optimizer
        self._symbolic_traced_module = symbolic_trace(module)
            # ShapeProp(self._symbolic_traced_module).propagate(example)

        self._forward_database = {}
        self._backward_database = {}
        self._optimizer_database = {}
        self._overall_database = {}
        self._get_fp_node_time()
        del self.env
        self._get_bp_node_time()
        self._get_optimizer_node_time()

    def _fp_node_run(self, node: torch.fx.node.Node, *args):
        self.args_iter : Iterator[Any] = iter(args)

        args, kwargs = self.fetch_args_kwargs_from_env(node)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)

        if node.op in self.attr:
            attr = self.attr[node.op]
        else:
            attr = getattr(self, node.op)
            self.attr[node.op] = attr

        if node.name in self._forward_database:
            raise RuntimeError(f"Node {node} repeat in {self.name} graph")
        else:
            
            if node.op == "placeholder":
                #placeholder optime is fixed, thus only need once profiling
                self.env[node] = self.timer._call_function_once(attr, node, args, kwargs)
            else:
                self.env[node] = self.timer._call_function(attr, node, args, kwargs)

        self._forward_database = self.timer._get_database()
        

    def _get_fp_node_time(self, initial_env = None):
        self.env = initial_env if initial_env else {}
        self.attr = {}
        self.timer._init_database()

        for node in self._symbolic_traced_module.graph.nodes:

            if node in self.env:
                continue

            if node.name in self._forward_database:
                continue
            else:
                self._fp_node_run(node, self.example)


    def _get_bp_node_time(self):
        self.timer._init_database()
        y = self.module(self.example)[1]
        make_dot(y, self.module.named_parameters(), self.timer._make_hook)
        y.backward(y)
        self.timer._bp_profiling()
        # make_dot(y, self.module.named_parameters(), self.timer._empty_hook)
        self._backward_database = self.timer._get_database()

    def _get_optimizer_node_time(self):
        self.timer._init_database()
        self.timer._call_optimizer(self.optimizer.zero_grad, "optimizer_zero")
        self.timer._call_optimizer(self.optimizer.step, "optimizer_step")
        self._optimizer_database = self.timer.database

    def _get_overall_database(self):
        self._overall_database = {**self._forward_database, **self._backward_database, **self._optimizer_database}
        return self._overall_database
                
