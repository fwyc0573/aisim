import time
import sys
import torch
import statistics 
import torch.autograd.profiler as torch_profiler
sys.setrecursionlimit(1500)
from transformers import PreTrainedModel

class Timer():

    def __init__(self, profiling_steps: int, name: str):
        self.count = 0
        self.profiling_steps = 1
        self.name = name
        self.database = dict()
        self.variance = dict()
        self.grad_fn_list = []
        self.grad_fn_input_list = []
        self.backward_op_dict = dict()

        self.id_list = []
        self.id_list_bp = []

    def _init_database(self):
        self.database = dict()
        self.variance = dict()

    # def _bp_profiling(self):
    #     # self.count = 0
    #     for var_name, outputs in zip(self.grad_fn_list, self.grad_fn_input_list):
    #         name = var_name['name']
    #         var = var_name['var']
    #         # if name in self.database:
    #         #     raise RuntimeError(f"Node {name} repeat in {self.name} graph")
    #         # else:
    #         # with torch_profiler.profile(use_cuda=True) as prof:
    #         #     var(*outputs)
    #         # cuda_total = float(sum([e.self_cuda_time_total for e in prof.function_events]))
            
    #         # for e in prof.function_events:
    #         #     if e.self_cuda_time_total != 0:
    #         #         self.count += 1
    #         #         print(e.name)
    #         # self.database[name] = cuda_total / 1e6
    #         # self.variance[name] = 0
    #     print(self.count)

    def _bp_profiling(self, function, args):
        with torch_profiler.profile(use_cuda=True) as prof:
            function(args)
        count = -1
        result = 0
        for e in prof.function_events:
            if e.self_cuda_time_total != 0 and e.cpu_parent is None:
                count += 1
            if e.self_cuda_time_total != 0:
                self.database[self.id_list_bp[count]] += e.self_cuda_time_total  / 1e6
                result += e.self_cuda_time_total
        # print(result, count, self.count)

    def _all_profiling(self, module, example):

        id_list = self.id_list + self.id_list_bp
        for i in range(5):
            if isinstance(module, PreTrainedModel):
                y = module(example)
                if 'pooler_output' in y.__dict__:
                    y = y.pooler_output
                else:
                    y = y.last_hidden_state
            else:
                y = module(example)
            y.backward(y)
        with torch_profiler.profile(use_cuda=True) as prof:
            if isinstance(module, PreTrainedModel):
                y = module(example)
                if 'pooler_output' in y.__dict__:
                    y = y.pooler_output
                else:
                    y = y.last_hidden_state
            else:
                y = module(example)
            y.backward(y)
        count = -1
        result = 0
        for e in prof.function_events:
            if e.self_cuda_time_total != 0 and e.cpu_parent is None:
                count += 1
                # print(e.name, id_list[count])
                self.database[id_list[count]] = 0
            if e.self_cuda_time_total != 0:
                if id_list[count] not in self.database:
                    self.database[id_list[count]] = e.self_cuda_time_total  / 1e6
                else:
                    self.database[id_list[count]] += e.self_cuda_time_total  / 1e6
                result += e.self_cuda_time_total
        # print(result, count, self.count, len(id_list))

    def _get_bp_node_op(self, var):
        return type(var).__name__

    def _make_hook(self, var):
        def hook(inputs, outputs):
            if self._get_bp_node_op(var) not in self.backward_op_dict:
                self.backward_op_dict[self._get_bp_node_op(var)] = 0
            else:
                self.backward_op_dict[self._get_bp_node_op(var)] += 1

            name = self._get_bp_node_op(var) + str(self.backward_op_dict[self._get_bp_node_op(var)])
            self.id_list_bp.append(name)

            if name in self.database:
                raise RuntimeError(f"Node {name} repeat in {self.name} graph")
            else:
                self.count += 1
                # with torch_profiler.profile(use_cuda=True) as prof:
                #     var(*outputs)
                # torch.cuda.synchronize()
                # cuda_total = float(sum([e.self_cuda_time_total for e in prof.function_events]))
                
                # for e in prof.function_events:
                #     if e.self_cuda_time_total != 0:
                #         self.count += 1
                #         print(e.name, e.self_cuda_time_total)
                # for i in range(10):
                #     var(*outputs)
    
                # data_list = []
                # for _ in range(10):
                #     torch.cuda.synchronize()
                #     ss = time.perf_counter()
                #     for i in range(10):
                #         var(*outputs)
                #     torch.cuda.synchronize()
                #     ee = time.perf_counter()
                #     data_list.append((ee-ss))
                # self.database[name] = statistics.mean(data_list) / 10
                # self.variance[name] = statistics.variance(data_list) / 100

                # self.database[name] = 0 #cuda_total / 1e6
                # self.variance[name] = 0
            
            # self.grad_fn_list.append({'var':var, 'name':self._get_bp_node_op(var) +str(self.backward_op_dict[self._get_bp_node_op(var)])})
            # self.grad_fn_input_list.append(outputs)

        return hook

    def _empty_hook(self, var):
        def hook(inputs, outputs):
            pass
        return hook

    def _call_function(self, function, node, args, kwargs):
        with torch_profiler.profile(use_cuda=True) as prof:
            output = function(node.target, args, kwargs)
        cuda_total = float(sum([e.self_cuda_time_total for e in prof.function_events]))
        for e in prof.function_events:
            if e.self_cuda_time_total != 0  and e.cpu_parent is None:
                self.count += 1
                self.id_list.append(node.name)
        self.database[node.name] = 0
        self.variance[node.name] = 0
        return output

    def _call_function_profile(self, function, args):
        function(args)
        with torch_profiler.profile(use_cuda=True) as prof:
            function(args)
        count = 0
        result = 0
        for e in prof.function_events:
            if e.self_cuda_time_total != 0:
                self.database[self.id_list[count]] += e.self_cuda_time_total  / 1e6
                result += e.self_cuda_time_total
                count += 1

    def _call_function_once(self, function, node, args, kwargs):
        output = function(node.target, args, kwargs)
        self.database[node.name] = 0
        self.variance[node.name] = 0
        return output


    def _call_optimizer(self, function, name):
        
        function()
        with torch_profiler.profile(use_cuda=True) as prof:
            function()
        cuda_total = 0
        for e in prof.function_events:
            if e.self_cuda_time_total != 0  and e.cpu_parent is None:
                # print(e.name, e.self_cuda_time_total)
                cuda_total += e.self_cuda_time_total
        # cuda_total = float(sum([e.self_cuda_time_total for e in prof.function_events]))
        self.database[name] = cuda_total / 1e6
        self.variance[name] = 0
    
    def _get_database(self):
        return self.database

    def _get_variance(self):
        return self.variance


def make_dot(var, params, hook):
    """ Produces Graphviz representation of PyTorch autograd graph.
    
    Blue nodes are trainable Variables (weights, bias).
    Orange node are saved tensors for the backward pass.
    
    Args:
        var: output Variable
        params: list of (name, Parameters)
    """
    
    param_map = {id(v): k for k, v in params}

    seen = set()
    
    def add_nodes(var):
        if var not in seen:
            
            node_id = str(id(var))
            
            var.register_hook(hook(var))
            # print(var.name())
            # if(var.name() == "CudnnConvolutionBackward"):
            #     print(var.getAttribute("padding"))
            seen.add(var)
            
            if hasattr(var, 'next_functions'):
                # print(var.name(), var.next_functions)
                for u in var.next_functions:
                    if u[0] is not None:
                        add_nodes(u[0])
                        
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    add_nodes(t)

    add_nodes(var.grad_fn)
