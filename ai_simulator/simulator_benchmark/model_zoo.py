# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
#from superscaler.plan_gen import TFParser, TorchParser

class ModelZoo():
    ''' Store the available models in benchmark
    '''
    # Use a dict to store valid platform_type string
    __valid_platform = ['tensorflow', 'pytorch', 'torch']


    def __init__(self, config):
        self.__init_parser(config)
        self.__init_models(config)

    def __init_parser(self, config):
        if 'platform' not in config:
            raise ValueError("platform is not provided in config")
        if config['platform'] not in self.__valid_platform:
            raise ValueError("platform is invalid")

        if config['platform'] == 'tensorflow':
            from superscaler.plan_gen import TFParser as Parser
        elif config['platform'] == 'pytorch' or config['platform'] == 'torch':
            from superscaler.plan_gen import TorchParser as Parser
        else:
            raise ValueError("platform is invalid")

        self.parser = Parser

    def __init_models(self, config):
        self.config = config
        self.__models = []
        self.__sub_models = {}
        self.__types = {}
        self.__batch_sizes = {}
        self.__graph_path = {}
        self.__graph_path_multi_gpu = {}
        self.__database_path = {}
        self.__nccl_dataset = {}

        if 'baseline_path' in config:
            self.__baseline = json.load((open(config['baseline_path'])))
        else:
            self.__baseline = {}

        for c, gpu in config['enviroments'].items():
            self.__nccl_dataset[gpu] = {}
            nccl_path = config['nccl_path'] + 'nccl_' + str(gpu) + '.log'
            with open(nccl_path, 'r') as f:
                for line in f:
                    if 'sum' in line:
                        data = line.split()
                        self.__nccl_dataset[gpu][int(data[1])] = float(data[-4])

        for model in config['tasks']:
            if model in self.__models:
                raise ValueError("Idential tasks \"{}\" are simulated twice".format(task))
            else:
                self.__models.append(model)
                task = config['tasks'][model]

                if 'model' not in task:
                    raise ValueError("Task \"{}\" are ininlized without model".format(task))
                else:
                    self.set_sub_models(model, task['model'])
    
                if 'type' not in task:
                    self.set_type(model, 'CV')
                else:
                    self.set_type(model, task['type'])

                if 'batch_size' not in task:
                    raise ValueError("Task \"{}\" are ininlized without batch_size".format(task))
                else:
                    self.set_batch_size(model, task['batch_size'])

                if 'graph_path' not in task:
                    raise ValueError("Task \"{}\" are ininlized without graph".format(task))
                else:
                    self.set_graph_path(model, task['graph_path'])

                if 'graph_path_multi_gpu' not in task:
                    raise ValueError("Task \"{}\" are ininlized without graph_path_multi_gpu".format(task))
                else:
                    self.set_graph_path_multi_gpu(model, task['graph_path_multi_gpu'])

                if 'database_path' not in task:
                    raise ValueError("Task \"{}\" are ininlized without database_path".format(task))
                else:
                    self.set_database_path(model, task['database_path'])

    def exist_model(self, model):
        if model in self.__models:
            return True
        else:
            return False

    def get_model_list(self):
        return self.__models

    def set_graph_path(self, model, graph_path):
        self.__graph_path[model] = graph_path

    def get_graph_path(self, model):
        return self.__graph_path[model]

    def set_sub_models(self, model, sub_model):
        self.__sub_models[model] = sub_model

    def get_sub_models(self, model):
        return self.__sub_models[model]

    def set_type(self, model, type):
        self.__types[model] = type

    def get_type(self, model):
        return self.__types[model]

    def set_batch_size(self, model, batch_size):
        self.__batch_sizes[model] = batch_size

    def get_batch_size(self, model):
        return self.__batch_sizes[model]

    def set_graph_path_multi_gpu(self, model, graph_path):
        self.__graph_path_multi_gpu[model] = graph_path

    def get_graph_path_multi_gpu(self, model):
        return self.__graph_path_multi_gpu[model]

    def set_database_path(self, model, database_path):
        self.__database_path[model] = database_path

    def get_database_path(self, model):
        return self.__database_path[model]

    def get_baseline(self):
        return self.__baseline

    def set_baseline_time(self, gpu, model, baseline_time):
        if str(gpu) in self.__baseline:
            self.__baseline[str(gpu)][model] = baseline_time
        else:
            self.__baseline[str(gpu)] = {}
            self.__baseline[str(gpu)][model] = baseline_time
    
    def dump_baseline(self):
        json.dump(self.__baseline, open(self.config['baseline_path'], 'w'))

    def get_baseline_time(self, gpu, model):
        baseline_time = 1.0
        if str(gpu) in self.__baseline:
            if model in self.__baseline[str(gpu)]:
                baseline_time = self.__baseline[str(gpu)][model]
        return baseline_time

    def get_nccl_dataset(self):
        return self.__nccl_dataset
