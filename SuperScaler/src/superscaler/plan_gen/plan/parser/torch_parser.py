# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json
from superscaler.plan_gen.plan.parser.DAG_parser import DAGParser, ParserError
from superscaler.plan_gen.plan.parser.profiler.database_backend import \
     DatabaseBackendLocalFile
from superscaler.plan_gen.plan.parser.profiler.profiler import TFProfiler

class TorchParser(DAGParser):

    def __init__(self, db_type=DatabaseBackendLocalFile, nccl_dataset=None, **kwargs):
        try:
            self.__Profiler = TFProfiler(db_type, **kwargs)
        except Exception:
            self.__Profiler = None
        super().__init__("TorchParser")

        if nccl_dataset is None:
            self._allreduce_dict = {}

            val_list = [[9.53,9.6,9.59,9.67,9.64,9.5,9.52,9.91,10.22,10.81,12.24,16.34,18.68,23.53,34,46.23,48.16,65.87,64.61,73.8,99.49,134.1,227.3,392,740.9,1404.7,2727.4,5228.3,10120,19778,39267,], 
                        [10.01,10.66,10.11,10.52,10.23,10.3,10.21,10.86,11.18,11.59,12.25,13.95,18.18,20.77,26.01,39.8,50.16,64.92,71.36,87.99,125.9,183.3,275.8,517.1,956.2,1870.5,3647.4,7191.1,14127,27867,55169,],
                        [14.36,14.28,14.45,14.52,14.52,14.69,14.77,15.56,16.63,17.73,18.26,19.52,22.34,26.93,29.8,35.28,46.6,62.62,81.73,102.9,141.8,232.2,353.2,562.9,1102.1,2090.2,4142.4,8150,16194,32081,63821,],
                        [29.42,29.13,29.34,29.05,33.14,31.88,36.73,34.7,38.04,36.88,42.13,49.58,53.51,70.84,75.74,77.22,88.9,108.4,129.3,149.8,204.3,383.1,595.3,939.4,1670.6,3058.4,5851.2,11136,21978,43506,86473,],
                        [45.18,45.86,47.6,49.79,50.28,46.81,52.3,52.04,57.02,56.49,59.08,78.45,86.11,110.6,102.2,107.9,115.1,131.1,171.8,218.9,279.2,405.6,711,1152.1,1855.6,3328.6,5993.1,11641,22679,44955,89222,],
                        [416.1373968,412.0354306,415.0058199,410.9038537,468.7557216,450.9333858,519.535234,490.8214707,529.9175039,512.2527571,546.1478037,623.3435803,647.7977846,786.8826751,603.098188,246.0689882,284.9576256,215.3627827,1311.521925,3058.844077,3783.543167,9639.412287,2824.202892,3010.503855,4634.402085,6459.763417,10031.41104,19459.57387,27023.13002,47849.18357]]

            list2 = [2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432,67108864,134217728,268435456,536870912,1073741824,2147483648,4294967296,8589934592]

            # bandwitdh = 200
            # new_bandwidth = 100

            cnt = 0
            for VM in [2, 4, 8, 16, 32, 256]:
                self._allreduce_dict[VM] = {}
                list1 = val_list[cnt]
                cnt+=1
                for val,length in zip(list1, list2):
                    self._allreduce_dict[VM][length] = val
        
        else:
            self._allreduce_dict = nccl_dataset

        # VMs = 256
        # VM = 256
        # cnt = 0
        # self._allreduce_dict[VM] = {}
        # for length in list2:
        #     self._allreduce_dict[VM][length] = 0.0
        # for pre in [2, 4, 8, 16, 32]:
        #     list1 = val_list[cnt]
        #     cnt+=1
        #     for val,length in zip(list1, list2):
        #         self._allreduce_dict[VM][length] += val / pre * VMs / len(val_list)

    def generate_fake_execution_time(self, attrs, gpu):
        if 'ddp' in attrs['op']:
            bucket_size = attrs['bucket_size']
            #print(bucket_size)
            VM = gpu
            for key,value in self._allreduce_dict[VM].items():
                if key < bucket_size:
                    lower_size = key
                    lower_time = value
                else:
                    higher_size = key
                    higher_time = value
                    break
            fake_time = lower_time + (higher_time - lower_time) / (higher_size - lower_size) * (bucket_size - lower_size)
        
        # print(attrs)
        #print(bucket_size,fake_time,lower_size,lower_time,higher_size,higher_time)

        if attrs['op'] == 'ddp_Allreduce':
            return fake_time * 0
        elif attrs['op'] == 'ddp_pre':
            return fake_time * 1
        else:
            return fake_time

    def extract_attrs(self, node, device, gpu):
        if 'attrs' in node:
            attrs = node['attrs']
        else:
            attrs = {}
        attrs['name'] = node['name']
        attrs['op'] = node['op']
        attrs['input'] = node['input_nodes']
        attrs['device'] = device
        attrs['tensor_type'] = node['output_types']
        attrs['output_shape'] = node['output_shapes']
        # TODO add Pytorch database
        execution_time = self.__Profiler.get_execution_time_by_key(node['name'])
        if execution_time is None:
            attrs['execution_time'] = self.generate_fake_execution_time(attrs, gpu)
            # attrs['execution_time'] = lower_time + (higher_time - lower_time) / (higher_size - lower_size) * (bucket_size - lower_size)
            # attrs['execution_time'] += attrs['bucket_size'] / 500
        else:
            attrs['execution_time'] = execution_time * 1000000

        return attrs

    def parse_graph(self, graph, device="device_0", gpu = 1):
        '''
        Parse all nodes from onnx DAG.
        Return the node_list that contains all parsed nodes
        graph_paths: path to onnx DAG
        '''

        self.torch_graph = json.load(open(graph))
        node_list = []
        for node in self.torch_graph:
            node_list.append(self.extract_attrs(node, device, gpu))

        return node_list

