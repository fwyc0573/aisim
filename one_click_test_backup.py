import os
import json
import argparse
import yaml
import torch
import torch.optim as optim
from TorchGraph.torch_graph import TorchGraph
from TorchGraph.DDP_graph import DDPGraph
from TorchGraph.torch_database import TorchDatabase
from TorchGraph.timer import Timer
from torchvision import models
from ai_simulator.simulator_benchmark.model_zoo import ModelZoo
from ai_simulator.simulator_benchmark.benchmark_tools import BenchmarkTools

# set up the parser
parser = argparse.ArgumentParser(
    prog='python3 simulator_benchmark.py',
    description='Run AI Simulator with 8 models benchmark',
    )

parser.add_argument('-c', '--config_path',
                    dest='config', default='config_torch.yaml',
                    help='config setting.')
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--batchsize", default=32, type=int)
parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--skip-coverage',
                    dest='skip_coverage', action='store_true',
                    help='skip testing the databse coverage')
parser.add_argument('--skip-accuracy',
                    dest='skip_accuracy', action='store_true',
                    help='skip testting the simulator accuracy')
parser.add_argument('--skip-data',
                    dest='skip_data', action='store_true',
                    help='skip testing the dataset')
parser.add_argument('--skip-nccl',
                    dest='skip_nccl', action='store_true',
                    help='skip testing the nccl')
parser.add_argument('--skip-baseline',
                    dest='skip_baseline', action='store_true',
                    help='skip testing the baseline')
parser.add_argument('--skip-graph',
                    dest='skip_graph', action='store_true',
                    help='skip testing the graph')
parser.add_argument('--skip-ddpgraph',
                    dest='skip_ddpgraph', action='store_true',
                    help='skip testing the ddpgraph')
parser.add_argument('--skip-op',
                    dest='skip_op', action='store_true',
                    help='skip testing the op')


nccl_meta_command = '/nccl-tests/build/all_reduce_perf -b 8 -e 1024M -f 2 -g {} > nccl_{}.log'
ddp_meta_command = 'python3 -m torch.distributed.launch --nproc_per_node {} \
    --nnodes 1 \
    --node_rank 0 \
    ddp_profile.py \
    --model {} \
    --batchsize {}'
ddp_graph_command = 'python3 -m torch.distributed.launch --nproc_per_node 1 \
    --nnodes 1 \
    --node_rank 0 \
    ddp_test.py \
    --model {} \
    --path ai_simulator/simulator_benchmark/data/torch/graphs/distributed/{}.json'

def one_click_test(args, config, model_zoo):

    if args.skip_data:
        return
    module = getattr(models, args.model)().cuda()
    example = torch.rand(args.batchsize, 3, 224, 224).cuda()
    optimizer = optim.SGD(module.parameters(), lr=0.01)

    # do all test within one click
    # nccl_test
    if not args.skip_nccl:
        for _, value in config['enviroments'].items():
            cmd = nccl_meta_command.format(value, value)
            os.system(cmd)

    # baseline test
    if not args.skip_baseline:
        for _, value in config['enviroments'].items():
            cmd = ddp_meta_command.format(value, args.model, args.batchsize)
            time = os.popen(cmd).read()
            print(time)
            model_zoo.set_baseline_time(value, args.model, float(time))

    if not args.skip_graph:
        g = TorchGraph(module, example, optimizer, args.model)
        g.dump_graph('ai_simulator/simulator_benchmark/data/torch/graphs/' + args.model + ".json")

    if not args.skip_ddpgraph:
        cmd = ddp_graph_command.format(args.model, args.model, args.batchsize)
        os.system(cmd)
    
    if not args.skip_op:
        timer = Timer(100, args.model)
        g = TorchDatabase(module, example, args.model, timer, optimizer)
        db = (g._get_overall_database())
        json.dump(db,
                  open('ai_simulator/simulator_benchmark/data/torch/database/' + args.model + "_db.json", 'w'),
                  indent=4)

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    model_zoo = ModelZoo(config)

    model_list = model_zoo.get_model_list()
    print(model_list)

    print(config)

    one_click_test(args, config, model_zoo)

    benchmarktools = BenchmarkTools(model_list,
                                    model_zoo,
                                    args.skip_coverage,
                                    args.skip_accuracy,
                                    config)
    benchmarktools.run()
