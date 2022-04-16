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
import transformer
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
parser.add_argument("--sequencelength", default=512, type=int)
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


nccl_meta_command = '/opt2/nccl-tests/build/all_reduce_perf -b 8 -e 1024M -f 2 -g {} > nccl_{}.log'
ddp_meta_command = 'python3 -m torch.distributed.launch --nproc_per_node {} \
    --nnodes 1 \
    --node_rank 0 \
    ddp_profile.py \
    --model {} \
    --batchsize {} \
    --type {}'
graph_command = 'python3 \
    torch_graph_test.py \
    --model {} \
    --path ai_simulator/simulator_benchmark/data/torch/graphs/{}.json \
    --batchsize {} \
    --type {}'
ddp_graph_command = 'python3 -m torch.distributed.launch --nproc_per_node 1 \
    --nnodes 1 \
    --node_rank 0 \
    ddp_test.py \
    --model {} \
    --path ai_simulator/simulator_benchmark/data/torch/graphs/distributed/{}.json \
    --batchsize {} \
    --type {}'

def nccl_test(args, config):
    for _, value in config['enviroments'].items():
        cmd = nccl_meta_command.format(value, value)
        os.system(cmd)

def baseline_test(args, config):
    for model in args.model_list:
        print(model, args.model_list)
        for _, value in config['enviroments'].items():
            cmd = ddp_meta_command.format(value,
                                          args.model_zoo.get_sub_models(model), 
                                          args.model_zoo.get_batch_size(model),
                                          args.model_zoo.get_type(model))
            time = os.popen(cmd).read().split('\n')[0]
            args.model_zoo.set_baseline_time(value, model, float(time))
    args.model_zoo.dump_baseline()
    print(args.model_zoo.get_baseline())

def TorchGraph_test(args, config):
    for model in args.model_list:
        cmd = graph_command.format(args.model_zoo.get_sub_models(model),
                                   model,
                                   args.model_zoo.get_batch_size(model),
                                   args.model_zoo.get_type(model))
        print(cmd)
        os.system(cmd)
    # for model in args.model_list:
    #     if args.model_zoo.get_type(model) == 'CV':
    #         module = getattr(models, args.model_zoo.get_sub_models(model))().cuda()
    #         example = torch.rand(args.model_zoo.get_batch_size(model), 3, 224, 224).cuda()
    #         optimizer = optim.SGD(module.parameters(), lr=0.01)
    #     elif args.model_zoo.get_type(model) == 'NLP':
    #         module = getattr(transformer, args.model_zoo.get_sub_models(model))().cuda()
    #         example = (torch.LongTensor(args.model_zoo.get_batch_size(model),512).random_() % 1000).cuda()
    #         optimizer = optim.SGD(module.parameters(), lr=0.01)
    #     g = TorchGraph(module, example, optimizer, model)
    #     g.dump_graph('ai_simulator/simulator_benchmark/data/torch/graphs/' + model + ".json")

    #     del(module)
    #     del(g)

def ddpgraph_test(args, config):
    for model in args.model_list:
        cmd = ddp_graph_command.format(args.model_zoo.get_sub_models(model),
                                       model,
                                       args.model_zoo.get_batch_size(model),
                                       args.model_zoo.get_type(model))
        print(cmd)
        os.system(cmd)

def op_test(args, config):
    for model in args.model_list:
        timer = Timer(100, args.model)
        if args.model_zoo.get_type(model) == 'CV':
            module = getattr(models, args.model_zoo.get_sub_models(model))().cuda()
            example = torch.rand(args.model_zoo.get_batch_size(model), 3, 224, 224).cuda()
            optimizer = optim.SGD(module.parameters(), lr=0.01)
        elif args.model_zoo.get_type(model) == 'NLP':
            module = getattr(transformer, args.model_zoo.get_sub_models(model))().cuda()
            example = (torch.LongTensor(args.model_zoo.get_batch_size(model),512).random_() % 1000).cuda()
            optimizer = optim.SGD(module.parameters(), lr=0.01)
        
        g = TorchDatabase(module, example, model, timer, optimizer)
        db = (g._get_overall_database())
        json.dump(db,
                  open('ai_simulator/simulator_benchmark/data/torch/database/' + model + "_db.json", 'w'),
                  indent=4)
        var = (g._get_overall_variance())
        json.dump(var,
                  open('ai_simulator/simulator_benchmark/data/torch/database/' + model + "_var.json", 'w'),
                  indent=4)

def one_click_test(args, config):

    if args.skip_data:
        return

    # do all test within one click

    # nccl_test
    if not args.skip_nccl:
        nccl_test(args, config)

    # baseline test
    if not args.skip_baseline:
        baseline_test(args, config)

    # TorchGraph test
    if not args.skip_graph:
        TorchGraph_test(args, config)

    # ddpgraph test
    if not args.skip_ddpgraph:
        ddpgraph_test(args, config)

    if not args.skip_op:
        op_test(args, config)

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_zoo = ModelZoo(config)
    args.model_zoo = model_zoo

    model_list = model_zoo.get_model_list()
    args.model_list = model_list

    print(model_list)
    print(config)

    one_click_test(args, config)

    benchmarktools = BenchmarkTools(args.model_list,
                                    args.model_zoo,
                                    args.skip_coverage,
                                    args.skip_accuracy,
                                    config)
    benchmarktools.run()
