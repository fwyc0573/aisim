import json
import torch
import torchvision
import deepspeed
from deepspeedgraph import PPGraph

import torch.optim as optim
import argparse

parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='alexnet',
                    help='model to benchmark')
parser.add_argument('-p',
                    '--pipeline-parallel-size',
                    type=int,
                    default=2,
                    help='pipeline parallelism')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

from torchvision import models
module = getattr(models, args.model)().cuda()
example = torch.rand(32, 3, 224, 224).cuda()
optimizer = optim.SGD(module.parameters(), lr=0.01)

g = PPGraph(module, example, args)