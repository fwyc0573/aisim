import json
import torch
import torchvision
from TorchGraph.torch_database import TorchDatabase
from torch.autograd import Variable
from TorchGraph.timer import Timer
import torch.optim as optim
import time

import torch.optim as optim


import argparse

parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--type', type=str, default='CV',
                    help='model types')
parser.add_argument("--batchsize", default=32, type=int)
parser.add_argument('--path', type=str, default='DDP.json',
                    help='path')
parser.add_argument('--path_var', type=str, default='DDP.json',
                    help='path')
args = parser.parse_args()

from torchvision import models
import transformer
model = args.model
timer = Timer(100, args.model)
if args.type == 'CV':
    module = getattr(models, args.model)().cuda()
    example = torch.rand(args.batchsize, 3, 224, 224).cuda()
    optimizer = optim.SGD(module.parameters(), lr=0.01)
elif args.type == 'NLP':
    module = getattr(transformer, args.model)().cuda()
    example = (torch.LongTensor(args.batchsize,512).random_() % 1000).cuda()
    optimizer = optim.SGD(module.parameters(), lr=0.01)

g = TorchDatabase(module, example, model, timer, optimizer)
db = (g._get_overall_database())
json.dump(db,
            open(args.path, 'w'),
            indent=4)
var = (g._get_overall_variance())
json.dump(var,
            open(args.path_var, 'w'),
            indent=4)

