import json
import torch
import torchvision
from TorchGraph.torch_graph import TorchGraph

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
args = parser.parse_args()

from torchvision import models
import transformer
if args.type == 'CV':
    module = getattr(models, args.model)().cuda()
    example = torch.rand(args.batchsize, 3, 224, 224).cuda()
    optimizer = optim.SGD(module.parameters(), lr=0.01)
elif args.type == 'NLP':
    module = getattr(transformer, args.model)().cuda()
    example = (torch.LongTensor(args.batchsize,512).random_() % 1000).cuda()
    optimizer = optim.SGD(module.parameters(), lr=0.01)


g = TorchGraph(module, example, optimizer, 'GPT2')
g.dump_graph(args.path)

