import json
import torch
import torchvision
from TorchGraph.DDP_graph import DDPGraph

import torch.optim as optim


import argparse

parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--batchsize", default=32, type=int)
parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--path', type=str, default='DDP.json',
                    help='path')
args = parser.parse_args()

from torchvision import models
module = getattr(models, args.model)().cuda()
example = torch.rand(args.batchsize, 3, 224, 224).cuda()
optimizer = optim.SGD(module.parameters(), lr=0.01)

g = DDPGraph(module, example, optimizer, 'resnet50')
# for node in g.get_output_json():
#     print(node)
# g.dump_graph(args.model+'DDP.json')
g.dump_graph(args.path)


# from transformers import GPT2Model, GPT2Config
# config = GPT2Config(
#         n_embd=1280,
#         n_layer=24,
#         n_head=20,
#     )

# module = GPT2Model(config).cuda()
# example = (torch.LongTensor(1,224).random_() % 1000).cuda()
# optimizer = optim.SGD(module.parameters(), lr=0.01)

# g = TorchGraph(module, example, optimizer, 'GPT2')
# for node in g.get_output_json():
#     print(node)
