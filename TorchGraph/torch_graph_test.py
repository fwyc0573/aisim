import json
import torch
import torchvision
from torch_graph import TorchGraph

import torch.optim as optim


import argparse

parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
args = parser.parse_args()

from torchvision import models
module = getattr(models, args.model)().cuda()
example = torch.rand(32, 3, 224, 224).cuda()
optimizer = optim.SGD(module.parameters(), lr=0.01)

g = TorchGraph(module, example, optimizer, 'GPT2')
# for node in g.get_output_json():
#     print(node)
g.dump_graph(args.model + "test.json")
