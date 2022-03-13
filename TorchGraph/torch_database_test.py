import json
import torch
import torchvision
from torch_database import TorchDatabase
from torch.autograd import Variable
from timer import Timer
import torch.optim as optim
import time

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

# module = torchvision.models.resnet101(pretrained=True).cuda()
# optimizer = optim.SGD(module.parameters(), lr=0.01)
# example = torch.rand(32, 3, 224, 224).cuda()

timer = Timer(100, 'alexnet')
g = TorchDatabase(module, example, 'alexnet', timer, optimizer)

db = (g._get_overall_database())
json.dump(db,
          open(args.model + 'db.json', 'w'),
          indent=4)
