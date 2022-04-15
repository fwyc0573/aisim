import json
import torch
import torchvision
from torch.autograd import Variable
import torch.optim as optim
import time
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# from transformers import BertModel, BertConfig

# config = BertConfig(
#     hidden_size=1024,
#     num_hidden_layers=24,
#     num_attention_heads=16,
#     intermediate_size=4096
# )

### 2. 初始化我们的模型、数据、各种配置  ####
# DDP：从外部得到local_rank参数
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--repeat", default=20, type=int)
parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--bucket_cap_mb', type=int, default=25,
                    help='ddp bucket_cap_mb')
parser.add_argument("--batchsize", default=32, type=int)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank
bucket_cap_mb = FLAGS.bucket_cap_mb
# DDP：DDP backend初始化
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端


# module = models.resnet152().to(local_rank)
# # DDP: 构造DDP model
# example = torch.rand(32, 3, 224, 224).cuda()
# optimizer = optim.SGD(module.parameters(), lr=0.01)

from torchvision import models

module = getattr(models, FLAGS.model)().cuda()
example = torch.rand(FLAGS.batchsize, 3, 224, 224).cuda()
optimizer = optim.SGD(module.parameters(), lr=0.01)

module = DDP(module, device_ids=[local_rank], output_device=local_rank, bucket_cap_mb=bucket_cap_mb)
y = module(example)
test = torch.ones_like(y)
y.backward(test)

def benchmark_step():
    # optimizer.zero_grad()
    output = module(example)
    output.backward(output)
    # optimizer.step()

for i in range(10):
    benchmark_step()
torch.cuda.synchronize()
ss = time.perf_counter()
for i in range(FLAGS.repeat):
    benchmark_step()
torch.cuda.synchronize()
ee = time.perf_counter()

if FLAGS.local_rank == 0:
    print((ee - ss)/FLAGS.repeat*1000)
