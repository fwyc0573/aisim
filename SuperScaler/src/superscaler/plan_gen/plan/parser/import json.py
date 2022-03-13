import json
import torch
import torchvision
from torch.autograd import Variable
from timer import Timer
import torch.optim as optim
import time
import argparse
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


### 2. 初始化我们的模型、数据、各种配置  ####
# DDP：从外部得到local_rank参数
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--buffersize", default= 2 ** 10, type=int)
parser.add_argument('--no-allreduce', action='store_true', default=False)

FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

# DDP：DDP backend初始化vi
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端


buffer_size = FLAGS.buffersize
global_group = torch.distributed.new_group()


data = torch.zeros(buffer_size).cuda()
conv = nn.Conv2d(32, 32, kernel_size=3, padding=1).cuda()
example = torch.rand(32, 32, 224, 224).cuda()

def benchmark_step():
    if FLAGS.no_allreduce:
        output = conv(example)
    else:
        work = torch.distributed.all_reduce(
            data, group=global_group, async_op=True).get_future()
        # output = conv(example)
        work.wait()


for i in range(10):
    benchmark_step()

ss = time.perf_counter()
for i in range(10):
    benchmark_step()
ee = time.perf_counter()

print(FLAGS.buffersize)
print("real ", (ee - ss))
