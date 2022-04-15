import json
import torch
import torchvision
from torch.autograd import Variable
import torch.optim as optim
import time
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd.profiler_util import (_format_time, EventList, FunctionEvent, FunctionEventAvg)

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

model = getattr(models, FLAGS.model)().cuda()
example = torch.rand(32, 3, 224, 224).cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01)

module = DDP(model, device_ids=[local_rank], output_device=local_rank)

from torch.autograd.profiler_util import (_format_time, EventList, FunctionEvent, FunctionEventAvg)

import torch.autograd.profiler as torch_profiler

model = torchvision.models.resnet152(pretrained=False).cuda()
x = torch.rand([32, 3, 224, 224]).cuda()


import pandas as pd

results = {}

for i in range(2):
    for j in range(5):
        y = module(example)
        y.backward(y)
    with torch_profiler.profile(use_cuda=True) as prof:
        y = module(example)
        y.backward(y)

    if i == 0:
        continue

    #print(prof.table(top_level_events_only=True))#, sort_by="self_cuda_time_total"))
    event_list = prof.function_events
    count = 0
    self_cuda_time_total = 0
    for e in event_list:
        if e.self_cuda_time_total != 0:
            key = e.name + str(count)
            if key not in results:
                results[key] = []
            results[key].append(e.self_cuda_time_total)
            results[key].append(str(e))
            self_cuda_time_total += e.self_cuda_time_total
            print(e.name, e.self_cuda_time_total)
            count += 1
    if 'average_step_time' not in results:
        results['average_step_time'] = []
    results['average_step_time'].append(self_cuda_time_total / 1000)
    # results['average_step_time'].append(self_cuda_time_total / 1000)
    # self_cuda_time_total = (sum([e.self_cuda_time_total for e in event_list])) / 1000
    print(local_rank, self_cuda_time_total / 1000)
    # print(count)
# print(results)
df = pd.DataFrame(results)
df.to_csv('log' + str(local_rank) + '.csv')
result = []
for e in event_list:
    if e.self_cuda_time_total != 0:
        result.append(e.name)
        count += 1

# for result in results:
#     print(result)
json.dump(results, open('event' + str(FLAGS.local_rank) + '.json', 'w'), indent=4)

