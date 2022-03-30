import torch
import torchvision
from torch.autograd.profiler_util import (_format_time, EventList, FunctionEvent, FunctionEventAvg)

import torch.autograd.profiler as torch_profiler

model = torchvision.models.vgg19(pretrained=False).cuda()
x = torch.rand([32, 3, 224, 224]).cuda()

# Layer does not have to be a leaf layer
# paths = [("AlexNet", "features", "3"), ("AlexNet", "classifier")]

test = torch.ones_like(model(x))
for i in range(1):
    y = model(x)
    y.backward(test)
y = model(x)
with torch_profiler.profile(use_cuda=True) as prof:
    y.backward(test)

print(prof.table(top_level_events_only=True))#, sort_by="self_cuda_time_total"))
event_list = prof.function_events
count = 0
for e in event_list:
    if e.self_cuda_time_total != 0 and e.cpu_parent is None:
        print(e.name,  e.self_cuda_time_total)
        count += 1

self_cuda_time_total = _format_time(sum([e.self_cuda_time_total for e in event_list]))
print(self_cuda_time_total)
print(count)

# print torchprof data
# print(prof)

# count1 = 0
# self_cuda_total = 0
# cuda_total = 0
# for path, _ in prof.trace_profile_events.items():
#     events = [te for t_events in prof.trace_profile_events[path] for te in t_events]
#     print(path)
#     for e in events:
#         if e.self_cuda_time_total != 0:
#             # print(e.name, e.self_cuda_time_total)
#             count1+=1
#     self_cuda_total += (sum([getattr(e, "self_cuda_time_total", 0) for e in events]))
#     cuda_total += (sum([e.cuda_time_total for e in events]))
# print(self_cuda_total)
# print(count, count1)



# events = [te for _, t_events in prof.trace_profile_events.items() for te in t_events]
# self_cuda_total = sum([getattr(e, "self_cuda_time_total", 0) for e in events])
# cuda_total=sum([e.cuda_time_total for e in events]),
# print(self_cuda_total, cuda_total)

# for key, value in prof.trace_profile_events.items():

# print(prof.trace_profile_events)

# trace, event_lists_dict = prof.raw()

# print(event_lists_dict[trace[2].path][0])
