import os
import time
import torch
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import schedule
from torch.profiler import profile, record_function, ProfilerActivity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Standalone Allreduce')
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank. ")
    parser.add_argument('--backend', type=str, default='nccl', help='distributed backend')
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=1120)
    parser.add_argument("--profile", type=int, default=0, help="Whether to enable torch profiler.")
    parser.add_argument(
        '--p',
        type=int,
        default=5000,
        required=False,
        help='The P dim of Tensor(P, Q) to transfer in NCCL AllReduce.',
    )
    parser.add_argument(
        '--q',
        type=int,
        default=1400,
        required=False,
        help='The Q dim of Tensor(P, Q) to transfer in NCCL AllReduce.',
    )
    parser.add_argument(
        '--num_steps',
        type=int,
        default=10,
        required=False,
        help='The iteration number.',
    )

    args = parser.parse_args()
    P = args.p
    Q = args.q
    local_rank = args.local_rank
    torch.manual_seed(args.random_seed)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.backend)
    shape = [P, Q]
    message = torch.randn(*shape).cuda()
    
    profile_schedule = torch.profiler.schedule(
        wait=2,
        warmup=2,
        active=2,
        repeat=1
    )
    worker_name="allreduce_P{}_Q{}".format(P,Q)
    with torch.profiler.profile(
            schedule=profile_schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("/mnt", worker_name),
            record_shapes= True,
            with_flops =True,
            with_stack=True
        ) as profiler:

        start = time.perf_counter()
        for _ in range(args.num_steps):
            dist.all_reduce(message, op=dist.ReduceOp.SUM, async_op=True)
            profiler.step()
        torch.cuda.synchronize()
        end = time.perf_counter()
        elapsed = end - start
    
    print("gpu {} avg. elapsed time: {} ms".format(dist.get_rank(), elapsed*1000/args.num_steps))
