model=vgg19
gpu=2
bucket_cap_mb=1024
script=ddp_profile.py

# NCCL ENV
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
export LD_PRELOAD=/opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so
export LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt/hpcx/nccl_rdma_sharp_plugin/lib:$LD_LIBRARY_PATH
export NCCL_PLUGIN_P2P=ib
# export NCCL_IB_HCA=mlx5_0
# export NCCL_ALGO=Ring
# export NCCL_SOCKET_IFNAME=eth0
# Some ENV I do not understand yet
# export UCX_IB_PCI_RELAXED_ORDERING=on
# export NCCL_IB_PCI_RELAXED_ORDERING=1
# export NCCL_UCX_TLS=rc_x,cuda_copy,cuda_ipc
# export UCX_MEMTYPE_CACHE=n
# export UCX_RC_MLX5_TM_ENABLE=y
# export NCCL_UCX_RNDV_SCHEME=get_zcopy
# export NCCL_UCX_RNDV_THRESH=0
# export NCCL_IB_AR_THRESHOLD=0
# export NCCL_NET_GDR_READ=1
python3 -m torch.distributed.launch --nproc_per_node $gpu \
    --nnodes 1 \
    --node_rank 0 \
    $script \
    --model $model \
    --bucket_cap_mb $bucket_cap_mb
