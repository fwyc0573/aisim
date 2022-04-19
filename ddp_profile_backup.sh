
NNRANK=0

NNODES=$1
model=$2
batchsize=$3
type=$4

export  PYTHONPATH=/mnt/aisim/SuperScaler/src/
export LD_PRELOAD=/opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so
export LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt/hpcx/nccl_rdma_sharp_plugin/lib
export NCCL_UCX_TLS=rc_x,cuda_copy,cuda_ipc
export NCCL_UCX_RNDV_THRESH=0
export NCCL_UCX_RNDV_SCHEME=get_zcopy
export UCX_RC_MLX5_TM_ENABLE=y
export UCX_MEMTYPE_CACHE=n
export NCCL_IB_PCI_RELAXED_ORDERING=1
export UCX_IB_PCI_RELAXED_ORDERING=on
export NCCL_NET_GDR_LEVEL=5
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_SOCKET_IFNAME=eth0
export NCCL_PLUGIN_P2P=ib
export NCCL_TOPO_FILE=/opt/microsoft/ndv4-topo.xml
export NCCL_DEBUG_SUBSYS=INIT
export NCCL_IB_AR_THRESHOLD=0
python3 -m torch.distributed.launch --nproc_per_node 8 \
    --nnodes $NNODES \
    --node_rank $NNRANK \
    --master_addr msrhpc-msccl-000000 \
    ddp_profile.py \
    --model $model \
    --batchsize $batchsize \
    --type $type
