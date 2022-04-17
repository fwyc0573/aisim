
NNRANK=0

NNODES=$1
model=$2
batchsize=$3
type=$4

export  PYTHONPATH=/mnt/aisim/SuperScaler/src/

python3 -m torch.distributed.launch --nproc_per_node 8 \
    --nnodes $NNODES \
    --node_rank $NNRANK \
    --master_addr msrhpc-msccl-000000 \
    ddp_profile.py \
    --model $model \
    --batchsize $batchsize \
    --type $type
