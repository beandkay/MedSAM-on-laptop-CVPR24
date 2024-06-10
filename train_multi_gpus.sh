#!/bin/bash


GPUS_PER_NODE=2 # <- Specify the number of GPUs per machine here

## Master node setup
MAIN_HOST=`hostname -s`
export MASTER_ADDR=localhost

# Get a free port using python
export MASTER_PORT=$(python - <<EOF
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('', 0))  # OS will allocate a free port
free_port = sock.getsockname()[1]
sock.close()
print(free_port)
EOF
)

export NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES)) # M nodes x N GPUs

echo "nnodes: ${NNODES}"

dataroot="/data/MedSAM_data/train_npy_full"
pretrained_checkpoint="work_dir/LiteMedSAM/medsam_lite.pth"

torchrun train_multi_gpus.py \
    -i ${dataroot} \
    -task_name MedSAM-Lite-Box \
    -pretrained_checkpoint ${pretrained_checkpoint} \
    -work_dir ./work_dir_ddp \
    -batch_size 26 \
    -num_workers 16 \
    -lr 0.0001 \
    -num_epochs 20 \
    --data_aug \
    -world_size ${WORLD_SIZE} \
    -node_rank ${NODE_RANK} \
    -use_amp \
    -init_method env:// \
    -resume work_dir_ddp

echo "END TIME: $(date)"
