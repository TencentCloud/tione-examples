num_nodes=${NODE_NUM}
node_index=${INDEX}
num_gpus=${GPU_NUM_PER_NODE}
maddr=${CHIEF_IP}

echo TIACC demo
echo num_nodes ${num_nodes}
echo node_index ${node_index}
echo num_gpus ${num_gpus}
echo maddr ${maddr}

sleep 5 # wait for dns establish

#export NCCL_SOCKET_IFNAME=eth0
#export CUDA_VISIBLE_DEVICES=0
export NCCL_IB_DISABLE=1
NCCL_DEBUG=INFO NCCL_IB_GID_INDEX=3 NCCL_IB_SL=3 NCCL_IB_HCA=mlx5_2:1 NCCL_NET_GDR_READ=1

if [ ! -d "~/.cache/torch/hub/checkpoints" ];then
    mkdir -p ~/.cache/torch/hub/checkpoints
fi
cp -f resnet50-0676ba61.pth ~/.cache/torch/hub/checkpoints

python3 -m torch.distributed.launch --nnode=${num_nodes} --node_rank=${node_index} --nproc_per_node=${num_gpus} --master_addr=${maddr} --master_port=23457 main.py \
        /opt/ml/input/data/image_classify -a resnet50 --batch-size 128 --lr 0.01 --pretrained --local_world_size=${num_gpus}
