echo NODE_LIST ${NODE_LIST}
split_list=(${NODE_LIST//,/ })
echo split_list ${split_list[@]}

cur_ipgpus=${split_list[${INDEX}]}
echo cur_ipgpus ${cur_ipgpus}

cur_ipgpus_list=(${cur_ipgpus//:/ })
echo cur_ipgpus_list ${cur_ipgpus_list[@]}

num_gpus=${cur_ipgpus_list[1]}
echo num_gpus ${num_gpus}

num_nodes=${#split_list[@]}
echo num_nodes ${num_nodes}

PORT=${PORT:-29501}
echo PORT ${PORT}

mipgpus=${split_list[0]}
echo mipgpus ${mipgpus}

mipgpus_list=(${mipgpus//:/ })
echo mipgpus_list ${mipgpus_list[@]}

maddr=${mipgpus_list[0]}
echo maddr ${maddr}

echo index ${INDEX}

sleep 5 # wait for dns establish

export NCCL_IB_DISABLE=1
NCCL_DEBUG=INFO NCCL_IB_GID_INDEX=3 NCCL_IB_SL=3 NCCL_IB_HCA=mlx5_2:1 NCCL_NET_GDR_READ=1

python3 -m torch.distributed.launch --nnode=${num_nodes} --node_rank=${INDEX} --nproc_per_node=${num_gpus} --master_addr=${maddr} main.py \
        /opt/ml/input/data/ -a resnext50_32x4d --batch-size 128 --local_world_size=${num_gpus}

#TODO test
#sleep 99999m
