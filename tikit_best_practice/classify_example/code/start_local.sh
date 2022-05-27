num_nodes=1
INDEX=0
num_gpus=1
maddr=127.0.0.1

export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./model" ];then
    mkdir model
fi

if [ ! -d "~/.cache/torch/hub/checkpoints" ];then
    mkdir -p ~/.cache/torch/hub/checkpoints
fi
cp -f resnet50-0676ba61.pth ~/.cache/torch/hub/checkpoints

python3 -m torch.distributed.launch --nnode=${num_nodes} --node_rank=${INDEX} --nproc_per_node=${num_gpus} --master_addr=${maddr} main.py \
        ../data -a resnet50 --batch-size 128 --lr 0.01 --pretrained --local_world_size=${num_gpus} --model_save_path=./model/
