export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
# export CUDA_VISIBLE_DEVICES=4,5,6,7
export exp_log_name=logs/AdvertiseGen.0.7.2.AquilaChat2

mkdir -vp $exp_log_name

# bash run_sft.sh 
# input args: 
# batchsize max_seq_length
# trust_remote_code
# model_path
# TIACC_TRAINING_CUDA_KERNEL
# TIACC_TRAINING_STATIC_ZERO
# gradient_checkpointing
# gradient_accumulation_steps
# exp_log_name

# aquila2 34B

#bash run_sft_mutil_node_eth0.sh 4 1024 true /mnt/data/tilearn/pretrain_models/AquilaChat2-34B 0 O2
#bash run_sft_mutil_node_eth0.sh 4 1024 true /mnt/data/tilearn/pretrain_models/AquilaChat2-34B 1 O2
#bash run_sft_mutil_node_eth0.sh 8 1024 true /mnt/data/tilearn/pretrain_models/AquilaChat2-34B 0 O2
#bash run_sft_mutil_node_eth0.sh 8 1024 true /mnt/data/tilearn/pretrain_models/AquilaChat2-34B 1 O2

#bash run_sft_mutil_node_eth0.sh 4 1024 true /mnt/data/tilearn/pretrain_models/AquilaChat2-34B 0 O3
#bash run_sft_mutil_node_eth0.sh 4 1024 true /mnt/data/tilearn/pretrain_models/AquilaChat2-34B 1 O3
#bash run_sft_mutil_node_eth0.sh 8 1024 true /mnt/data/tilearn/pretrain_models/AquilaChat2-34B 0 O3
#bash run_sft_mutil_node_eth0.sh 8 1024 true /mnt/data/tilearn/pretrain_models/AquilaChat2-34B 1 O3

#bash run_sft_mutil_node_eth0.sh 2 2048 true /mnt/data/tilearn/pretrain_models/AquilaChat2-34B 0 O2
#bash run_sft_mutil_node_eth0.sh 2 2048 true /mnt/data/tilearn/pretrain_models/AquilaChat2-34B 1 O2
#bash run_sft_mutil_node_eth0.sh 4 2048 true /mnt/data/tilearn/pretrain_models/AquilaChat2-34B 0 O2
#bash run_sft_mutil_node_eth0.sh 4 2048 true /mnt/data/tilearn/pretrain_models/AquilaChat2-34B 1 O2

#bash run_sft_mutil_node_eth0.sh 2 2048 true /mnt/data/tilearn/pretrain_models/AquilaChat2-34B 0 O3
#bash run_sft_mutil_node_eth0.sh 2 2048 true /mnt/data/tilearn/pretrain_models/AquilaChat2-34B 1 O3
#bash run_sft_mutil_node_eth0.sh 4 2048 true /mnt/data/tilearn/pretrain_models/AquilaChat2-34B 0 O3
#bash run_sft_mutil_node_eth0.sh 4 2048 true /mnt/data/tilearn/pretrain_models/AquilaChat2-34B 1 O3

bash run_sft_mutil_node_eth0.sh 1 4096 true /mnt/data/tilearn/pretrain_models/AquilaChat2-34B 0 O2
bash run_sft_mutil_node_eth0.sh 1 4096 true /mnt/data/tilearn/pretrain_models/AquilaChat2-34B 1 O2
bash run_sft_mutil_node_eth0.sh 2 4096 true /mnt/data/tilearn/pretrain_models/AquilaChat2-34B 0 O2
bash run_sft_mutil_node_eth0.sh 2 4096 true /mnt/data/tilearn/pretrain_models/AquilaChat2-34B 1 O2

bash run_sft_mutil_node_eth0.sh 1 4096 true /mnt/data/tilearn/pretrain_models/AquilaChat2-34B 0 O3
bash run_sft_mutil_node_eth0.sh 1 4096 true /mnt/data/tilearn/pretrain_models/AquilaChat2-34B 1 O3
bash run_sft_mutil_node_eth0.sh 2 4096 true /mnt/data/tilearn/pretrain_models/AquilaChat2-34B 0 O3
bash run_sft_mutil_node_eth0.sh 2 4096 true /mnt/data/tilearn/pretrain_models/AquilaChat2-34B 1 O3

bash run_sft_mutil_node_eth0.sh 4 4096 true /mnt/data/tilearn/pretrain_models/AquilaChat2-34B 1 O3


bash run_sft_mutil_node_eth0.sh 8 4096 true /mnt/data/tilearn/pretrain_models/AquilaChat2-7B 0 O3
bash run_sft_mutil_node_eth0.sh 8 4096 true /mnt/data/tilearn/pretrain_models/AquilaChat2-7B 1 O3

bash run_sft_mutil_node_eth0.sh 8 4096 true /mnt/data/tilearn/pretrain_models/AquilaChat2-7B 0 O2
bash run_sft_mutil_node_eth0.sh 8 4096 true /mnt/data/tilearn/pretrain_models/AquilaChat2-7B 1 O2

bash run_sft.sh 8 4096 true /mnt/data/tilearn/pretrain_models/AquilaChat2-7B 0 O3
bash run_sft.sh 8 4096 true /mnt/data/tilearn/pretrain_models/AquilaChat2-7B 1 O3

bash run_sft.sh 8 4096 true /mnt/data/tilearn/pretrain_models/AquilaChat2-7B 0 O2
bash run_sft.sh 8 4096 true /mnt/data/tilearn/pretrain_models/AquilaChat2-7B 1 O2

