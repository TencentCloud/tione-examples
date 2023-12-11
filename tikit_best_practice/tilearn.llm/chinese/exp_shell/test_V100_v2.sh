export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
# export CUDA_VISIBLE_DEVICES=4,5,6,7
export exp_log_name=logs/V0.6.4.AdvertiseGen.V100.finish

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


# # O3
bash run_sft.sh 1 1024 true /mnt/data/tilearn/pretrain_models/AquilaChat2-7B 1 O3
bash run_sft.sh 1 1024 true /mnt/data/tilearn/pretrain_models/Baichuan2-7B-Chat 1 O3
bash run_sft.sh 1 1024 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-chat-hf 1 O3
bash run_sft.sh 1 1024 false /mnt/data/tilearn/pretrain_models/bloomz-7b1-mt 1 O3
bash run_sft.sh 1 1024  true /mnt/data/tilearn/pretrain_models/baichuan-7B 1 O3


# O3.5
bash run_sft.sh 1 1024 true /mnt/data/tilearn/pretrain_models/AquilaChat2-7B 1 O3.5
bash run_sft.sh 1 1024 true /mnt/data/tilearn/pretrain_models/Baichuan2-7B-Chat 1 O3.5
bash run_sft.sh 1 1024 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-chat-hf 1 O3.5
bash run_sft.sh 1 1024 false /mnt/data/tilearn/pretrain_models/bloomz-7b1-mt 1 O3.5
bash run_sft.sh 1 1024  true /mnt/data/tilearn/pretrain_models/baichuan-7B 1 O3.5

TILEARN_LLM_BAICHUAN_13B=2 bash run_sft.sh 1 512 true /mnt/data/tilearn/pretrain_models/Baichuan2-13B-Chat 1 O3.5_V100
bash run_sft.sh 1 512 true /mnt/data/tilearn/pretrain_models/Baichuan-13B-Base 1 O3.5_V100
bash run_sft.sh 1 512  false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 1 O3.5_V100

# ds
export tilearn_deepspeed_patch=0

# # O3
bash run_sft.sh 1 1024 true /mnt/data/tilearn/pretrain_models/AquilaChat2-7B 0 O3
bash run_sft.sh 1 1024 true /mnt/data/tilearn/pretrain_models/Baichuan2-7B-Chat 0 O3
bash run_sft.sh 1 1024 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-chat-hf 0 O3
bash run_sft.sh 1 1024 false /mnt/data/tilearn/pretrain_models/bloomz-7b1-mt 0 O3
bash run_sft.sh 1 1024  true /mnt/data/tilearn/pretrain_models/baichuan-7B 0 O3

# O3.5
bash run_sft.sh 1 1024 true /mnt/data/tilearn/pretrain_models/AquilaChat2-7B 0 O3.5
bash run_sft.sh 1 1024 true /mnt/data/tilearn/pretrain_models/Baichuan2-7B-Chat 0 O3.5
bash run_sft.sh 1 1024 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-chat-hf 0 O3.5
bash run_sft.sh 1 1024 false /mnt/data/tilearn/pretrain_models/bloomz-7b1-mt 0 O3.5
bash run_sft.sh 1 1024  true /mnt/data/tilearn/pretrain_models/baichuan-7B 0 O3.5

TILEARN_LLM_BAICHUAN_13B=2 bash run_sft.sh 1 512 true /mnt/data/tilearn/pretrain_models/Baichuan2-13B-Chat 0 O3.5_V100
bash run_sft.sh 1 512 true /mnt/data/tilearn/pretrain_models/Baichuan-13B-Base 0 O3.5_V100
bash run_sft.sh 1 512  false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 0 O3.5_V100







