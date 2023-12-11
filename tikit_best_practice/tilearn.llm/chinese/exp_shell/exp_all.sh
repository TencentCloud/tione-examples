export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
# export CUDA_VISIBLE_DEVICES=4,5,6,7
export exp_log_name=logs.AdvertiseGen.0.7.1

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


# aquila2 7B
bash run_sft.sh 16 1024 true /mnt/data/tilearn/pretrain_models/AquilaChat2-7B 0
bash run_sft.sh 8 2048 true /mnt/data/tilearn/pretrain_models/AquilaChat2-7B 0
bash run_sft.sh 2 4096 true /mnt/data/tilearn/pretrain_models/AquilaChat2-7B 0
bash run_sft.sh 2 8192 true /mnt/data/tilearn/pretrain_models/AquilaChat2-7B 0
bash run_sft.sh 1 8192 true /mnt/data/tilearn/pretrain_models/AquilaChat2-7B 0

bash run_sft.sh 16 1024 true /mnt/data/tilearn/pretrain_models/AquilaChat2-7B 1
bash run_sft.sh 8 2048 true /mnt/data/tilearn/pretrain_models/AquilaChat2-7B 1
bash run_sft.sh 2 4096 true /mnt/data/tilearn/pretrain_models/AquilaChat2-7B 1
bash run_sft.sh 2 8192 true /mnt/data/tilearn/pretrain_models/AquilaChat2-7B 1
bash run_sft.sh 1 8192 true /mnt/data/tilearn/pretrain_models/AquilaChat2-7B 1


# baichuan2 13B
bash run_sft.sh 8 1024 true /mnt/data/tilearn/pretrain_models/Baichuan2-13B-Chat 0
bash run_sft.sh 4 2048 true /mnt/data/tilearn/pretrain_models/Baichuan2-13B-Chat 0
bash run_sft.sh 2 4096 true /mnt/data/tilearn/pretrain_models/Baichuan2-13B-Chat 0
bash run_sft.sh 1 8192 true /mnt/data/tilearn/pretrain_models/Baichuan2-13B-Chat 0

TILEARN_LLM_BAICHUAN_13B=2 bash run_sft.sh 8 1024 true /mnt/data/tilearn/pretrain_models/Baichuan2-13B-Chat 1
TILEARN_LLM_BAICHUAN_13B=2 bash run_sft.sh 4 2048 true /mnt/data/tilearn/pretrain_models/Baichuan2-13B-Chat 1
TILEARN_LLM_BAICHUAN_13B=2 bash run_sft.sh 2 4096 true /mnt/data/tilearn/pretrain_models/Baichuan2-13B-Chat 1
TILEARN_LLM_BAICHUAN_13B=2 bash run_sft.sh 1 8192 true /mnt/data/tilearn/pretrain_models/Baichuan2-13B-Chat 1


# baichuan2 7B
bash run_sft.sh 16 1024 true /mnt/data/tilearn/pretrain_models/Baichuan2-7B-Chat 0
bash run_sft.sh 8 2048 true /mnt/data/tilearn/pretrain_models/Baichuan2-7B-Chat 0
bash run_sft.sh 4 4096 true /mnt/data/tilearn/pretrain_models/Baichuan2-7B-Chat 0
bash run_sft.sh 2 8192 true /mnt/data/tilearn/pretrain_models/Baichuan2-7B-Chat 0
bash run_sft.sh 1 8192 true /mnt/data/tilearn/pretrain_models/Baichuan2-7B-Chat 0

bash run_sft.sh 16 1024 true /mnt/data/tilearn/pretrain_models/Baichuan2-7B-Chat 1
bash run_sft.sh 8 2048 true /mnt/data/tilearn/pretrain_models/Baichuan2-7B-Chat 1
bash run_sft.sh 4 4096 true /mnt/data/tilearn/pretrain_models/Baichuan2-7B-Chat 1
bash run_sft.sh 2 8192 true /mnt/data/tilearn/pretrain_models/Baichuan2-7B-Chat 1
bash run_sft.sh 1 8192 true /mnt/data/tilearn/pretrain_models/Baichuan2-7B-Chat 1


# baichuan1 13B
bash run_sft.sh 8 1024 true /mnt/data/tilearn/pretrain_models/Baichuan-13B-Chat 0
bash run_sft.sh 4 2048 true /mnt/data/tilearn/pretrain_models/Baichuan-13B-Chat 0
bash run_sft.sh 2 4096 true /mnt/data/tilearn/pretrain_models/Baichuan-13B-Chat 0
bash run_sft.sh 1 8192 true /mnt/data/tilearn/pretrain_models/Baichuan-13B-Chat 0

bash run_sft.sh 8 1024 true /mnt/data/tilearn/pretrain_models/Baichuan-13B-Chat 1
bash run_sft.sh 4 2048 true /mnt/data/tilearn/pretrain_models/Baichuan-13B-Chat 1
bash run_sft.sh 2 4096 true /mnt/data/tilearn/pretrain_models/Baichuan-13B-Chat 1
bash run_sft.sh 1 8192 true /mnt/data/tilearn/pretrain_models/Baichuan-13B-Chat 1


# baichuan1 7B
bash run_sft.sh 16 1024 true /mnt/data/tilearn/pretrain_models/baichuan-7B 0
bash run_sft.sh 8 2048 true /mnt/data/tilearn/pretrain_models/baichuan-7B 0
bash run_sft.sh 4 4096 true /mnt/data/tilearn/pretrain_models/baichuan-7B 0
bash run_sft.sh 2 8192 true /mnt/data/tilearn/pretrain_models/baichuan-7B 0
bash run_sft.sh 1 8192 true //mnt/data/tilearn/pretrain_models/baichuan-7B 0

bash run_sft.sh 16 1024 true /mnt/data/tilearn/pretrain_models/baichuan-7B 1
bash run_sft.sh 8 2048 true /mnt/data/tilearn/pretrain_models/baichuan-7B 1
bash run_sft.sh 4 4096 true /mnt/data/tilearn/pretrain_models/baichuan-7B 1
bash run_sft.sh 2 8192 true /mnt/data/tilearn/pretrain_models/baichuan-7B 1
bash run_sft.sh 1 8192 true /mnt/data/tilearn/pretrain_models/baichuan-7B 1


# llama2 13B
bash run_sft.sh 8 1024 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 0
bash run_sft.sh 4 2048 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 0
bash run_sft.sh 2 4096 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 0
bash run_sft.sh 1 8192 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 0

bash run_sft.sh 8 1024 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 1
bash run_sft.sh 4 2048 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 1
bash run_sft.sh 2 4096 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 1
bash run_sft.sh 1 8192 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 1


# llama2 7B
bash run_sft.sh 16 1024 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 0
bash run_sft.sh 8 2048 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 0
bash run_sft.sh 4 4096 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 0
bash run_sft.sh 2 8192 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 0
bash run_sft.sh 1 8192 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 0

bash run_sft.sh 16 1024 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 1
bash run_sft.sh 8 2048 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 1
bash run_sft.sh 4 4096 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 1
bash run_sft.sh 2 8192 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 1
bash run_sft.sh 1 8192 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 1


# bloooz 7B
bash run_sft.sh 16 1024 false /mnt/data/tilearn/pretrain_models/bloomz-7b1-mt 0
bash run_sft.sh 8 2048 false /mnt/data/tilearn/pretrain_models/bloomz-7b1-mt 0
bash run_sft.sh 4 4096 false /mnt/data/tilearn/pretrain_models/bloomz-7b1-mt 0
bash run_sft.sh 2 8192 false /mnt/data/tilearn/pretrain_models/bloomz-7b1-mt 0
bash run_sft.sh 1 8192 false /mnt/data/tilearn/pretrain_models/bloomz-7b1-mt 0

bash run_sft.sh 16 1024 false /mnt/data/tilearn/pretrain_models/bloomz-7b1-mt 1
bash run_sft.sh 8 2048 false /mnt/data/tilearn/pretrain_models/bloomz-7b1-mt 1
bash run_sft.sh 4 4096 false /mnt/data/tilearn/pretrain_models/bloomz-7b1-mt 1
bash run_sft.sh 2 8192 false /mnt/data/tilearn/pretrain_models/bloomz-7b1-mt 1
bash run_sft.sh 1 8192 false /mnt/data/tilearn/pretrain_models/bloomz-7b1-mt 1


