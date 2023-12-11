export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
export exp_log_name=gpu_memory_logs

mkdir -vp $exp_log_name

# bash run_sft.sh 
# input: batchsize max_seq_length trust_remote_code model_path TIACC_TRAINING_CUDA_KERNEL


# # llama 13B
bash run_sft.sh 8 1024 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 0
bash run_sft.sh 16 1024 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 0
bash run_sft.sh 24 1024 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 0

bash run_sft.sh 4 2048 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 0
bash run_sft.sh 8 2048 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 0

bash run_sft.sh 2 4096 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 0
bash run_sft.sh 4 4096 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 0

bash run_sft.sh 1 8192 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 0
bash run_sft.sh 2 8192 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 0



bash run_sft.sh 8 1024 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 1
bash run_sft.sh 16 1024 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 1
bash run_sft.sh 24 1024 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 1

bash run_sft.sh 4 2048 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 1
bash run_sft.sh 8 2048 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 1

bash run_sft.sh 2 4096 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 1
bash run_sft.sh 4 4096 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 1

bash run_sft.sh 1 8192 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 1
bash run_sft.sh 2 8192 false /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 1


# # llama 7B
bash run_sft.sh 16 1024 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 0
bash run_sft.sh 32 1024 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 0
bash run_sft.sh 40 1024 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 0

bash run_sft.sh 8 2048 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 0
bash run_sft.sh 16 2048 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 0
bash run_sft.sh 24 2048 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 0

bash run_sft.sh 4 4096 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 0
bash run_sft.sh 8 4096 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 0
bash run_sft.sh 16 4096 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 0

bash run_sft.sh 1 8192 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 0
bash run_sft.sh 2 8192 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 0
bash run_sft.sh 4 8192 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 0
bash run_sft.sh 8 8192 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 0


bash run_sft.sh 16 1024 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 1
bash run_sft.sh 32 1024 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 1
bash run_sft.sh 40 1024 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 1

bash run_sft.sh 8 2048 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 1
bash run_sft.sh 16 2048 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 1
bash run_sft.sh 24 2048 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 1

bash run_sft.sh 4 4096 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 1
bash run_sft.sh 8 4096 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 1
bash run_sft.sh 16 4096 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 1

bash run_sft.sh 1 8192 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 1
bash run_sft.sh 2 8192 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 1
bash run_sft.sh 4 8192 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 1
bash run_sft.sh 8 8192 false /mnt/data/tilearn/pretrain_models/Llama-2-7b-hf 1







