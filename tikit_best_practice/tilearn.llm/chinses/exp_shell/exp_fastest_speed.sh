export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
export exp_log_name=logs_fastest_speed

mkdir -vp $exp_log_name

# bash run_sft.sh 
# input: batchsize max_seq_length trust_remote_code model_path TIACC_TRAINING_CUDA_KERNEL TIACC_TRAINING_STATIC_ZERO gradient_checkpointing exp_log_name


# # # baichuan2 13B
bash run_sft.sh 1 4096 true /mnt/data/tilearn/pretrain_models/Baichuan2-13B-Chat 0 O2
bash run_sft.sh 1 4096 true /mnt/data/tilearn/pretrain_models/Baichuan2-13B-Chat 0 O2

TILEARN_LLM_BAICHUAN_13B=2 bash run_sft.sh 1 4096 true /mnt/data/tilearn/pretrain_models/Baichuan2-13B-Chat 1

TILEARN_LLM_BAICHUAN_13B=2 bash run_sft.sh 1 4096 true /mnt/data/tilearn/pretrain_models/Baichuan2-13B-Chat 1
TILEARN_LLM_BAICHUAN_13B=2 bash run_sft.sh 8 4096 true /mnt/data/tilearn/pretrain_models/Baichuan2-13B-Chat 1


# # # baichuan2 7B
# bash run_sft.sh 2 4096 true /mnt/data/tilearn/pretrain_models/Baichuan2-7B-Chat 0
# bash run_sft_with_zero3_nocheckpoint.sh 2 4096 true /mnt/data/tilearn/pretrain_models/Baichuan2-7B-Chat 0

# bash run_sft_with_zero3_nocheckpoint.sh 2 4096 true /mnt/data/tilearn/pretrain_models/Baichuan2-7B-Chat 1

# bash gradient_checkpointing_enable.sh 2 4096 true /mnt/data/tilearn/pretrain_models/Baichuan2-7B-Chat 1
# bash gradient_checkpointing_enable.sh 16 4096 true /mnt/data/tilearn/pretrain_models/Baichuan2-7B-Chat 1


# # # baichuan1 13B
# bash run_sft.sh 1 4096 true /mnt/data/tilearn/pretrain_models/Baichuan-13B-Chat 0
# bash run_sft_with_zero3_nocheckpoint.sh 1 4096 true /mnt/data/tilearn/pretrain_models/Baichuan-13B-Chat 0

# bash run_sft_with_zero3_nocheckpoint.sh 1 4096 true /mnt/data/tilearn/pretrain_models/Baichuan-13B-Chat 1

# bash gradient_checkpointing_enable.sh 1 4096 true /mnt/data/tilearn/pretrain_models/Baichuan-13B-Chat 1
# bash gradient_checkpointing_enable.sh 8 4096 true /mnt/data/tilearn/pretrain_models/Baichuan-13B-Chat 1

# # # baichuan1 7B
# MODEL_PATH=/mnt/data/tilearn/pretrain_models/baichuan-7B

# bash run_sft.sh 2 4096 true $MODEL_PATH 0
# bash run_sft_with_zero3_nocheckpoint.sh 2 4096 true $MODEL_PATH 0

# bash run_sft_with_zero3_nocheckpoint.sh 2 4096 true $MODEL_PATH 1

# bash gradient_checkpointing_enable.sh 2 4096 true $MODEL_PATH 1
# bash gradient_checkpointing_enable.sh 16 4096 true $MODEL_PATH 1


# # # llama2 13B
# bash run_sft.sh 1 4096 true /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 0
# bash run_sft_with_zero3_nocheckpoint.sh 1 4096 true /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 0

# bash run_sft_with_zero3_nocheckpoint.sh 1 4096 true /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 1

# bash gradient_checkpointing_enable.sh 1 4096 true /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 1
# bash gradient_checkpointing_enable.sh 8 4096 true /mnt/data/tilearn/pretrain_models/Llama-2-13b-chat-hf 1


# # # llama2 7B
# MODEL_PATH=/mnt/data/tilearn/pretrain_models/Llama-2-7b-hf

# bash run_sft.sh 2 4096 true $MODEL_PATH 0
# bash run_sft_with_zero3_nocheckpoint.sh 2 4096 true $MODEL_PATH 0

# bash run_sft_with_zero3_nocheckpoint.sh 2 4096 true $MODEL_PATH 1

# bash gradient_checkpointing_enable.sh 2 4096 true $MODEL_PATH 1
# bash gradient_checkpointing_enable.sh 16 4096 true $MODEL_PATH 1



# # # bloooz 7B
# MODEL_PATH=/mnt/data/lemonqin/bloomz-7b1-mt

# bash run_sft.sh 2 4096 true $MODEL_PATH 0
# bash run_sft_with_zero3_nocheckpoint.sh 2 4096 true $MODEL_PATH 0

# bash run_sft_with_zero3_nocheckpoint.sh 2 4096 true $MODEL_PATH 1

# bash gradient_checkpointing_enable.sh 2 4096 true $MODEL_PATH 1
# bash gradient_checkpointing_enable.sh 16 4096 true $MODEL_PATH 1