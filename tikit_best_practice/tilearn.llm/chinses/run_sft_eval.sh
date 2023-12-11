set -exo pipefail

echo 3 > /proc/sys/vm/drop_caches


lr=1e-4
lora_rank=8
lora_alpha=32
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

#pretrained_model=/mnt/data/pretrain_model/chinese-alpaca-plus-7b-merged/
#chinese_tokenizer_path=/mnt/data/pretrain_model/chinese-alpaca-plus-7b-merged/
pretrained_model=./output/checkpoint-200/
chinese_tokenizer_path=./output/checkpoint-200/


export exp_name_all=${1:-""}
export checkpoint_step=${2:-"checkpoint-100"}
export chinese_tokenizer_path=${3:-""}

pretrained_model=${exp_name_all}/${checkpoint_step}/
chinese_tokenizer_path=$chinese_tokenizer_path


#dataset_dir=path/to/sft/data/dir
per_device_train_batch_size=8
per_device_eval_batch_size=1
training_steps=100
gradient_accumulation_steps=1
output_dir=./output/
peft_model=./output/peft/
train_file=/mnt/data/dataset/AdvertiseGen/train.json
validation_file=/mnt/data/dataset/AdvertiseGen/dev.json

deepspeed_config_file=ds_zero2_no_offload.json
#deepspeed_config_file=ds_zero3.json

### TIACC: USE_TIACC_TRAINING=1
### Deepspeed: USE_TIACC_TRAINING=0
USE_TIACC_TRAINING=0
#export TIACC_TRAINING_KERNEL=${USE_TIACC_TRAINING}
export TIACC_TRAINING_CUDA_KERNEL=${USE_TIACC_TRAINING}

#####################CONFIG
DO_TRAIN=0
DO_EVAL=1

export MAX_EVAL_SEQ_LENGTH=128
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

if [ ${DO_TRAIN} = 1 ]; then
    TRAIN_CONFIG="--do_train"
else
    TRAIN_CONFIG=""
fi
if [ ${DO_EVAL} = 1 ]; then
    EVAL_CONFIG="--do_eval --evaluation_strategy steps --eval_steps 200"
else
    EVAL_CONFIG=""
fi

torchrun --nnodes 1 --nproc_per_node 8 run_clm_sft_with_peft.py \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --train_file ${train_file} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    ${TRAIN_CONFIG} \
    ${EVAL_CONFIG} \
    --seed $RANDOM \
    --fp16 \
    --max_steps ${training_steps} \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 100 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 32 \
    --max_seq_length 1024 \
    --output_dir ${exp_name_all} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --validation_file ${validation_file} \
    --peft_path ${peft_model} \
    --gradient_checkpointing \
    --trust_remote_code true \
    --ddp_find_unused_parameters False 2>&1 | tee ${exp_name_all}/sft_eval.log 

    #--do_eval \
    #--evaluation_strategy steps \
    #--eval_steps 250 \
    #--deepspeed ${deepspeed_config_file} \
