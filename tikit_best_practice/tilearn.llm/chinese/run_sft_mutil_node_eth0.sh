set -exo pipefail

echo 3 > /proc/sys/vm/drop_caches

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
#export NCCL_IB_DISABLE=1 
#export NCCL_IBEXT_DISABLE=1

# export GLOO_SOCKET_IFNAME=bond0
# export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
#export NCCL_IB_TC=128
export NCCL_IB_GID_INDEX=3

bash /usr/local/qcloud/rdma/set_bonding.sh && bash /usr/local/qcloud/rdma/dscp.sh

export per_device_train_batch_size=${1:-8}
export max_seq_length=${2:-1024}
export trust_remote_code=${3:-false}
export model_name_or_path=${4:-"/mnt/data/tilearn/pretrain_models/Baichuan-13B-Chat"}
export gradient_checkpointing=${7:-true}
export gradient_accumulation_steps=${8:-1}
export exp_name=${9:-""}

pretrained_model=$model_name_or_path
chinese_tokenizer_path=$model_name_or_path


IFS='/' read -ra arr <<< "$pretrained_model"
for i in "${arr[@]}"; do
    echo "Fruit: $i"
    model_name=$i
done

echo $gradient_checkpointing

if [[ $gradient_checkpointing == true ]]; then
    gradient_checkpointing_info=""
else
    gradient_checkpointing_info="_nocheckpoint"
fi

if [[ $gradient_accumulation_steps -gt 1 ]]; then
    gradient_accumulation_steps_info="_gradient_accumulation_steps-"${gradient_accumulation_steps}
else
    gradient_accumulation_steps_info=""
fi




lr=1e-4
lora_rank=8
lora_alpha=32
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

# pretrained_model=/mnt/data/pretrain_model/chinese-alpaca-plus-7b-merged/
# chinese_tokenizer_path=/mnt/data/pretrain_model/chinese-alpaca-plus-7b-merged/

# pretrained_model=/mnt/data/tilearn/pretrain_models/Baichuan-13B-Chat
# chinese_tokenizer_path=/mnt/data/tilearn/pretrain_models/Baichuan-13B-Chat

# pretrained_model=/mnt/data/lemonqin/bloomz-7b1-mt
# chinese_tokenizer_path=/mnt/data/lemonqin/bloomz-7b1-mt

# pretrained_model=/mnt/data/lemonqin/bloomz-560m
# chinese_tokenizer_path=/mnt/data/lemonqin/bloomz-560m

#dataset_dir=path/to/sft/data/dir
# per_device_train_batch_size=8
per_device_eval_batch_size=1
output_dir=./output/
peft_model=./output/peft/
train_file=/mnt/data/dataset/AdvertiseGen/train.json
validation_file=/mnt/data/dataset/AdvertiseGen/dev.json
# train_file=../../../dataset/tione/cnn-part-zh.json
# validation_file=../../../dataset/tione/cnn-part-zh.json

deepspeed_config_file=ds_zero2.json
#deepspeed_config_file=ds_zero3.json

#####################CONFIG
DO_TRAIN=1
DO_EVAL=0

### -------------------------------------------------------------------
### TIACC CUDA Kernel
### TIACC: TIACC_TRAINING_CUDA_KERNEL=1
### Deepspeed: TIACC_TRAINING_CUDA_KERNEL=0
# export TIACC_TRAINING_CUDA_KERNEL=1
export TIACC_TRAINING_CUDA_KERNEL=${5:-1}
#export TIACC_TRAINING_KERNEL=${USE_TIACC_TRAINING}

### TIACC STATIC ZERO
### Open: TIACC_TRAINING_CUDA_KERNEL='O2' 
### support 'O2' / 'O2.5' / 'O3' / 'O3.5' / 'O3_Q8'(doing)
### Close: TIACC_TRAINING_CUDA_KERNEL='None'
# export TIACC_TRAINING_STATIC_ZERO='O2' #'None'
export TIACC_TRAINING_STATIC_ZERO=${6:-"O2"}
if [ ${TIACC_TRAINING_STATIC_ZERO} = "O2" ]; then
    train_config="_zero2"
elif [ ${TIACC_TRAINING_STATIC_ZERO} = "O2.5" ]; then
    train_config="_zero2_offload"
elif [ ${TIACC_TRAINING_STATIC_ZERO} = "O3" ]; then
    train_config="_zero3"
elif [ ${TIACC_TRAINING_STATIC_ZERO} = "O3.5" ]; then
    train_config="_zero3_offload"
elif [ ${TIACC_TRAINING_STATIC_ZERO} = "O3.5_V100" ]; then
    train_config="_zero3_offload"
else 
    train_config=""
fi

# ### TIACC DYNAMIC ZERO
# ### TIACC: TIACC_TRAINING_DYNAMIC_ZERO=1 and set TIACC_ZERO_STAGE/TIACC_ZERO_STAGE/TIACC_PLACEMENT/TIACC_SHARD_INIT/TIACC_CPU_INIT
# ### Deepspeed: TIACC_TRAINING_DYNAMIC_ZERO=0
# export TIACC_TRAINING_DYNAMIC_ZERO=0
# export TIACC_ZERO_STAGE=3 #for TIACC_TRAINING_DYNAMIC_ZERO=1
# export TIACC_PLACEMENT='cpu' #'cuda' #for TIACC_TRAINING_DYNAMIC_ZERO=1
# export TIACC_SHARD_INIT=0 #for TIACC_TRAINING_DYNAMIC_ZERO=1
# export TIACC_CPU_INIT=1 #for TIACC_TRAINING_DYNAMIC_ZERO=1
# ### -------------------------------------------------------------------

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
# if [ ${TIACC_TRAINING_DYNAMIC_ZERO} = 0 ]; then
#   #USE_DS="--deepspeed=./ds_config_zero3.json"
#   USE_DS="--deepspeed=${deepspeed_config_file}"
# else
#   USE_DS=""
# fi
if [[ $TIACC_TRAINING_STATIC_ZERO == *"O2"* ]]; then
    USE_DS="--deepspeed=../accelerate/tests/deepspeed/ds_config_zero2.json"
    # USE_DS="--deepspeed=./ds_zero2.json"
elif [[ $TIACC_TRAINING_STATIC_ZERO == *"O3"* ]]; then
    USE_DS="--deepspeed=../accelerate/tests/deepspeed/ds_config_zero3.json"
else
    USE_DS=""
fi

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5, 

# export TIACC_TRAINING_CUDA_KERNEL_LEVEL=1
CVD=(${CUDA_VISIBLE_DEVICES//,/ })
NUM_GPUS=${#CVD[@]}
echo $NUM_GPUS

if [ $NUM_GPUS -eq 0 ];then
    NUM_GPUS=8
fi


max_steps=100

if [[ $save_model == true ]]; then
    save_model=100
else
    save_model=100000000000
fi




exp_all_name=sft_${model_name}${exp_name}${train_config}${gradient_checkpointing_info}_${max_seq_length}${gradient_accumulation_steps_info}_${per_device_train_batch_size}_${TIACC_TRAINING_CUDA_KERNEL}



torchrun --nnodes 2 --nproc_per_node $NUM_GPUS --master_addr $master_addr --master_port $master_port --node_rank=0 run_clm_sft_with_peft.py \
    ${USE_DS} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --train_file ${train_file} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    ${TRAIN_CONFIG} \
    ${EVAL_CONFIG} \
    --seed $RANDOM \
    --bf16 \
    --max_steps $max_steps \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps $save_model \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 32 \
    --max_seq_length $max_seq_length \
    --output_dir ${output_dir}/${exp_all_name} \
    --overwrite_output_dir \
    --ddp_timeout 300 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --validation_file ${validation_file} \
    --peft_path ${peft_model} \
    --gradient_checkpointing $gradient_checkpointing \
    --trust_remote_code ${trust_remote_code} \
    --ddp_find_unused_parameters False 2>&1 | tee ./${exp_log_name}/${exp_all_name}.log 

    #--do_eval \
    #--evaluation_strategy steps \
    #--eval_steps 250 \
    #--deepspeed ${deepspeed_config_file} \

