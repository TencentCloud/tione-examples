#!/usr/bin/env bash
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

GPUS_PER_NODE=${GPU_NUM_PER_NODE}
TIACC_MODE=0
MODEL=resnet50
MASTER_ADDR=${CHIEF_IP}
MASTER_PORT=23499
NNODES=${NODE_NUM}
NODE_RANK=${INDEX}

help_func() {
    echo "Usage:"
    echo "sh start_cloud.sh  [-t,--tiacc, 0/1]          open/close tiacc acceleration engine"
    echo "                   [-m,--model name]          model name - resnet, resnest"
    echo "                   [-h,--help]                print usage"
    echo ""
}
die() { echo "$*" >&2; exit -1; }  # complain to STDERR and exit with error
needs_arg() { if [ -z "$OPTARG" ]; then die "No arg for --$OPT option"; fi; }

help_func
OPT_STR="t:m:h-"
while getopts $OPT_STR: OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    t | tiacc )                 needs_arg; TIACC_MODE="$OPTARG" ;;
    m | model )                 needs_arg; MODEL="$OPTARG" ;;
    h | help )     die "";;
    ??* )          die "Illegal long option --$OPT" ;;  # bad long option
    ? )            exit 2 ;;  # bad short option (error reported via getopts)
  esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
CONFIG_DEFAULT="/opt/ml/input/data/image_classify/ --lr 0.01 --print-freq 5 --pretrained --local_world_size=$GPUS_PER_NODE --model_save_path=/opt/ml/model/"

if [ "$MODEL" = "resnet50" ] && [ "$TIACC_MODE" -eq "0" ] ;then
    LAUNCH=torch.distributed.launch
    CONFIG=$CONFIG_DEFAULT" -a resnet50 --batch-size 128"
elif [ "$MODEL" = "resnet50" ] && [ "$TIACC_MODE" -eq "1" ] ;then
    LAUNCH=tiacc_training.distributed.launch
    CONFIG=$CONFIG_DEFAULT" -a resnet50 --batch-size 128 --tiacc"
elif [ "$MODEL" = "resnest50" ] && [ "$TIACC_MODE" -eq "0" ] ;then
    LAUNCH=torch.distributed.launch
    CONFIG=$CONFIG_DEFAULT" -a resnest50 --batch-size 128"
elif [ "$MODEL" = "resnest50" ] && [ "$TIACC_MODE" -eq "1" ] ;then
    LAUNCH=tiacc_training.distributed.launch
    CONFIG=$CONFIG_DEFAULT" -a resnest50 --batch-size 128 --tiacc"
else
    echo "input TIACC_MODE is wrong, the range is [0-1], but $TIACC_MODE"
    echo "input MODEL is wrong, the range is [resnet50, resnest50], but $MODEL"
fi

echo "TIACC_MODE: $TIACC_MODE"
echo "MODEL: $MODEL"
echo "DISTRIBUTED_ARGS: $DISTRIBUTED_ARGS"
echo "LAUNCH: $LAUNCH"
echo "CONFIG: $CONFIG"

if [ ! -d "~/.cache/torch/hub/checkpoints" ];then
    mkdir -p ~/.cache/torch/hub/checkpoints
fi
cp -f resnet50-0676ba61.pth ~/.cache/torch/hub/checkpoints

python3 -u -m $LAUNCH $DISTRIBUTED_ARGS main.py $CONFIG

