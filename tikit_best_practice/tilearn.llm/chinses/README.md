# Usage

代码从 https://github.com/ymcui/Chinese-LLaMA-Alpaca 修改得到

## dataset
使用从 https://github.com/THUDM/ChatGLM-6B/tree/main/ptuning 中获取的ADGEN数据集

### 下载数据集
ADGEN 数据集任务为根据输入（content）生成一段广告词（summary）。

```json
{
    "content": "类型#上衣*版型#宽松*版型#显瘦*图案#线条*衣样式#衬衫*衣袖型#泡泡袖*衣款式#抽绳",
    "summary": "这件衬衫的款式非常的宽松，利落的线条可以很好的隐藏身材上的小缺点，穿在身上有着很好的显瘦效果。领口装饰了一个可爱的抽绳，漂亮的绳结展现出了十足的个性，配合时尚的泡泡袖型，尽显女性甜美可爱的气息。"
}
```

从 [Google Drive](https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view?usp=sharing) 或者 [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1) 下载处理好的 ADGEN 数据集，将解压后的 `AdvertiseGen` 目录放到/mnt/data/dataset/目录下。

```shell
tree /mnt/data/dataset/AdvertiseGen

/mnt/data/dataset/AdvertiseGen
├── dev.json
└── train.json

0 directories, 2 files
```

## env

在tione上使用请使用镜像环境： tione.tencentcloudcr.com/qcloud-ti-platform/llm-train:23.07-gpu-py310-cu121-deepspeed-tilearn-llm-v1.6.0

```shell
pip3 install sentencepiece jieba rouge_chinese nltk easydict pandas

# 安装最新的 tilearn 加速框架
pip3 install tilearn.ops -i https://g-bnvx3728-pypi.pkg.coding.net/tione/tilearn/simple
pip3 install tilearn.llm -i https://pypi.org/simple  
```


## train

```shell
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

bash run_sft.sh 2 4096 false /path/Llama-2-7b-hf 1 O3 false 8

```

## eval
```shell
# bash run_sft_eval.sh 
# input args: 
# run_sft_eval.sh
# output_model_dir_path
# output_model_name
# tokenize_path

bash run_sft_eval.sh ./output/sft_AquilaChat2-7B_zero2_4096_2_1 checkpoint-100 /mnt/data/tilearn/pretrain_models/AquilaChat2-7B
```