# step1: searching
python calc_module_scales.py --model_name_or_path /mnt/data/tilearn/pretrain_models/Baichuan2-13B-Base/ \
    --save_path /mnt/data/tilearn/pretrain_models/debug/ \
    --calib_data /mnt/data/huecheng/datasets/wikitext

# step2: quantizing
python quant_module.py --model_name_or_path /mnt/data/tilearn/pretrain_models/Baichuan2-13B-Base/ \
    --save_path /mnt/data/tilearn/pretrain_models/debug/ \
    --test_data /mnt/data/huecheng/datasets/wikitext