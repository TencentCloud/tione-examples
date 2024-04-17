# step1: searching
python calc_module_scales.py --model_name_or_path ../five_minutes_demo/Baichuan2-13B-Base/ \
    --save_path ./output/ \
    --calib_data ../five_minutes_demo/dataset/wikitext

# step2: quantizing
python quant_module.py --model_name_or_path ../five_minutes_demo/Baichuan2-13B-Base/ \
    --save_path ./output/ \
    --test_data ../five_minutes_demo/dataset/wikitext