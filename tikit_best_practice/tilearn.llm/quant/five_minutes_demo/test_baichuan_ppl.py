import torch
from tilearn.llm.quant import AutoLayerwiseSmoothQForCausalLM, LayerwiseSmoothQuantizeConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import argparse
from process_data import eval_ppl, get_loaders


parser = argparse.ArgumentParser()
## TODO need to change cfs path
parser.add_argument('--model_name_or_path', type=str, default='./Baichuan2-13B-Base',help='HF model')    # Huggingface model name
parser.add_argument('--dataset', type=str, default='./dataset/wikitext', help='Path to load the dataset.')
parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
parser.add_argument('--save_path', type=str, default='./LayerwiseSearchSMQ_Baichuan2-13B-Base/', help='Path to save the quant model.')

args = parser.parse_args()


###Tilearn quantizing
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
model = AutoLayerwiseSmoothQForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True)
model.quantize(save_dir=args.save_path, save_for_inference_param_flag=False)
model.half().cuda()
print('AutoLayerwiseSmoothQ for baichuan2-13b PPL is: {}'.format(eval_ppl(model, tokenizer, args.dataset, args.seed)))
del model
torch.cuda.empty_cache()

# fp16
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True)
model.seqlen = 4096
model.half().cuda()
print('Original FP16 for baichuan2-13b PPL is: {}'.format(eval_ppl(model, tokenizer, args.dataset, args.seed)))
del model
torch.cuda.empty_cache()

# weight-only
from tilearn.llm.quant import AutoMinMaxQForCausalLM, MinMaxQuantizeConfig
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoMinMaxQForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True)
quant_config = MinMaxQuantizeConfig(bits=8)
model.quantize(quant_config=quant_config, need_to_pack=False)
model.half().cuda()
print('WeightOnlyInt8 for baichuan2-13b PPL is: {}'.format(eval_ppl(model, tokenizer, args.dataset, args.seed)))
del model
torch.cuda.empty_cache()

# smoothquant
from tilearn.llm.quant import AutoSmoothQForCausalLM, SmoothQuantizeConfig
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoSmoothQForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True)
model.seqlen = 4096
print("loading calibdation data")
dataloader, _ = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
print("dataset loading complete")
quant_config = SmoothQuantizeConfig(bits=8,
                                    smoothq_quant_output_method='per_channel',
                                    act_quant='per_token',
                                    weight_quant='per_channel',
                                    smooth_alpha=0.5,
                                    for_fake=True)
model.quantize(dataloader, quant_config=quant_config, cache_examples_on_gpu=True)
model.half().cuda()
print('SmoothQuant for baichuan2-13b PPL is: {}'.format(eval_ppl(model, tokenizer, args.dataset, args.seed)))
del model
torch.cuda.empty_cache()
