import torch
from tilearn.llm.quant import AutoLayerwiseSmoothQForCausalLM, LayerwiseSmoothQuantizeConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import argparse
from process_data import eval_ppl, get_loaders


parser = argparse.ArgumentParser()
## TODO need to change cfs path
parser.add_argument('--model_name_or_path', type=str, default='/mnt/data/tilearn_demo/Baichuan2-13B-Base',help='HF model')    # Huggingface model name
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
