import torch
from tilearn.llm.quant import AutoLayerwiseSmoothQForCausalLM, LayerwiseSmoothQuantizeConfig
from transformers import AutoConfig, AutoTokenizer
import argparse
from process_data import get_loaders

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default='/mnt/data/tilearn_demo/Baichuan2-13B-Base',help='HF model')    # Huggingface model name
parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
parser.add_argument('--save_path', type=str, default=None, help='Path to save the quant model.')
parser.add_argument('--calib_data', type=str, default=None, help='Path to load the calibration data.')
args = parser.parse_args()


##step1: searching
model = AutoLayerwiseSmoothQForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto', torch_dtype=torch.float16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
print(model.hf_device_map)
print("loading calibdation data")
dataloader, _ = get_loaders(args.calib_data, nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
print("dataset loading complete")
quant_config = LayerwiseSmoothQuantizeConfig(bits=8,
                                    smoothq_quant_output_method='per_channel',
                                    act_quant='per_token',
                                    weight_quant='per_channel')
# support examples type: List[Dict[str, torch.Tensor]]
model.search(examples=dataloader, quant_config=quant_config, save_dir=args.save_path, alpha_choice=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
del model
torch.cuda.empty_cache()