from vllm import LLM, SamplingParams
import torch, time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model",
                    type=str,
                    default="./LayerwiseSearchSMQ_Baichuan2-13B-Base",
                    help='Path to load the model.')
parser.add_argument("--num-prompts",
                    type=int,
                    default=16,
                    help="Number of prompts to process.")
parser.add_argument("--input-len",
                    type=int,
                    default=1000,
                    help="Input prompt length for each request")
parser.add_argument("--output-len",
                    type=int,
                    default=100,
                    help="Output length for each request. Overrides the "
                    "output length from the dataset.")
parser.add_argument('--quantization',
                    '-q',
                    choices=['ifq', 'smoothquant', None],
                    default=None)
parser.add_argument("--tensor-parallel-size", "-tp",
                    type=int,
                    default=1)

args = parser.parse_args()


def perf(model, input_ids):
    prompt_size = len(input_ids[0])

    sampling_params = SamplingParams(top_k=1, max_tokens=args.output_len)
    ic = 1
    start = time.time()
    outputs = model.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)
    end = time.time()

    sampling_params = SamplingParams(top_k=1, max_tokens=1)
    start = time.time()
    ic = 2
    for _ in range(ic):
        outputs = model.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)
    end = time.time()
    num_tokens = len(outputs[0].outputs[0].token_ids)
    assert num_tokens == 1
    performance = f"batch_size: {len(input_ids)}, prompt_size: {prompt_size}, first_token_latency: {(end - start) * 1000.0 / ic : 8.2f} ms"

    sampling_params = SamplingParams(top_k=1, max_tokens=args.output_len)
    start = time.time()
    ic = 2
    for _ in range(ic):
        outputs = model.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)
    end = time.time()
    num_tokens = len(outputs[0].outputs[0].token_ids)
    performance += f", generated_tokens: {num_tokens}, throughput: {num_tokens * len(input_ids) / ((end - start) / ic) : 5.2f} tokens/second"
    return performance


if __name__ == '__main__':
    model = LLM(model=args.model,
                trust_remote_code=True,
                tensor_parallel_size=args.tensor_parallel_size,
                quantization=args.quantization,
                dtype=torch.float16)

    input_ids = []
    for _ in range(args.num_prompts):
        torch.manual_seed(1)
        input_ids.append(torch.randint(low=1, high=10, size=(args.input_len,)).to(torch.int64).tolist())

    if args.quantization == "smoothquant":
        prefix = "AutoLayerwiseSmoothQ"
    elif args.quantization == "ifq":
        prefix = "WeightOnlyInt8"
    elif args.quantization == None:
        prefix = "Original FP16"
    else:
        print(f"Not supported quantization type: {args.quantization}")

    performance = perf(model, input_ids)
    print(f"{prefix} performance: {performance}")
