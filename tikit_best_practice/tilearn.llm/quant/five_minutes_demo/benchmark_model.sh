# AutoLayerwiseSmoothQ
python test_baichuan_performance.py -q smoothquant

# WeightOnlyInt8
python test_baichuan_performance.py -q ifq

# Original FP16
python test_baichuan_performance.py

# L20:
# AutoLayerwiseSmoothQ performance: batch_size: 16, prompt_size: 1000, first_token_latency:  2387.11 ms, generated_tokens: 100, throughput:  233.47 tokens/second
# WeightOnlyInt8 performance: batch_size: 16, prompt_size: 1000, first_token_latency:  4313.80 ms, generated_tokens: 100, throughput:  185.76 tokens/second
# Original FP16 performance: batch_size: 16, prompt_size: 1000, first_token_latency:  4042.74 ms, generated_tokens: 100, throughput:  112.71 tokens/second
