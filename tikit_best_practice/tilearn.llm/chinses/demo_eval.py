 
import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM

def generate_prompt(text):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{text}

### Response: """

#path = "/mnt/data/pretrain_model/chinese-alpaca-plus-7b-merged/"
path = "./output/checkpoint-100/"

tokenizer = LlamaTokenizer.from_pretrained(path)
model = LlamaForCausalLM.from_pretrained(path).half().to('cuda')
model.eval()

#text = '类型#上衣*风格#性感*衣样式#风衣*衣领型#翻领*衣门襟#双排扣*衣款式#露肩'
text = '类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞'
#text = '根据关键字生成广告：类型#上衣*风格#性感*衣样式#风衣*衣领型#翻领*衣门襟#双排扣*衣款式#露肩'
prompt = generate_prompt(text)
#prompt = text
input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')

with torch.no_grad():
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=128,
        temperature=1,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.15
    ).cuda()
output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"input_ids:{input_ids[0]}, output_ids:{output_ids[0]}, prompt:{prompt}, output:{output}")
print(output.replace(prompt, '').strip())
