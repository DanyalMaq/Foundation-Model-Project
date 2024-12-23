import requests
from PIL import Image

import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration 
from peft import PeftModel, PeftConfig

model_id = "Salesforce/blip2-opt-2.7b"
sft_lora_model_id = "sadmankiba/blip2-sft"

class Blip2:
    def __init__(self, sft_lora=False):
        if sft_lora:
            config = PeftConfig.from_pretrained(sft_lora_model_id)
            model = Blip2ForConditionalGeneration.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.float16)
            self.model = PeftModel.from_pretrained(model, sft_lora_model_id).to(0)
        else:
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True, 
            ).to(0)

        self.processor = AutoProcessor.from_pretrained(model_id)
        
        
    def __call__(self, image, question):
        # Each value in "content" has to be a list of dicts with types ("text", "image") 
        prompt = f"Question: {question} Answer:"

        inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)

        generated_ids = self.model.generate(**inputs, max_new_tokens=300)
        text_output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return text_output

    
def build_blip2(sft_lora):
    return Blip2(sft_lora)

def test_blip2(sft_lora):
    model = build_blip2(sft_lora)
    
    # from web
    # image_file = "http://images.cocodataset.org/train2017/000000039768.jpg"
    # image = Image.open(requests.get(image_file, stream=True).raw)
          
    image_file = "images/2519330533_597840098a_o.jpg"
    image = Image.open(image_file).convert('RGB')  # PIL already opens in RGB mode
    response = model(image, "What are inside this?")
    print(response) 

# Output
# BLIP2-Base
# Image: "images/2519330533_597840098a_o.jpg"
# Answer: Oysters
# 
# BLIP2-LoRA-SFT
# Image: "images/2519330533_597840098a_o.jpg"
# Answer: The oysters are inside the shell. The oysters are ...

if __name__ == '__main__':
    test_blip2(sft_lora=True)