import torch
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import PeftModel, PeftConfig

# Procedure
# 1. Load POPE dataset from HF and truncate. Load images.
# 2. Load BLIP2 model and processor
# 3. Run inference on POPE. Use model.generate(). 
# 4. Get metrics accuracy, precision, recall, f1 

from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re

dataset = load_dataset("lmms-lab/POPE")
print(dataset)

# Sample POPE
# {'id': '0', 'question_id': '1', 'question': 'Is there a snowboard in the image?', 
# 'answer': 'yes', 'image_source': 'COCO_val2014_000000310196', 'category': 'adversarial',
# 'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x427 at 0x7F29402C7250>}

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

def get_blip2_model(path: str, lora: bool = False):
    if lora: 
        config = PeftConfig.from_pretrained(path)
        model = Blip2ForConditionalGeneration.from_pretrained(config.base_model_name_or_path)
        lora_model = PeftModel.from_pretrained(model, path).to(device)
        return lora_model
            
    model = Blip2ForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.float16
        )
    model = model.to(device)
    
    return model

def run_eval():
    responses = []
    # model, name = get_blip2_model("Salesforce/blip2-opt-2.7b", lora=False), "blip2-base"
    # model, name = get_blip2_model("blip2-sft", lora=False), "blip2-sft"
    model, name = get_blip2_model("sadmankiba/blip2-sft", lora=True), "blip2-sft"
    
    for i, entry in enumerate(tqdm(dataset['test'])): 
        entry['question'] = entry['question'].replace('in the imange', 'in the image') # Fix typo 
        match = re.search(r'Is there (a|an) (.+?) in the image\?', entry['question'])
        if match:
            object_in_question = match.group(2)
        
        ques_format = 'mention_yes_no' # 'orig' for blip2-base, blip2-dpo, 'mention_yes_no' for blip2-sft
        if ques_format == 'orig':
            question = entry['question']
        elif ques_format == 'use_find':
            question = entry['question'].replace('Is there', 'Do you see')
            question = question.replace('in the image', '')
        elif ques_format == 'mention_yes_no':
            # question = entry['question'] + " Say yes or no."
            # question = entry['question'] + " Say no always."
            # question = 'What objects do you see in image? ' # + entry['question']
            # question = f'Any {object_in_question} here? Say yes if you see it, no otherwise.'
            # question = f'I don\'t see any {object_in_question} here. Do you see any?'
            # question = f'There is no {object_in_question} here. True or False?'
            # question = f'Is any {object_in_question} present in the image?'
            question = f'Which is true? There is no {object_in_question} in the image. Or, yes, there is a {object_in_question} in the image.'
        else: 
            print("Invalid question format")
            break
        prompt = f"Question: {question} Answer:"
        image = entry['image']
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

        output = model.generate(**inputs, max_new_tokens=20)
        
        decoded_output = processor.batch_decode(output, skip_special_tokens=True)[0].strip()
        decoded_response = decoded_output.split('Answer:')[1].strip()

        yes_no_response = None
        if ('yes' in decoded_response.lower() 
            or f'there is a {object_in_question}' in decoded_response.lower()
            or f'there is an {object_in_question}' in decoded_response.lower()):
            yes_no_response = 'yes'
        elif 'no' in decoded_response.lower():
            yes_no_response = 'no'
        else:
            yes_no_response = 'unknown'
        
        responses.append({
            'id': entry['id'],
            'question': question,
            'answer': entry['answer'],
            'image_source': entry['image_source'],
            'category': entry['category'],
            'response': yes_no_response
        })
        
        # Save every 1000 responses
        if (i + 1) % 1000 == 0:
            print("Saving responses...")
            pd.DataFrame(responses).to_csv(f"../data/pope_{name}_responses_{i+1}.csv", index=False)

      
def print_metrics():
    data_file="../data/pope_blip2-sft_responses_9000.csv"
    df = pd.read_csv(data_file)

    categories = ['adversarial', 'random', 'popular']
    metrics = {}

    for category in categories + ['all']:
        if category == 'all':
            subset = df
        else:
            subset = df[df['category'] == category]
        
        y_true = subset['answer']
        y_pred = subset['response']
        
        if len(y_true) == 0:
            continue
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label='yes')
        recall = recall_score(y_true, y_pred, pos_label='yes')
        f1 = f1_score(y_true, y_pred, pos_label='yes')
        
        metrics[category] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    for category, metric in metrics.items():
        print(f"Metrics for {category}:")
        print(f"Accuracy: {metric['accuracy']:.4f}")
        print(f"Precision: {metric['precision']:.4f}")
        print(f"Recall: {metric['recall']:.4f}")
        print(f"F1 Score: {metric['f1']:.4f}")
        print()

if __name__ == "__main__":
    # run_eval()
    print_metrics()