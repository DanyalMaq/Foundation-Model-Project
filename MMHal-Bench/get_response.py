import argparse
import json

import requests
from PIL import Image
from io import BytesIO

from blip2 import build_blip2
from llava_onevision import build_llava_onevision

def load_image(image_file):
    try:
        if image_file.startswith('http') or image_file.startswith('https'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image
    except Exception as e:
        print(f"Error loading image: {image_file}")
        print(e)
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='response_template.json', help='template file containing images and questions')
    parser.add_argument('--output', type=str, default='response_mymodel.json', help='output file containing model responses')
    parser.add_argument('--model', type=str, default='blip2', help='model. blip2 | onevision')
    args = parser.parse_args()

    # build the model using your own code
    if args.model == 'onevision':
        mymodel = build_llava_onevision()
    elif args.model == 'blip2':
        mymodel = build_blip2(sft_lora=False)
    elif args.model == 'blip2-sft':
        mymodel = build_blip2(sft_lora=True)

    json_data = json.load(open(args.input, 'r'))

    for idx, line in enumerate(json_data):
        image_src = line['image_src']
        image = load_image(image_src)
        if image:    
            question = line['question']
            response = mymodel(image, question)
            print(idx, image_src, response)
            line['model_answer'] = response
        else: 
            line['model_answer'] = ''

    with open(args.output, 'w') as f:
        json.dump(json_data, f, indent=2)
