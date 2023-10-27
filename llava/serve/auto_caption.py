import argparse
from pprint import pprint
from tqdm import tqdm
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

DEFAULT_FRAME_CAPTIONING_PROMPT = """This image is a frame from an animated movie. Describe the scene in detail. Describe composition and movement. Do not use hedge words like "may be", "possibly", or "appears to". Make your best guess and state your opinion as fact. Answer in three lines or less. Your objective is to output a caption, so you should not format your response as an answer to a question, but as a caption"""

DEFAULT_SCENE_CAPTIONING_PROMPT = """"WIP"""

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def caption_batch(args, image_batches):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)

    # if 'llama-2' in model_name.lower():
    #     conv_mode = "llava_llama_2"
    # elif "v1" in model_name.lower():
    #     conv_mode = "llava_v1"
    # elif "mpt" in model_name.lower():
    #     conv_mode = "mpt"
    # else:
    #     conv_mode = "llava_v0"

    batch_captions = []
    for batch in tqdm(image_batches):
        if not batch:
            raise ValueError(f"Invalid image batch: {batch}")

        if isinstance(batch[0], str):
            batch = [load_image(path) for path in batch]
        frame_captions = []

        for image in batch:
            #create new conversation (reset context) for every image

            caption = caption_image(args, tokenizer, model, image_processor, image)
            # print(f"caption: = {caption}")
            frame_captions.append(caption)
        batch_captions.append(frame_captions)
    return batch_captions

def caption_image(args, tokenizer, model, image_processor, image, prompt=None):

    model_name = get_model_name_from_path(args.model_path)
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles


    if prompt is None:
        prompt = DEFAULT_FRAME_CAPTIONING_PROMPT
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    if image is not None:
            # first message
        if model.config.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], prompt)
        image = None
    else:
            # later messages
        conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    conv.messages[-1][-1] = outputs

    if args.debug:
        print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
    return outputs


parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
parser.add_argument("--model-base", type=str, default=None)
# parser.add_argument("--image-file", type=str, required=True)
parser.add_argument("--num-gpus", type=int, default=1)
parser.add_argument("--conv-mode", type=str, default=None)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--load-8bit", action="store_true")
parser.add_argument("--load-4bit", action="store_true")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
    # parser.add_argument("--model-base", type=str, default=None)
    # # parser.add_argument("--image-file", type=str, required=True)
    # parser.add_argument("--num-gpus", type=int, default=1)
    # parser.add_argument("--conv-mode", type=str, default=None)
    # parser.add_argument("--temperature", type=float, default=0.2)
    # parser.add_argument("--max-new-tokens", type=int, default=512)
    # parser.add_argument("--load-8bit", action="store_true")
    # parser.add_argument("--load-4bit", action="store_true")
    # parser.add_argument("--debug", action="store_true")
    # args = parser.parse_args()

    #image batches is a list of batches of n frames from the same scene.
    test_paths2 = ['https://canvascontent.krea.ai/1aadc7f7-1438-4cc4-8099-646a5503b407.png', 'https://canvascontent.krea.ai/9a3389a9-d226-418d-9518-4dd5eaf73790.png', 'https://canvascontent.krea.ai/7788fe1d-6fb6-4592-bfd0-f94870e407c2.png']
    test_paths = ['ov1.jpeg', 'ov2.jpeg', 'ov3.jpeg']

    image_batches = [
        [load_image(path) for path in test_paths], 
        [load_image(path) for path in test_paths2], 
    ]
    pprint(image_batches)
    # batch_captions = caption_batch(args, image_batches)
    
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)

    a = caption_image(args, tokenizer, model, image_processor, load_image(test_paths[0]))
    print("a: ", a)
