import copy
import json
import os
import argparse
import re
import tqdm
import os
from openai import OpenAI
import base64
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText, AutoTokenizer
from qwen_vl_utils import process_vision_info
import json
import os
import argparse
import tqdm
import re
from typing import Dict, Optional
import torch

SYSTEM_PROMPT = """
Your response should be in the following format:

<Explanation>
your explanation for your answer choice
</Explanation>
<Answer>
A/B/C/D
</Answer>
<Confidence>
0-100%
</Confidence>
"""

PROMPT = """Question: {question}

Answer Choices: 
A. {A}
B. {B}
C. {C}
D. {D}
"""

MAX_API_RETRY = 20
REQ_TIME_GAP = 4


def extract_answer_fields(text: str) -> Dict[str, Optional[str]]:
    """
    Extract Explanation, Answer, and Confidence fields from model output.

    Expected format:
    <Explanation>...</Explanation>
    <Answer>...</Answer>
    <Confidence>...</Confidence>
    """

    def extract(tag: str) -> Optional[str]:
        pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None

    explanation = extract("Explanation")
    answer = extract("Answer")
    confidence = extract("Confidence")

    # Optional: normalize outputs
    if answer is not None:
        answer = answer.strip().upper()

    if confidence is not None:
        confidence = confidence.strip().rstrip("%")

    return {
        "response_text": text,
        "explanation": explanation,
        "answer_option": answer,
        "confidence": confidence,
    }


def encode_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path {image_path} does not exist.")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def query_api(args, user_prompt, img_path=None, vqa=False):
    base64_image = encode_image(img_path) if img_path else None
    for i in range(MAX_API_RETRY):
        try:
            if vqa:
                if base64_image is None:
                    raise ValueError("base64_image is empty")
                response = client.chat.completions.create(
                    model=args.eval_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": user_prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                                }
                            ]}
                    ],
                    temperature=args.temperature
                    # n = args.n,
                )
                return response.choices[0].message.content
            else:
                response = client.chat.completions.create(
                    model=args.eval_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=args.temperature,
                    # n = args.n,
                )
                return response.choices[0].message.content

        except Exception as e:
            print(f'\n{i} error:\n{e}\n')
    raise RuntimeError(f"Failed after {MAX_API_RETRY} retries.")


def query_local(model, processor=None, user_prompt="", img_path=None, vqa=False):
    for i in range(MAX_API_RETRY):
        try:
            if vqa:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": img_path,
                            },
                            {
                                "type": "text",
                                "text": user_prompt
                            },
                        ],
                    }
                ]

                # Preparation for inference
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                image_inputs, video_inputs = process_vision_info(messages)

                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )

                inputs = inputs.to(device)
                generated_ids = model.generate(**inputs, max_new_tokens=8192)

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                return output_text[0]
            else:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]

                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                inputs = processor(
                    text=[text],
                    padding=True,
                    return_tensors="pt",
                )

                inputs = inputs.to(device)
                generated_ids = model.generate(**inputs, max_new_tokens=8192)

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                return output_text[0]

        except Exception as e:
            print(f'\n{i} error:\n{e}\n')

    raise RuntimeError(f"Failed after {MAX_API_RETRY} retries.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_model", type=str, default="gpt-4o", help="Model to use for evaluation")
    parser.add_argument("--local", action='store_true', help="Use local model for evaluation")
    parser.add_argument("--local_model_path", type=str, default="", help="Model path")
    parser.add_argument("--temperature", type=float, default=1, help="Temperature for the model")
    parser.add_argument("--n", type=int, default=1, help="Number of responses to generate")
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--input_dir", type=str, default="qa")
    parser.add_argument("--type", type=str, default="qa", help="question type, qa or vqa")
    args = parser.parse_args()

    if args.local:
        device = f'cuda:{args.device}'
        model_path = args.local_model_path
        if args.type == 'vqa':
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map="auto"
            )
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map="auto"
            )
            processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"Model device: {model.device}")
    else:
        BASE_URL = os.getenv("BASE_URL")
        API_KEY = os.getenv("API_KEY")
        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )
        print(f"Using llm ")
    data_path = f"{args.input_dir}/{args.type}.json"
    eval_data_path = f"{args.input_dir}/{args.type}_eval_{args.eval_model}_test.json"

    with open(data_path, 'r') as f:
        data = json.load(f)

    if os.path.exists(eval_data_path):
        with open(eval_data_path, 'r') as f:
            eval_data = json.load(f)
        for item in eval_data:
            if item["final_answer"][0] not in ["A", "B", "C", "D"]:
                print(f"Remove invalid eval data {item['id']}")
                eval_data.remove(item)
    else:
        eval_data = []

    new_data = []
    for item in data:
        if any(item['question'] == exit_item['question'] for exit_item in eval_data):
            print(f"Skip {item['id']}")
        else:
            new_data.append(item)
    data = new_data

    for i, item in enumerate(data):

        prompt = PROMPT.format(
            question=item['question'],
            A=item['choices']['A'],
            B=item['choices']['B'],
            C=item['choices']['C'],
            D=item['choices']['D']
        )

        print("Prompt:\n", prompt)
        print('====================================================')

        item['eval'] = []
        item['final_answer'] = []

        for _ in tqdm.tqdm(range(args.n), desc="Processing responses"):
            while True:
                if args.type == 'vqa':
                    img_path = f"{args.input_dir}/image/{item['fig_path']}"
                    if args.local:
                        response = query_local(model, processor, prompt, img_path=img_path, vqa=True)
                    else:
                        response = query_api(args, prompt, img_path=img_path, vqa=True)
                else:
                    if args.local:
                        response = query_local(model, processor, user_prompt=prompt, vqa=False)
                    else:
                        response = query_api(args, prompt, vqa=False)

                result = extract_answer_fields(response)
                if result:
                    item['eval'].append(result)
                    item['final_answer'].append(result['answer_option'])
                    break
                else:
                    print("No match found.")
        correct_rate = sum(1 for eval_item in item['eval'] if eval_item['answer_option'] == item['answer']) / len(
            item['eval'])
        item['correct_rate'] = correct_rate
        eval_data.append(item)
        if not os.path.exists(eval_data_path):
            os.makedirs(os.path.dirname(eval_data_path), exist_ok=True)
        with open(eval_data_path, 'w') as f:
            json.dump(eval_data, f, indent=4)