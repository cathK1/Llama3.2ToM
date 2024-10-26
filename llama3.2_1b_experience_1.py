import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os


def question_from_row(row):
    prompt = f"""
    ### INSTRUCTION: ANSWER WITH A, B, C, OR D ONLY.
    ### STORY: {row['STORY']}
    ### QUESTION: {row['QUESTION']}
    ### A. {row['OPTION-A']}
    ### B. {row['OPTION-B']}
    ### C. {row['OPTION-C']}
    ### D. {row['OPTION-D']}
    ### ANSWER:"""
    return prompt

def save_in_json(json_str):
    if not os.path.isfile("llama3.2_1b_experience_1_results.json"):
        with open("llama3.2_1b_experience_1_results.json", "w") as f:
            json.dump([], f)

    with open("llama3.2_1b_experience_1_results.json", "r") as f:
        json_content = json.load(f)

    json_content.append(json_str)
        
    with open("llama3.2_1b_experience_1_results.json", "w") as f:
        json.dump(json_content, f, ensure_ascii=False, indent=4)


if "__name__" == "__main__":
        
    files = os.listdir("./data")
    json_files = [file for file in files if file.endswith(".jsonl")]

    for file in json_files:
        task = file.split(".")[0]
        with open(f"data/{file}", "r", encoding='utf-8') as f:
            data = [json.loads(line) for line in f.readlines()]

    model_id = "meta-llama/Llama-3.2-1B"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)


    for row in tqdm(data):
        prompt = question_from_row(row)
        print("Prompt:\n", prompt)

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        with torch.no_grad():
            final_logits = model(input_ids).logits[0,-1]
        answer_idxs = torch.tensor([tokenizer.encode(l)[-1] for l in 'ABCD'])
        answer_logits = final_logits[answer_idxs]
        answer_probs = final_logits.softmax(dim=0)[answer_idxs]

        # print('Answer probs (all):', answer_probs)
        # print('Predicted answer:', "ABCD"[answer_probs.argmax()])
        # print('True answer:', row['答案\nANSWER'])

        json_str = {"prompt": prompt,
                    "predicted": "ABCD"[answer_probs.argmax()],
                    "true": row['答案\nANSWER']}

        save_in_json(json_str)