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

def save_in_json(json_str, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    results_file = os.path.join(output_path, "llama3.2_3b_results.json")

    if not os.path.isfile(results_file):
        with open(results_file, "w") as f:
            json.dump([], f)

    with open(results_file, "r") as f:
        json_content = json.load(f)

    json_content.append(json_str)
        
    with open(results_file, "w") as f:
        json.dump(json_content, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    data_path = os.getenv("DATA_PATH", "./data") 
    output_path = os.getenv("OUTPUT_PATH", "./output")  
    

    files = os.listdir(data_path)
    json_files = [file for file in files if file.endswith(".jsonl")]

    print("JSON files found:", json_files)

    model_id = "meta-llama/Llama-3.2-3B"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    for file in tqdm(json_files):
        task = file.split(".")[0]
        with open(os.path.join(data_path, file), "r", encoding='utf-8') as f:
            data = [json.loads(line) for line in f.readlines()]
            print(f"Loaded {len(data)} rows from {file}")
        
        for row in tqdm(data):
            prompt = question_from_row(row)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids

            with torch.no_grad():
                final_logits = model(input_ids).logits[0, -1]
            answer_idxs = torch.tensor([tokenizer.encode(l)[-1] for l in 'ABCD'])
            answer_logits = final_logits[answer_idxs]
            answer_probs = final_logits.softmax(dim=0)[answer_idxs]

            json_str = {
                "file": file,
                "prompt": prompt,
                "predicted": "ABCD"[answer_probs.argmax()],
                "true": row['答案\nANSWER']
            }

            save_in_json(json_str, output_path)
