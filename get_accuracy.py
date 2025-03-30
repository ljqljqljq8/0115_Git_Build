import os
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer,AutoTokenizer

from peft import PeftModel, PeftConfig
import tqdm
import pandas as pd

model_id = "/home/lab730/Desktop/network-chatbot/llama-3.1-8b/8B"
model_id = "/home/lab730/Desktop/network-chatbot/llama-2-7b/7B"

def get_response(eval_prompt, model, tokenizer):
    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
    # print(f"model inputs: {model_input}")
    model.eval()
    with torch.no_grad():
        # print("-------------------------------------output: -------------------------------------")
        model_output_tokens = model.generate(**model_input, max_new_tokens=800, pad_token_id=tokenizer.eos_token_id)[0]
        model_output = tokenizer.decode(model_output_tokens, skip_special_tokens=True)
        # print(f"model output: {model_output}")
    
    return model_output[ len(eval_prompt)+1: ]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora_path', type=str, help='path to the lora weights')
    parser.add_argument('--dataset_path', type=str, help='path to the test_dataset')
    parser.add_argument('--output_file_path', type=str, help='path to the output_file')
    parser.add_argument('--output_file_name', type=str, help='name of the output_file')
    args = parser.parse_args()

    # tokenizer = LlamaTokenizer.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = LlamaForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16)    
    # model = PeftModel.from_pretrained(model, args.lora_path)
    
    # test_dataset = pd.read_excel('chatbot_datasets/Data_OpenSource.xlsx', sheet_name=args.dataset)
    test_dataset = pd.read_csv(args.dataset_path)

    os.makedirs(args.output_file_path, exist_ok=True)

    output_file_path = os.path.join(args.output_file_path, 'llama2-7b-local.txt')
    output_file_path = os.path.join(args.output_file_path, args.output_file_name)


    output_file = open(output_file_path, "w")
    # Test batch prompt
    for prompt_idx in tqdm.tqdm(range(len(test_dataset['Question']))):

        question = test_dataset['Question'][prompt_idx]
        prompt = (
            f"Answer the question: {question}\n" 
            # "Return the correct result in the format: "
            # "'The correct answer is [ANSWER]'"
            # "(Explanation is not required)"
            # f"Answer the question: {question}\n"
            # "Return the correct result in the format: "
            # "'The correct answer is [ANSWER]'"
            # "(Explanation is not required)"
        )

        response = get_response(prompt, model=model, tokenizer=tokenizer)
        output_file.write(f"Prompt {prompt_idx}:\n{response}\n")

    # test one prompt
    # question = test_dataset['Question'][5]
    # prompt = (
    #         f"Answer the question: {question}\n"
    #         # "Answer the question only with the correct option." 
    #         # "Return: 'The correct answer is: [ANSWER]'. " 
    #         # "Do NOT provide explanations\n"
    #         # f"The question is:\n {question}\n"
    # )

    # response = get_response(prompt, model=model, tokenizer=tokenizer)
    # print(f"Question is: {prompt}")
    # # response = get_response(args.Question_prompt, model=model, tokenizer=tokenizer)
    # print(f"Answer is: {response}")

    # output_file.write(f"prompt 5:\n{response}\n")