import os
import csv
import re
import datasets
import numpy as np
import pandas as pd
from openpyxl import Workbook

# raw_data_file, parse_data_file, process_data_file, analysis_file = "raw_data.csv", "parse_data.csv", "process_data.csv", "analysis.csv"
# ablation_exps = ["qa", "qa_rephrase", "qar_correct_expl", "qar_correct_wrong1_expl", "qar_correct_wrong2_expl"]

# bots_test = [f"../../data_15k_augment/finetuned_{e}epoch.txt" for e in range(1, 21)] + [
#     f"../../data_15k_ablation_augment/{a}/finetuned_{e}epoch.txt" for a in ablation_exps for e in range(1, 21)
# ] + [
#     "../../baselines/testset_350_results/claude-3-haiku.txt",
#     "../../baselines/testset_350_results/gemma-7b-it.txt",
#     "../../baselines/testset_350_results/gpt-4o-mini.txt",
#     "../../baselines/testset_350_results/llama-2-7b.txt",
#     "../../baselines/testset_350_results/llama-2-13b.txt",
#     # "../baselines/testset_350_results/llama-2-70b.txt",
#     "../../baselines/testset_350_results/mixtral-8x7b-32768.txt",
# ]

# bot_names = [f"our_{e}epoch" for e in range(1, 21)] + [f"{a}_{e}epoch" for a in ablation_exps for e in range(1, 21)] + [
#     "claude-3-haiku",
#     "gemma-7b-it",
#     "gpt-4o-mini",
#     "llama-2-7b",
#     "llama-2-13b",
#     # "llama-2-70b",
#     "mixtral-8x7b-32768",
# ] 
bots_test = [
    # "our+baseline+ablation/Baseline_Results/answers-chatglm3.txt",
    # "our+baseline+ablation/Baseline_Results/answers-gemma:7b.txt",
    # "our+baseline+ablation/Baseline_Results/answers-glm4:9b.txt",
    # "our+baseline+ablation/Baseline_Results/answers-llama2:7b.txt",
    # "our+baseline+ablation/Baseline_Results/answers-llama2:13b.txt",
    # "our+baseline+ablation/Baseline_Results/answers-llama2:70b.txt",
    # "our+baseline+ablation/Baseline_Results/answers-llama3.1:8b.txt",
    # "our+baseline+ablation/Baseline_Results/answers-llava:7b.txt",
    # "our+baseline+ablation/Baseline_Results/answers-mistral:7b.txt",
    # "our+baseline+ablation/Baseline_Results/answers-qwen:7b.txt",
    # "our+baseline+ablation/Baseline_Results/answers-Baichuan2-7B.txt",
    # "our+baseline+ablation/Baseline_Results/answers-chatglm2.txt"
    # "our+baseline+ablation/Baseline_Result_Hard/answers-EntropyYue_chatglm3.txt",
    # "our+baseline+ablation/Baseline_Result_Hard/answers-gemma:7b-instruct-q3_K_L.txt",
    # "our+baseline+ablation/Baseline_Result_Hard/answers-llama2:7b-1.txt",
    # "our+baseline+ablation/Baseline_Result_Hard/answers-llama2:70b-chat-q3_K_M.txt",
    # "our+baseline+ablation/Baseline_Result_Hard/answers-qwen:7b.txt",
    # "our+baseline+ablation/Baseline_Result_Hard/Baichuan2-7B.txt"
    # "our+baseline+ablation/Baseline_Result_Hard/Llava2.txt"
    # "our+baseline+ablation/Baseline_Result_Hard/answers-mistral:7b.txt"
    # "our+baseline+ablation/Baseline_Result_Hard/answers-llama3.1:8b.txt"
    # "our+baseline+ablation/Baseline_Result_Hard/answers-llama2:13b.txt"
    # "Results/llama-3.1-8b/llama3.1-8b-epoch16-Hard-local.txt"
    # "Results/llama-3.1-8b/llama3.1-8b-qa-epoch16-Hard-local.txt"
    # "Results/llama-3.1-8b/llama3.1-8b-epoch17-Hard-local.txt"
    # "Results/llama-3.1-8b/llama3.1-8b-qa-epoch16-Easy-local.txt"
    # "Results/llama-3.1-8b/llama3.1-8b-epoch17-Hard-50-local.txt",
    # "our+baseline+ablation/Baseline_Result_Hard/answers-llama3.1:8b-50-1.txt",
    # "our+baseline+ablation/Baseline_Result_Hard/answers-llava:7b.txt",
    # "our+baseline+ablation/Baseline_Result_Hard/Llava2.txt"
    # "our+baseline+ablation/Baseline_Result_Hard/Baichuan2-7B.txt",
    # "our+baseline+ablation/Baseline_Result_Hard/gemma-7b-local.txt"
    # "our+baseline+ablation/Baseline_Result_Hard/mistral-7b-local.txt",
    # "our+baseline+ablation/Baseline_Result_Hard/Qwen-7B.txt"
    # "our+baseline+ablation/Baseline_Result_Hard/answers-EntropyYue_chatglm3.txt"
    # "Results/llama-3.1-8b/llama3.1-8b-epoch16-Hard-50-local.txt"
    # "our+baseline+ablation/Baseline_Result_Hard/llama2-7b-local.txt"
    # "our+baseline+ablation/Baseline_Result_Hard/answers-llama2:70b-chat-q3_K_M.txt"
    # "our+baseline+ablation/Baseline_Result_Hard/chatglm3.txt"
    # "our+baseline+ablation/Baseline_Result_Hard/llama2-13b-local.txt"
    "our+baseline+ablation/Baseline_Result_Hard/gemma-7b-it-local.txt"
    # "Results/llama-3.1-8b/llama3.1-8b-qa-epoch16-Hard-50-local.txt",
    # "Results/llama-3.1-8b/llama3.1-8b-qar-epoch16-Hard-50-local.txt"
    # "Results/llama-3.1-8b/llama3.1-8b-qa-epoch16-Easy-local.txt"
]

bot_names = [
    # "chatglm3",
    # "gemma-7b",
    # "glm4:9b",
    # "llama2:7b-1",
    # "llama2:13b-Q4",
    # "llama2:70b",
    # "llama3.1:8b-epoch17",
    # "llava:7b",
    # "mistral:7b",
    # "qwen:7b",
    # "Baichuan2-7B",
    # "chatglm2"
    # "chatglm3-local",
    # "llama2-13b-local"
    "gemma-7b-it"
    # "gemma-7b",
    # "llama2:7b-3",
    # # "llama2:13b",
    # "llama2:70b",
    # "qwen:7b",
    # "baichuan-local",
    # "gemma-local"
    # "mistral-local",
    # "qwen-local"
    # "llava2"
    # "llama3.1:8b-epoch16",
    # "llama2-7b-local"
    # "llama3.1:8b",
    # "llama3.1:8b-qa-epoch16",
    # "llama3.1:8b-qar-epoch16"
] 

BASE_PATH = "our+baseline+ablation/Baseline_File_50"
# BASE_PATH = "our+baseline+ablation/Baseline_File_Easy"

# BASE_PATH = "our+baseline+ablation/Baseline_File_All"
# bots_test = ["./Results/llama-3.1-8b/chatbot(finetuned_17k_augmented_ablation_fold0)/fold10/epoch17/8b_output_batch_epoch0.txt"]
# bot_names = ['llama3.1-8b-epoch0']

# bots_test = ["./Results/llama-2-7b/answers-llama2:7b-chat-q3_K_L.txt"]
# bot_names = ['llama2:7b-local']

# bots_test = ["./Results/llama-2-7b/answers-llama2:7b-chat-q3_K_S.txt"]
# bot_names = ['llama2:7b-local-v3']

# bots_test = ["./our+baseline+ablation/Baseline_Results/answers-llama2:70b-chat-q3_K_M.txt"]
# bot_names = ['llama2:70b-local']


# bots_test = ["./Results/llama-3.1-8b/chatbot(finetuned_17k_augmented_ablation_fold0)/fold10/epoch3/8b_output_batch_CA1.txt"]
# bot_names = ['llama3.1-8b-epoch0-Final-1']

# bots_test = ["our+baseline+ablation/Baseline_Results/answer-llama2-7b-new.txt"]
# bot_names = ['llama2-7b-new']
# raw_data_file = os.path.join(BASE_PATH, bot_names[0], f"{bot_names[0]}-raw_data.csv")
# parse_data_file = os.path.join(BASE_PATH, bot_names[0], f"{bot_names[0]}-parse_data.csv")
# process_data_file = os.path.join(BASE_PATH, bot_names[0], f"{bot_names[0]}-process_data.csv")
# analysis_file = os.path.join(BASE_PATH, bot_names[0], f"{bot_names[0]}-analysis.csv")


answers = ""
metrics = ['TP', 'FP', 'FN']
choices = ['A', 'B', 'C', 'D']

ill_cnt = 0

def match_choice(match:str):
    # print(f"{match = }")
    return re.search(re.compile(r'(?:[^a-zA-Z]|^)([ABCD])(?:[^a-zA-Z]|$)'), match).group(1)
def filter_prompt(prompt: str, pattern: re.Pattern):
    """
    Ê¹ÓÃ¸ø¶¨ÕýÔòÔÚ prompt ÖÐ²éÕÒËùÓÐÆ¥Åä£¬
    ½«½á¹ûÍ³Ò»×ª»»Îª´óÐ´×ÖÄ¸£¬²¢È¥³ýÖØ¸´¡£
    Èç¹ûÖ»ÓÐÒ»¸öÎ¨Ò»´ð°¸£¬Ôò·µ»Ø¸Ã´ð°¸£»
    Èç¹û´æÔÚ¶à¸öÇÒ²»Ò»ÖÂ£¬Ôò·µ»Ø "X" ±íÊ¾²»Ò»ÖÂ£»
    Èç¹ûÎ´Æ¥Åäµ½£¬Ôò·µ»Ø None¡£
    """
    matches = pattern.findall(prompt)
    results = []
    for m in matches:
        # m ¿ÉÄÜÎª×Ö·û´®»òÔª×é£¨µ±ÕýÔòÖÐÓÐ¶à¸ö²¶»ñ×éÊ±£©
        candidate = m[0] if isinstance(m, tuple) else m
        candidate = candidate.strip().upper()
        if candidate in choices:
            results.append(candidate)
    # È¥ÖØ£¨±£³ÖË³Ðò£©
    unique = []
    for r in results:
        if r not in unique:
            unique.append(r)
    if len(unique) == 1:
        return unique[0]
    elif len(unique) > 1:
        return "X"  # ¶à¸ö²»Í¬´ð°¸
    return None

def get_choice(prompt: str) -> str:

    global ill_cnt
    prompt = prompt.strip()
    if len(prompt) == 1:
        return prompt.upper() if prompt.upper() in choices else "_"
    if len(prompt) == 0:
        return "_"
    # Èç¹ûÌáÊ¾Ê××Ö·û¼´ÎªÑ¡Ïî£¬ÇÒ½ô¸ú»»ÐÐ£¬ÔòÖ±½Ó·µ»Ø¸Ã×ÖÄ¸
    if len(prompt) >= 2 and prompt[0].upper() in choices and prompt[1] == '\n':
        return prompt[0].upper()
    
    patterns = [
        # Æ¥Åä "The correct answer is (D)"£¨¿ÉÑ¡ option ºÍÀ¨ºÅ£©
        re.compile(r'[Tt]he\s+correct\s+answer\s+is\s*(?:option\s+)?\(?\s*([ABCD])\s*\)?'),
        # Æ¥Åä "Correct answer:" ¸ñÊ½
        re.compile(r'Correct\s+answer:\s*(?:option\s+)?\(?\s*([ABCD])\s*\)?'),
        # Æ¥Åä "The answer is (C)" ¸ñÊ½
        re.compile(r'[Tt]he\s+answer\s+is\s*(?:option\s+)?\(?\s*([ABCD])\s*\)?'),
        # ÐÂÔö£ºÆ¥ÅäÀàËÆ "nswer:" »ò "swer:" ¸ñÊ½£¨ÔÊÐíÈ±Ê§ "a" »ò "an"£©
        re.compile(r'(?:(?:[Aa]?nswer)|swer):\s*(?:option\s+)?\(?\s*([ABCD])\s*\)?', re.IGNORECASE),
        # Æ¥Åä "Answer:" ¸ñÊ½
        re.compile(r'Answer:\s*(?:option\s+)?\(?\s*([ABCD])\s*\)?'),
        # »ØÍËÄ£Ê½1£ºÆ¥ÅäÐÐÊ×ÒÔÀ¨ºÅ°üÎ§µÄ´ð°¸£¬Èç "(A)" »ò "B)" ¿ªÍ·£¨¶àÐÐÄ£Ê½£©
        re.compile(r'^[\(\s]*([ABCD])[\)\.\s]', re.MULTILINE),
        # »ØÍËÄ£Ê½2£ºÆ¥Åä¶ÀÁ¢³öÏÖµÄ´ð°¸×ÖÄ¸£¨×óÓÒ²»ÄÜÓÐÆäËû×ÖÄ¸£©
        re.compile(r'(?:[^A-Za-z]|^)\(?\s*([ABCD])\s*\)?(?:[^A-Za-z]|$)')
    ]
    
    for pattern in patterns:
        result = filter_prompt(prompt, pattern)
        if result is not None and result != "":
            if result in choices:
                return result
            elif result == "X":
                # Èç¹ûµ±Ç°Ä£Ê½´æÔÚ²»Ò»ÖÂ½á¹û£¬Ôò¼ÌÐø³¢ÊÔÆäËûÄ£Ê½
                continue
    ill_cnt += 1
    return "_"

# def filter_prompt(prompt:str, pattern:re.Pattern):
#     # print(f"{prompt = }, {pattern = }")
#     matches = pattern.findall(prompt)
#     matches = [match_choice(m) for m in matches]
#     if len(matches) > 1:
#         for m in matches:
#             if m != matches[0]:
#                 # print(f"Inconsistent results: {prompt = }")
#                 return matches
#         return match_choice(matches[0])
#     if len(matches) == 0:
#         # print(f"No Answer found: {prompt = }")
#         return "_"
#     return matches[0]

# def get_choice(prompt:str) -> str:
#     prompt = prompt.strip('\n').strip(' ').strip('\t')
#     if len(prompt) == 1:
#         prompt = prompt.upper()
#         return prompt if prompt in choices else "_"
#     elif len(prompt) == 0:
#         return "_"

#     if len(prompt) >= 2 and prompt[0] in choices and prompt[1] == '\n':
#         return prompt[0]


#     # matching """Correct answer:\n[ABCD])"""
#     filter_result = filter_prompt(prompt, re.compile(r'Correct answer:[\n\t ]*[ABCDabcd]\)', re.MULTILINE))
#     if len(filter_result) == 1 and filter_result in choices:
#         return filter_result
#     elif len(filter_result) > 1:
#         pass
#         # print(f"Inconsistent results in matching {r'Correct answer:\n[ABCDabcd]\)'} in prompt: {prompt}")

#     # matching """[ABCD])"""
#     filter_result = filter_prompt(prompt, re.compile(r'^[ABCD]\)', re.MULTILINE))
#     if len(filter_result) == 1 and filter_result in choices:
#         return filter_result
#     elif len(filter_result) > 1:
#         pass
#         # print(f"Inconsistent results in matching {r'^[ABCD\)]'} in prompt: {prompt}")
    
#     # matching """The correct answer is [ABCD])"""
#     filter_result = filter_prompt(prompt, re.compile(r'[Tt]he correct answer is (?:option )?[ABCD]', re.MULTILINE))
#     if len(filter_result) == 1 and filter_result in choices:
#         return filter_result
#     elif len(filter_result) > 1:
#         pass
#         # print(f"Inconsistent results in matching {r'The correct answer is [ABCD]'} in prompt: {prompt}")

#     # matching """The answer is [ABCD])"""
#     filter_result = filter_prompt(prompt, re.compile(r'[Tt]he answer is (?:option )?[ABCD]', re.MULTILINE))
#     if len(filter_result) == 1 and filter_result in choices:
#         return filter_result
#     elif len(filter_result) > 1:
#         pass
#         # print(f"Inconsistent results in matching {r'The answer is [ABCD]'} in prompt: {prompt}")

#     # matching """[beginning or non-letter][ABCD][ending or non-letter]"""
#     filter_result = filter_prompt(prompt, re.compile(r'(?:[^a-zA-Z]|^)[ABCD](?:[^a-zA-Z]|$)', re.MULTILINE))
#     if len(filter_result) == 1 and filter_result in choices:
#         return filter_result
#     elif len(filter_result) > 1:
#         # print(f"Inconsistent results in matching {r'(?:[^a-zA-Z]|^)[ABCD](?:[^a-zA-Z]|$)'} in prompt: {prompt}\n")
#         # return input("Please check the answer of the prompt and enter ('X' for inconsistent results, '_' for no answer): ")
#         pass

#     global ill_cnt
#     if len(filter_result) == 0:
#         ill_cnt += 1
#         return "_"
#     elif len(filter_result) > 1:
#         ill_cnt += 1
#         return "X"
#     return filter_result[0]

def evaluate_prompt(prompt, expected_answer) -> int:
    choice = get_choice(prompt)
    global answers
    answers += choice
    if choice == expected_answer.upper():
        return 1
    return 0

def csv_to_excel(csv_files, output_file):
    wb = Workbook()
    wb.remove(wb.active)
    for csv_file in csv_files:
        sheet_name = csv_file.split('.')[0]
        ws = wb.create_sheet(title=sheet_name)
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                ws.append(row)
    wb.save(output_file)

if __name__ == '__main__':

    csv_file = "HardDataset-50.csv" 
    # csv_file = "Hard-set.csv" 
    # csv_file = "Datasets_17k_balance_test_10fold_0.csv" 

    test_set = pd.read_csv(csv_file)
    # test_set = pd.read_csv("/home/lab730/Desktop/network-chatbot/llama-3.1-8b/chatbot_datasets/Datasets_17k_balance_10fold/test/Datasets_17k_balance_test_10fold_0.csv")
    correct_answer = [response.lstrip(' \n\t')[0] for response in test_set['Correct Answer']]
    correct_answer = [c.upper() for c in correct_answer]
    answers += "".join(ans for ans in correct_answer)

    for i, (bot_test, bot_name) in enumerate(zip(bots_test, bot_names)):
        ill_cnt = 0
        valid_cnt = { bot_name:0 }
        data_cnt = { f"{bot_name}_{c}_{m}":0 for m in metrics for c in choices}
        data_cnt.update({ f"{bot_name}_{c}_total":0 for c in choices})

        bot_path = os.path.join(BASE_PATH, bot_name)
        if not os.path.exists(bot_path):
            os.makedirs(bot_path)

        raw_data_file = os.path.join(BASE_PATH, bot_name, f"{bot_name}-raw_data.csv")
        parse_data_file = os.path.join(BASE_PATH, bot_name, f"{bot_name}-parse_data.csv")
        process_data_file = os.path.join(BASE_PATH, bot_name, f"{bot_name}-process_data.csv")
        analysis_file = os.path.join(BASE_PATH, bot_name, f"{bot_name}-analysis.csv")

        raw_data = []
        parse_data = []
        process_data = []

        # main judging part
        # print(f"Evaluating {bot_name}")
        # input_file = open(f"./old/{bot_name}.txt", 'r')
        input_file = open(f"{bot_test}", 'r')
        raw_data_row = []
        parse_data_row = []
        process_data_row = []
        prompt_idx = 0
        prompt = ""

        for line in input_file:# prompt {prompt_idx}:\n
            if line == f'Prompt {prompt_idx}:\n':
                # print(f"evaluate prompt: {prompt}")
                if prompt_idx != 0:
                    choice, correct_choice = get_choice(prompt), correct_answer[prompt_idx-1]
                    eval_result = int(choice == correct_choice)
                    valid_cnt[bot_name] += eval_result

                    raw_data_row.append(prompt)
                    parse_data_row.append(choice)
                    process_data_row.append(eval_result)

                    # print(f"{correct_choice = }, {prompt_idx = }, {ablation_name = }")
                    data_cnt[bot_name+"_"+correct_choice+"_total"] += 1
                    if eval_result:
                        data_cnt[bot_name+"_"+correct_choice+"_TP"] += 1
                    else:
                        # os.makedirs(os.path.dirname(f"wrong_answers/{bot_name}.csv"), exist_ok=True)
                        # with open(f"wrong_answers/{bot_name}.csv", "a+") as f:
                        #     writer = csv.writer(f)
                        #     writer.writerow([test_set[prompt_idx-1]['Q&A'], prompt])
                        data_cnt[bot_name+"_"+correct_choice+"_FP"] += 1
                        if choice in choices:
                            data_cnt[bot_name+"_"+choice+"_FN"] += 1
                prompt_idx += 1
                prompt = ""
            else:
                prompt += line
        # evaluate last prompt
        choice, correct_choice = get_choice(prompt), correct_answer[prompt_idx-1]
        eval_result = int(choice == correct_choice)

        raw_data_row.append(prompt)
        parse_data_row.append(choice)
        process_data_row.append(eval_result)

        valid_cnt[bot_name] += eval_result
        data_cnt[bot_name+"_"+correct_choice+"_total"] += 1
        if eval_result:
            data_cnt[bot_name+"_"+correct_choice+"_TP"] += 1
        else:
            # os.makedirs(os.path.dirname(f"wrong_answers/{bot_name}.csv"), exist_ok=True)
            # with open(f"wrong_answers/{bot_name}.csv", "a+") as f:
            #     writer = csv.writer(f)
            #     writer.writerow([test_set[prompt_idx-1]['Q&A'], prompt])
            data_cnt[bot_name+"_"+correct_choice+"_FP"] += 1
            if choice in choices:
                data_cnt[bot_name+"_"+choice+"_FN"] += 1

        # print(f"{bot_name:15} Valid: {valid_cnt[bot_name]}")
        raw_data.append(raw_data_row)
        parse_data.append(parse_data_row)
        process_data.append(process_data_row)

        # print(f"{'# of correct responses ('+str(len(correct_answer))+' in total)'}")

        raw_data = np.array(raw_data).T
        parse_data = np.array(parse_data).T
        process_data = np.array(process_data).T
    
        # write raw data
        # os.system(f"rm {raw_data_file}")
        with open(raw_data_file, "w") as csvfile:
            writer = csv.writer(csvfile)
            raw_data_header = ["Question", "Answer"] + [bot_name]
            writer.writerow(raw_data_header)
            for i in range(len(test_set)):
                row = [test_set['Question'][i], test_set['Correct Answer'][i]]
                row += [raw_data[i][0]]
                writer.writerow(row)

        # write parsed data
        # os.system(f"rm {parse_data_file}")
        with open(parse_data_file, "w") as csvfile:
            writer = csv.writer(csvfile)
            parse_data_header = ["Correct Choice"] + [bot_name]
            writer.writerow(parse_data_header)
            for i in range(len(test_set)):
                row = [correct_answer[i]] + [parse_data[i][0]]
                writer.writerow(row)

        # write processed data
        # os.system(f"rm {process_data_file}")
        with open(process_data_file, "w") as csvfile:
            writer = csv.writer(csvfile)
            process_data_header = [bot_name]
            writer.writerow(process_data_header)
            for i in range(len(test_set)):
                row = [process_data[i][0]]
                writer.writerow(row)
        
        # write analysis data
        # os.system(f"rm {analysis_file}")
        with open(analysis_file, "w") as csvfile:
            writer = csv.writer(csvfile)
            analysis_header = ['Model', "Correct Count", "Total Num of Questions"]
            for c in choices:
                analysis_header += [f"{c}_{m}" for m in metrics] + [f'total_num_of_{c}']
            writer.writerow(analysis_header)
            total_question_num = len(correct_answer)

            row: list = [f'{bot_name}']
            row.append(valid_cnt[bot_name])
            # print(f"{bot_name:40} {valid_cnt[bot_name]} correct in {total_question_num}, accuracy: {valid_cnt[bot_name]/total_question_num}")
        
            row.append(total_question_num)
            for c in choices:
                row += [data_cnt[f"{bot_name}_{c}_{m}"] for m in metrics] + [data_cnt[f"{bot_name}_{c}_total"]]
            writer.writerow(row)
            # print("")
    
        # csv_to_excel([raw_data_file, parse_data_file, process_data_file, analysis_file], os.path.join(BASE_PATH, "lama3-8b.xlsx"))
        
        print(f"{ill_cnt = }")

    # raw_data = []
    # parse_data = []
    # process_data = []

    # for i, (bot_name, bot) in enumerate(zip(bots_test, bot_names)):

    #     raw_data_file = os.path.join(BASE_PATH, bot, f"{bot}-raw_data.csv")
    #     parse_data_file = os.path.join(BASE_PATH, bot, f"{bot}-parse_data.csv")
    #     process_data_file = os.path.join(BASE_PATH, bot, f"{bot}-process_data.csv")
    #     analysis_file = os.path.join(BASE_PATH, bot, f"{bot}-analysis.csv")
    # # main judging part
    # # for bot_name in bots_test:

    #     input_file = open(f"{bot_name}", 'r')
    #     raw_data_row = []
    #     parse_data_row = []
    #     process_data_row = []
    #     prompt_idx = 0
    #     prompt = ""

    #     for line in input_file:# prompt {prompt_idx}:\n
    #         if line == f'prompt {prompt_idx}:\n':
    #             # print(f"evaluate prompt: {prompt}")
    #             if prompt_idx != 0:
    #                 choice, correct_choice = get_choice(prompt), correct_answer[prompt_idx-1]
    #                 eval_result = int(choice == correct_choice)
    #                 valid_cnt[bot_name] += eval_result

    #                 raw_data_row.append(prompt)
    #                 parse_data_row.append(choice)
    #                 process_data_row.append(eval_result)

    #                 # print(f"{correct_choice = }, {prompt_idx = }, {ablation_name = }")
    #                 data_cnt[bot_name+"_"+correct_choice+"_total"] += 1
    #                 if eval_result:
    #                     data_cnt[bot_name+"_"+correct_choice+"_TP"] += 1
    #                 else:
    #                     # os.makedirs(os.path.dirname(f"wrong_answers/{bot_name}.csv"), exist_ok=True)
    #                     # with open(f"wrong_answers/{bot_name}.csv", "a+") as f:
    #                     #     writer = csv.writer(f)
    #                     #     writer.writerow([test_set[prompt_idx-1]['Q&A'], prompt])
    #                     data_cnt[bot_name+"_"+correct_choice+"_FP"] += 1
    #                     if choice in choices:
    #                         data_cnt[bot_name+"_"+choice+"_FN"] += 1
    #             prompt_idx += 1
    #             prompt = ""
    #         else:
    #             prompt += line
    #     # evaluate last prompt
    #     choice, correct_choice = get_choice(prompt), correct_answer[prompt_idx-1]
    #     eval_result = int(choice == correct_choice)

    #     raw_data_row.append(prompt)
    #     parse_data_row.append(choice)
    #     process_data_row.append(eval_result)

    #     valid_cnt[bot_name] += eval_result
    #     data_cnt[bot_name+"_"+correct_choice+"_total"] += 1
    #     if eval_result:
    #         data_cnt[bot_name+"_"+correct_choice+"_TP"] += 1
    #     else:
    #         # os.makedirs(os.path.dirname(f"wrong_answers/{bot_name}.csv"), exist_ok=True)
    #         # with open(f"wrong_answers/{bot_name}.csv", "a+") as f:
    #         #     writer = csv.writer(f)
    #         #     writer.writerow([test_set[prompt_idx-1]['Q&A'], prompt])
    #         data_cnt[bot_name+"_"+correct_choice+"_FP"] += 1
    #         if choice in choices:
    #             data_cnt[bot_name+"_"+choice+"_FN"] += 1

    #     # print(f"{bot_name:15} Valid: {valid_cnt[bot_name]}")
    #     raw_data.append(raw_data_row)
    #     parse_data.append(parse_data_row)
    #     process_data.append(process_data_row)

    # # print(f"{'# of correct responses ('+str(len(correct_answer))+' in total)'}")

    # raw_data = np.array(raw_data).T
    # parse_data = np.array(parse_data).T
    # process_data = np.array(process_data).T
   
    # # write raw data
    # # os.system(f"rm {raw_data_file}")
    # with open(raw_data_file, "w") as csvfile:
    #     writer = csv.writer(csvfile)
    #     raw_data_header = ["Question", "Answer"] + bot_names
    #     writer.writerow(raw_data_header)
    #     for i in range(len(test_set)):
    #         row = [test_set['Question'][i], test_set['Correct Answer'][i]]
    #         row += [raw_data[i][j] for j in range(len(bots_test))]
    #         writer.writerow(row)

    # # write parsed data
    # # os.system(f"rm {parse_data_file}")
    # with open(parse_data_file, "w") as csvfile:
    #     writer = csv.writer(csvfile)
    #     parse_data_header = ["Correct Choice"] + bot_names
    #     writer.writerow(parse_data_header)
    #     for i in range(len(test_set)):
    #         row = [correct_answer[i]] + [parse_data[i][j] for j in range(len(bots_test))]
    #         writer.writerow(row)

    # # write processed data
    # # os.system(f"rm {process_data_file}")
    # with open(process_data_file, "w") as csvfile:
    #     writer = csv.writer(csvfile)
    #     process_data_header = bot_names
    #     writer.writerow(process_data_header)
    #     for i in range(len(test_set)):
    #         row = [process_data[i][j] for j in range(len(bots_test))]
    #         writer.writerow(row)
    
    # # write analysis data
    # # os.system(f"rm {analysis_file}")
    # with open(analysis_file, "w") as csvfile:
    #     writer = csv.writer(csvfile)
    #     analysis_header = ['Model', "Correct Count", "Total Num of Questions"]
    #     for c in choices:
    #         analysis_header += [f"{c}_{m}" for m in metrics] + [f'total_num_of_{c}']
    #     writer.writerow(analysis_header)
    #     total_question_num = len(correct_answer)

    #     for i, bot_name in enumerate(bots_test):
    #         row: list = [f'{bot_names[i]}']
    #         row.append(valid_cnt[bot_name])
    #         # print(f"{bot_name:40} {valid_cnt[bot_name]} correct in {total_question_num}, accuracy: {valid_cnt[bot_name]/total_question_num}")
        
    #         row.append(total_question_num)
    #         for c in choices:
    #             row += [data_cnt[f"{bot_name}_{c}_{m}"] for m in metrics] + [data_cnt[f"{bot_name}_{c}_total"]]
    #         writer.writerow(row)
    #         # print("")
    
    # # csv_to_excel([raw_data_file, parse_data_file, process_data_file, analysis_file], os.path.join(BASE_PATH, "lama3-8b.xlsx"))
    
    # print(f"{ill_cnt = }")

