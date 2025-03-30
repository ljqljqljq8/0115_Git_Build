# Version=3.1
# MODEL_SIZE=8
# DIR=augmented_ablation_fold0
# FOLD_NUM=10
# DATASET=augmented_ablation_fold0
# EPOCH_NUM=0
# # cd llama-3.1-8b/get_results

# LORA_PATH=./Lora_results/0321/chatbot\(finetuned_17k_augmented_ablation_fold0\)/epoch17
# DATASET_NAME=Datasets_17k_balance_test_10fold_0.csv

# OUTPUT_FILE_PATH="./Results/llama-${Version}-${MODEL_SIZE}b/chatbot(finetuned_17k_${DIR})/fold${FOLD_NUM}/epoch${EPOCH_NUM}"


# # Run the Python script with arguments
# python get_prompts.py --lora_path ${LORA_PATH} --dataset ${DATASET_NAME} --output_file_path ${OUTPUT_FILE_PATH}

# echo "Script completed. Results saved to: $OUTPUT_FILE_PATH"

# /home/lab730/Desktop/network-chatbot/llama-3.1-8b/get_results/get_prompts.py

Version=2
MODEL_SIZE=7

# cd llama-3.1-8b/get_results
MODEL_id=/home/lab730/Desktop/network-chatbot/llama-2-7b/7B
LORA_PATH=./Lora_results/0321/chatbot\(finetuned_17k_augmented_ablation_fold0\)/epoch17
DATASET_NAME=Datasets_17k_balance_test_10fold_0.csv

OUTPUT_FILE_PATH="./Results/llama-${Version}-${MODEL_SIZE}b"
OUTPUT_FILE_NAME="llama2-7b-local.txt"

# Run the Python script with arguments
python get_prompts.py --model_id ${MODEL_id} --lora_path ${LORA_PATH} --dataset ${DATASET_NAME} --output_file_path ${OUTPUT_FILE_PATH} --output_file_name ${OUTPUT_FILE_NAME}

echo "Script completed. Results saved to: $OUTPUT_FILE_PATH"