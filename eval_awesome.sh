#!/usr/bin/env bash
model_names=(
#    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.1-70B-Instruct"
#    "mistralai/Mistral-7B-Instruct-v0.2"
#    "chatgpt-4o-latest"
#    "gpt-4o-mini"
#    "claude-3-haiku-20240307"
)
prompts_path=(
##    for fuzzing
#    "/scratch/yuzhou/projects/dataleakagents/attacks/token_level/blackbox/fuzz_70b/good_prompts.csv" # fuzz70b
#    "/scratch/yuzhou/projects/dataleakagents/attacks/token_level/blackbox/re_70b/good_prompts.csv" # re70b
    "/scratch/yuzhou/projects/dataleakagents/prompts/fuzz_awesome_gpt4o.csv" # gpt mini
    "/scratch/yuzhou/projects/dataleakagents/prompts/fuzz_awesome_gptmini.csv" # gpt mini
    "/scratch/yuzhou/projects/dataleakagents/prompts/fuzz_awesome_llama8b.csv" # 8b
    "/scratch/yuzhou/projects/dataleakagents/prompts/fuzz_awesome_mix8b.csv" # mix
###    react
    "/scratch/yuzhou/projects/dataleakagents/prompts/re_awesome_gpt4o.csv" # gpt4o
    "/scratch/yuzhou/projects/dataleakagents/prompts/re_awesome_gptmini.csv" # gpt mini
    "/scratch/yuzhou/projects/dataleakagents/prompts/re_awesome_llama8b.csv" # 8b
    "/scratch/yuzhou/projects/dataleakagents/prompts/re_awesome_mix8b.csv" # mix
)
declare -A model_urls
model_urls["meta-llama/Llama-3.1-8B-Instruct"]="http://localhost:54321/v1"
model_urls["meta-llama/Llama-3.1-70B-Instruct"]="http://localhost:54323/v1"
model_urls["mistralai/Mistral-7B-Instruct-v0.2"]="http://localhost:54322/v1"
model_urls["chatgpt-4o-latest"]=""
model_urls["gpt-4o-mini"]=""
model_urls["claude-3-haiku-20240307"]=""


for model_name in "${model_names[@]}"; do
    for prompts_data_path in "${prompts_path[@]}"; do
        echo "--------------------------------------------"
        echo "Running with model: $model_name and prompts data: $prompts_data_path"
        server_url=${model_urls[$model_name]}
        python evaluate_task.py \
            --model_name "$model_name" \
            --prompts_data_path "$prompts_data_path" \
            --n_samples 1 \
            --server_url "$server_url" \
            --api_key empty \
            --dataset_path test_data_pleak.csv
        echo "Finished Running with model: $model_name and prompts data: $prompts_data_path"
        echo "--------------------------------------------"
    done
done