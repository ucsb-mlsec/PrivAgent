# LLMRedAgents

## Code structure

The code is structured as follows:

- `LLMRedAgents/attacks` contains the red-teaming attacks.
    - `AutoRedAgents/methods/sentence_level` contains the sentence level methods.
    - `AutoRedAgents/methods/token_level` contains the token level methods.
- `LLMRedAgents/dataset` contains the system prompts for the system prompt stealing attack and the generated adversarial
  prompts for fine-tuning.
- `LLMRedAgents/defenses` contains the code for fine-tuning with RLHF and SFT.
- `LLMRedAgents/models` contains the target models under testing.
- `LLMRedAgents/reward` contains the reward functions/metrics.
- `LLMRedAgents/risk` contains the code related to individual risks, including specific datasets, actions, and rewards.

## TODO

- [x] support api calling model (vllm, chatgpt)

## How to run

### sentence rl

```shell
# launch vllm for helper model
CUDA_VISIBLE_DEVICES=0,1 vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --disable-log-requests --enable-prefix-caching --port 54321 --api-key empty
# launch vllm for target model
CUDA_VISIBLE_DEVICES=2,3 vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --disable-log-requests --enable-prefix-caching --port 54322 --api-key empty

python pipeline.py --helper_model meta-llama/Llama-3.1-8B-Instruct \
--helper_server_url http://localhost:54321/v1 \
--helper_api_key empty \
--target_model meta-llama/Llama-3.1-8B-Instruct \
--target_server_url http://localhost:54322/v1 \
--target_api_key empty \
--method sent_rl \
--env_num 2

# actually, the helper model and target model can be the same, we can use the same server url
python pipeline.py --helper_model meta-llama/Llama-3.1-8B-Instruct \
--helper_server_url http://localhost:54321/v1 \
--helper_api_key empty \
--target_model meta-llama/Llama-3.1-8B-Instruct \
--target_server_url http://localhost:54321/v1 \
--target_api_key empty \
--method sent_rl \
--env_num 2

# use tensorboard to monitor the training process
tensorboard --logdir=./tensorboard_logs
```

```shell
# evaluate
./eval.sh > eval_output.log 2>&1
```

```shell
cd attacks/token_level/blackbox
# chatgpt 4o
PYTHONPATH=~/projects/dataleakagents accelerate launch --multi_gpu --num_machines 1 --num_processes 6 --main_process_port 24599 FineTuneLLM.py --model_name=meta-llama/Meta-Llama-3-8B-Instruct --adafactor=False --tokenizer_name=meta-llama/Meta-Llama-3-8B-Instruct --load_in_4bit --victim_model chatgpt-4o-latest --output_dir chatgpt_origin_rl_finetune_bonus/ --batch_size 32 --requests_per_minute 20 --target_dataset train_data_pleak.csv --log_with wandb --wandb_exp_name chatgpt_origin_rl_finetune_bonus --wandb_entity rucnyz --epochs 30 --save_freq 2 --use_bonus_reawrd True

# gpt 4o mini
PYTHONPATH=~/projects/dataleakagents accelerate launch --multi_gpu --num_machines 1 --num_processes 6 --main_process_port 24599 FineTuneLLM.py --model_name=meta-llama/Meta-Llama-3-8B-Instruct --adafactor=False --tokenizer_name=meta-llama/Meta-Llama-3-8B-Instruct --load_in_4bit --victim_model gpt-4o-mini --output_dir mini_rl_finetune_bonus/ --batch_size 32 --requests_per_minute 30 --target_dataset train_data_pleak.csv --log_with wandb --wandb_exp_name mini_rl_finetune_bonus --wandb_entity rucnyz --epochs 40 --save_freq 4 --use_bonus_reawrd True

# calude
PYTHONPATH=~/projects/dataleakagents accelerate launch --multi_gpu --num_machines 1 --num_processes 6 --main_process_port 24598 FineTuneLLM.py --model_name=meta-llama/Meta-Llama-3-8B-Instruct --adafactor=False --tokenizer_name=meta-llama/Meta-Llama-3-8B-Instruct --load_in_4bit --victim_model claude-3-haiku-20240307 --output_dir claude_rl_finetune_bonus/ --batch_size 32 --requests_per_minute 20 --target_dataset train_data_pleak.csv --log_with wandb --wandb_exp_name claude_rl_finetune_bonus --wandb_entity rucnyz --epochs 48 --save_freq 4 --use_bonus_reawrd True


# llama 3.1 8B use vllm
vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --disable-log-requests --enable-prefix-caching --port 54321 --api-key empty
# llama 3.1 70B use vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 4 --gpu-memory-utilization 0.9 --disable-log-requests --enable-prefix-caching --port 54323 --api-key empty
#mistralai/Mistral-7B-Instruct-v0.2
vllm serve mistralai/Mistral-7B-Instruct-v0.2 --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --disable-log-requests --enable-prefix-caching --port 54322 --api-key empty


# mistral
PYTHONPATH=~/projects/dataleakagents accelerate launch --multi_gpu --num_machines 1 --num_processes 6 --main_process_port 24599 FineTuneLLM.py --model_name=meta-llama/Meta-Llama-3-8B-Instruct --adafactor=False --tokenizer_name=meta-llama/Meta-Llama-3-8B-Instruct --load_in_4bit --victim_model mistralai/Mistral-7B-Instruct-v0.2 --output_dir mxitral_rl_finetune_bonus/ --batch_size 32 --requests_per_minute 100 --target_dataset train_data_pleak.csv --log_with wandb --wandb_exp_name mxitral_rl_finetune_bonus --wandb_entity rucnyz --epochs 60 --save_freq 4 --use_bonus_reawrd True --server_url http://localhost:54322/v1 --api_key empty

# llama 3.1 8B use vllm
PYTHONPATH=~/projects/dataleakagents accelerate launch --multi_gpu --num_machines 1 --num_processes 6 --main_process_port 24599 FineTuneLLM.py --model_name=meta-llama/Meta-Llama-3-8B-Instruct --adafactor=False --tokenizer_name=meta-llama/Meta-Llama-3-8B-Instruct --load_in_4bit --victim_model meta-llama/Llama-3.1-8B-Instruct --output_dir llama3.1_8B_rl_finetune_bonus_1111/ --batch_size 32 --requests_per_minute 60 --target_dataset awesome_chatgpt_prompts.csv --log_with wandb --wandb_exp_name llama3.1_8B_rl_finetune_bonus --wandb_entity rucnyz --epochs 60 --save_freq 4 --use_bonus_reawrd True --server_url http://localhost:54321/v1 --api_key empty

# llama 3.1 70B use vllm
PYTHONPATH=~/projects/dataleakagents accelerate launch --multi_gpu --num_machines 1 --num_processes 6 --main_process_port 24599 FineTuneLLM.py --model_name=meta-llama/Meta-Llama-3-8B-Instruct --adafactor=False --tokenizer_name=meta-llama/Meta-Llama-3-8B-Instruct --load_in_4bit --victim_model meta-llama/Llama-3.1-70B-Instruct --output_dir llama3.1_70B_rl_finetune_bonus/ --batch_size 32 --requests_per_minute 80 --target_dataset awesome_chatgpt_prompts.csv --log_with wandb --wandb_exp_name llama3.1_70B_rl_finetune_bonus --wandb_entity rucnyz --epochs 60 --save_freq 4 --use_bonus_reawrd True --server_url http://localhost:54323/v1 --api_key empty

# deprecated
# llama3 after defense
PYTHONPATH=~/projects/dataleakagents accelerate launch --multi_gpu --num_machines 1 --num_processes 4 FineTuneLLM.py --model_name=meta-llama/Meta-Llama-3-8B-Instruct --adafactor=False --tokenizer_name=meta-llama/Meta-Llama-3-8B-Instruct --batch_size=32 --load_in_4bit --victim_model ../../../defenses/runs/checkpoint-805 --output_dir after_defense --log_with wandb --wandb_exp_name after_defense_llama3 --wandb_entity rucnyz

cd ../../..
python evaluate_task.py


# sft defense
cd defense
python SFTDefense.py --model_name=meta-llama/Meta-Llama-3-8B-Instruct --tokenizer_name=meta-llama/Meta-Llama-3-8B-Instruct --batch_size=8 --load_in_4bit --log_with wandb --wandb_exp_name sft_defense --wandb_entity rucnyz

PYTHONPATH=~/projects/dataleakagents CUDA_VISIBLE_DEVICES=4,5 accelerate launch --multi_gpu --num_machines 1  --num_processes 2 SFTDefense.py --model_name=meta-llama/Meta-Llama-3-8B-Instruct --tokenizer_name=meta-llama/Meta-Llama-3-8B-Instruct --batch_size=8 --load_in_4bit --log_with wandb --wandb_exp_name sft_defense --wandb_entity rucnyz
# rlhf defense
PYTHONPATH=~/projects/dataleakagents CUDA_VISIBLE_DEVICES=6,7 python RLHFDefense.py --model_name=meta-llama/Meta-Llama-3-8B-Instruct --adafactor=False --tokenizer_name=meta-llama/Meta-Llama-3-8B-Instruct --batch_size=16 --load_in_4bit --output_max_length 128 --log_with wandb --wandb_exp_name rlhf_defense --wandb_entity rucnyz


# utility evaluation
CUDA_VISIBLE_DEVICES=0,1 lm-eval --model hf --model_args pretrained=MODEL_PATH --device=cuda --task=wikitext,lambada
```

### resume

```shell
CUDA_VISIBLE_DEVICES=0,1,3,4,5,6 PYTHONPATH=~/projects/dataleakagents accelerate launch --multi_gpu --num_machines 1 --num_processes 6 --main_process_port 24601 FineTuneLLM.py --model_name=meta-llama/Llama-3.1-8B --adafactor=False --tokenizer_name=meta-llama/Llama-3.1-8B --load_in_4bit --victim_model chatgpt-4o-latest --output_dir rl_finetune_wobonus/ --batch_size 32 --requests_per_minute 20 --target_dataset awesome_chatgpt_prompts.csv --epochs 60 --save_freq 4 --use_bonus_reawrd False --resume True --resume_checkpoint /scratch/yuzhou/projects/dataleakagents/attacks/token_level/blackbox/rl_finetune_wobonus/epoch_10 --log_with wandb --wandb_exp_name rl_finetune_wobonus_2 --wandb_entity rucnyz
```

### vllm

```shell
# vllm
vllm serve meta-llama/Meta-Llama-3.1-405B-Instruct --tensor-parallel-size 8 --gpu-memory-utilization 0.95 --disable-log-requests --enable-prefix-caching --port 54321
curl http://localhost:54321/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Meta-Llama-3.1-405B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```