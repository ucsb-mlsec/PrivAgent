import os
from dataclasses import dataclass, field
from typing import Optional

import bitsandbytes
import numpy as np
import pandas as pd
from datasets import Dataset

import torch
from peft import LoraConfig  # for LoRA
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoTokenizer,
    HfArgumentParser,
)  # HuggingFace transformer models

from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    set_seed,
)  # Transformer RL
from transformers import BitsAndBytesConfig  # for quantization
from trl.core import LengthSampler

from project_env import PROMPT_PATH
from rewards.text_rewards import TextRewards  # our reward function

tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine-tune with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(
        default="", metadata={"help": "the tokenizer name"}
    )
    log_with: Optional[str] = field(
        default=None, metadata={"help": "use 'wandb' to log with wandb"}
    )
    wandb_exp_name: Optional[str] = field(
        default="default", metadata={"help": "the wandb experiment name"}
    )
    wandb_entity: Optional[str] = field(
        default=None, metadata={"help": "the wandb experiment name"}
    )
    learning_rate: Optional[float] = field(
        default=1.41e-5, metadata={"help": "the learning rate"}
    )
    output_max_length: Optional[int] = field(
        default=64, metadata={"help": "maximum length for generation"}
    )
    mini_batch_size: Optional[int] = field(
        default=1, metadata={"help": "the PPO minibatch size"}
    )
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(
        default=4, metadata={"help": "the number of ppo epochs"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(
        default=False, metadata={"help": "whether to use the adafactor optimizer"}
    )
    early_stopping: Optional[bool] = field(
        default=False, metadata={"help": "whether to early stop"}
    )
    target_kl: Optional[float] = field(
        default=0.1, metadata={"help": "kl target for early stopping"}
    )
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(
        default=False, metadata={"help": "whether to use the batched text gen"}
    )
    save_freq: Optional[int] = field(
        default=1, metadata={"help": "n steps to save the model"}
    )
    output_dir: Optional[str] = field(
        default="runs/rlhf/", metadata={"help": "n steps to save the model"}
    )
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={
            "help": "Initial KL penalty coefficient (used for adaptive and linear control)"
        },
    )

    adap_kl_ctrl: Optional[bool] = field(
        default=True, metadata={"help": "Use adaptive KL control, otherwise linear"}
    )
    load_in_8bit: Optional[bool] = field(
        default=False, metadata={"help": "whether to load the model in 8bit"}
    )
    load_in_4bit: Optional[bool] = field(
        default=False, metadata={"help": "whether to load the model in 4bit"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "use gradient checkpointing"}
    )
    # for 4bit quantization
    use_nested_quant: Optional[bool] = field(
        default=True, metadata={"help": "whether to use nested quant"}
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "Quantization type fp4 or nf4"}
    )
    # epochs
    epochs: Optional[int] = field(default=5, metadata={"help": "number of epochs"})


def build_dataset(tokenizer, dataset):
    num_proc = 4

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for query in examples["text"]:
            tokenized_question = tokenizer(query, padding=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns="text",
    )
    # todo: the filter will filter inputs shorter than 512? [just a question]
    # ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched = False, num_proc = num_proc)

    ds.set_format(type="torch")
    return ds


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


def score_refuse(response):
    rewards = []
    refuse_prompt = [
        "Sorry I cannot process your request",
        "I'm sorry, I cannot",
        "I apologize, I cannot",
        "I'm sorry, I can't",
    ]
    for i in range(len(response)):
        rewards.append(
            np.mean(
                TextRewards.distance_lcs(
                    [response[i]] * len(refuse_prompt), refuse_prompt
                )
            )
        )
    return rewards


def score_unrelated(response, prompt):
    # todo: the prompt should be the original system prompt?
    rewards = TextRewards.distance_lcs(response, prompt)
    return [1 - score for score in rewards]


def score(response, prompt):
    # unrelated = score_unrelated(response, prompt)
    refuse = score_refuse(response)
    return [
        torch.FloatTensor([i_refuse]) for i_refuse in refuse
    ]  # average of the two scores


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    config = PPOConfig(
        steps=script_args.steps,
        model_name=script_args.model_name,
        learning_rate=script_args.learning_rate,
        log_with=script_args.log_with,
        tracker_kwargs={
            "wandb": {
                "entity": script_args.wandb_entity,
                "name": script_args.wandb_exp_name,
            }
        },
        tracker_project_name="rl_finetune_llm",
        batch_size=script_args.batch_size,
        mini_batch_size=script_args.mini_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=script_args.early_stopping,
        target_kl=script_args.target_kl,
        ppo_epochs=script_args.ppo_epochs,
        seed=script_args.seed,
        init_kl_coef=script_args.init_kl_coef,  # 0.2
        adap_kl_ctrl=script_args.adap_kl_ctrl,  # Todo: how does the adaptive kl control work? [just a question]
        gradient_checkpointing=script_args.gradient_checkpointing,
        remove_unused_columns=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.tokenizer_name, padding_side="left"
    )
    # GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
    # only for this model.

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create the dataset
    # training_num = 1000
    # train_dataset = Dataset.from_dict({"questions": ["Please generate a prompt for me: "] * training_num, })
    attack_prompts = pd.read_csv(os.path.join(PROMPT_PATH, "attack_prompt.csv"))[
        "text"
    ].tolist()
    training_targets = pd.read_csv(os.path.join(PROMPT_PATH, "train_data_pleak.csv"))[
        "text"
    ].tolist()
    messages_lst = []
    for prompt in attack_prompts:
        for text in training_targets:
            messages_lst.append([
                {"role": "system", "content": text},
                {"role": "user", "content": prompt},
            ])
    messages = tokenizer.apply_chat_template(
        messages_lst, tokenize=False, add_generation_prompt=True
    )

    dataset = Dataset.from_dict({
        "text": messages,
        "system_prompt": [
            messages_lst[i][0]["content"] for i in range(len(messages_lst))
        ],
    })
    dataset = build_dataset(tokenizer, dataset)

    # training_targets = pd.read_csv('train_data_pleak.csv')['text'].tolist()
    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    # Now let's build the model, the reference model, and the tokenizer.
    # current_device = Accelerator().local_process_index

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if script_args.load_in_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True, bnb_compute_dtype=torch.float16
        )
    elif script_args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type=script_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=script_args.use_nested_quant,
        )
    else:
        raise ValueError("Please specify a quantization config")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        quantization_config=quant_config,
        device_map="auto",
        peft_config=lora_config,
        low_cpu_mem_usage=True if quant_config is not None else False,
        trust_remote_code=True,
    )

    optimizer = bitsandbytes.optim.PagedAdamW32bit(
        model.parameters(), lr=config.learning_rate
    )
    if script_args.adafactor:
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, model.parameters()),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=config.learning_rate,
        )
    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer,
    )

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        # "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": 100_000,
    }
    output_min_length = 15
    output_max_length = (
        script_args.output_max_length
    )  # Todo: we use 64? [just a question]
    output_length_sampler = LengthSampler(output_min_length, output_max_length)
    for epoch in tqdm(range(script_args.epochs), "epoch: "):
        for batch in tqdm(ppo_trainer.dataloader):
            if epoch >= config.total_ppo_epochs:
                break

            question_tensors = batch[
                "input_ids"
            ]  # todo: the input should have padding, do we need to give attention mask?

            response_tensors = ppo_trainer.generate(
                question_tensors,
                return_prompt=False,
                length_sampler=output_length_sampler,
                **generation_kwargs,
            )
            response_tensors = list(response_tensors)
            batch["response"] = tokenizer.batch_decode(
                response_tensors, skip_special_tokens=True
            )

            # Compute reward score
            print("start calculating reward")

            rewards = score(
                batch["response"], batch["system_prompt"]
            )  # todo: query is the untokenized version of attack input (input_ids), the target should be system prompt rather than attack prompt (query)??

            print("rewards calculated")
            # Run PPO step
            stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

        if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
            ppo_trainer.save_pretrained(script_args.output_dir + f"epoch_{epoch}")
