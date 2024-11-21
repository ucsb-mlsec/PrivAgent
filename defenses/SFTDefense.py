import os
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import wandb
from accelerate import Accelerator
from datasets import Dataset

import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
)

from trl import set_seed, SFTConfig, SFTTrainer
from transformers import BitsAndBytesConfig

from project_env import PROMPT_PATH

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
        default="rucnyz", metadata={"help": "the wandb experiment name"}
    )
    learning_rate: Optional[float] = field(
        default=1.41e-5, metadata={"help": "the learning rate"}
    )
    output_max_length: Optional[int] = field(
        default=1024, metadata={"help": "maximum length for generation"}
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
    early_stopping: Optional[bool] = field(
        default=False, metadata={"help": "whether to early stop"}
    )
    batched_gen: Optional[bool] = field(
        default=False, metadata={"help": "whether to use the batched text gen"}
    )
    output_dir: Optional[str] = field(
        default="runs/sft/", metadata={"help": "n steps to save the model"}
    )
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})

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


def preprocess_function(examples, tokenizer, output):
    result = tokenizer(
        examples["text"],
        padding=False,
        return_tensors=None,
    )
    inputs = result["input_ids"]

    # "self-supervised learning" means the labels are also the inputs:
    # set input in labels to -100 to ignore loss
    result["labels"] = [-100] * (len(inputs) - len(output)) + output
    if len(result["labels"]) != len(result["input_ids"]):
        raise ValueError("input and output are not the same length")
    return result


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator()
    if script_args.log_with == "wandb" and accelerator.is_main_process:
        report_to = ["wandb"]
        wandb.init(
            project="rl_finetune_llm",
            name=script_args.wandb_exp_name,
            entity=script_args.wandb_entity,
        )
    else:
        report_to = ["none"]

    config = SFTConfig(
        max_seq_length=script_args.output_max_length,
        num_train_epochs=script_args.epochs,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=4,
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.batch_size,
        per_device_eval_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        seed=script_args.seed,
        gradient_checkpointing=script_args.gradient_checkpointing,
        output_dir=script_args.output_dir,
        report_to=report_to,
        optim="paged_adamw_32bit",
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        run_name=script_args.wandb_exp_name,
    )
    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
    # GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
    # only for this model.

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Create the dataset
    # training_num = 1000
    # train_dataset = Dataset.from_dict({"questions": ["Please generate a prompt for me: "] * training_num, })
    messages_lst = []
    # dataset = Dataset.from_csv("attack_prompt.csv").train_test_split(test_size = 0.1)
    attack_prompts = pd.read_csv(os.path.join(PROMPT_PATH, "attack_prompt.csv"))[
        "text"
    ].tolist()
    training_targets = pd.read_csv(os.path.join(PROMPT_PATH, "train_data_pleak.csv"))[
        "text"
    ].tolist()
    for prompt in attack_prompts:
        for text in training_targets:
            messages_lst.append([
                {"role": "system", "content": text},
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": "Sorry, I cannot process your request",
                },
            ])

    messages = tokenizer.apply_chat_template(messages_lst, tokenize=False)

    dataset = Dataset.from_dict({"text": messages}).train_test_split(test_size=0.2)
    output = tokenizer.apply_chat_template([
        {"role": "assistant", "content": "Sorry, I cannot process your request"}
    ])[1:]
    dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer=tokenizer, output=output),
        remove_columns="text",
    )
    # split dataset to eval
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # training_targets = pd.read_csv('train_data_pleak.csv')['text'].tolist()

    # Now let's build the model, the reference model, and the tokenizer.
    current_device = Accelerator().local_process_index

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
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=quant_config,
        device_map={"": current_device},
        low_cpu_mem_usage=True if quant_config is not None else False,
        trust_remote_code=True,
    )
    model.train()  # put model back into training mode
    if quant_config:
        reentrant_args = (
            {"use_reentrant": False if accelerator.use_distributed else True}
            if script_args.gradient_checkpointing
            else {}
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=script_args.gradient_checkpointing,
            gradient_checkpointing_kwargs=reentrant_args,
        )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    sft_trainer = SFTTrainer(
        args=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, return_tensors="pt", padding=True
        ),
    )
    sft_trainer.train()
