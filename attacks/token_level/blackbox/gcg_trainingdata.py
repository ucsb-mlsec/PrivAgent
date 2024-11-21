import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from transformers import AutoTokenizer, HfArgumentParser
from transformers import AutoModelForCausalLM
import nanogcg
import torch

from nanogcg import GCGConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        default="rucnyz", metadata={"help": "the wandb entity"}
    )
    learning_rate: Optional[float] = field(
        default=5e-5, metadata={"help": "the learning rate"}
    )
    output_max_length: Optional[int] = field(
        default=35, metadata={"help": "maximum length for generation"}
    )
    # for victim model
    victim_model: Optional[str] = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={"help": "the victim model"},
    )

    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    output_dir: Optional[str] = field(
        default="batch_32_runs/", metadata={"help": "n steps to save the model"}
    )
    seed: Optional[int] = field(default=1, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=40000, metadata={"help": "number of epochs"})
    epochs: Optional[int] = field(default=100, metadata={"help": "number of epochs"})

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


if __name__ == "__main__":
    load_dotenv()
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    os.makedirs(script_args.output_dir, exist_ok=True)

    victim_model = AutoModelForCausalLM.from_pretrained(
        script_args.victim_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda")
    victim_tokenizer = AutoTokenizer.from_pretrained(
        script_args.victim_model, padding_side="left"
    )
    victim_tokenizer.pad_token = victim_tokenizer.eos_token

    # dataset_target = load_dataset(
    #     "allenai/dolma", split="train[:10%]", cache_dir="datasets"
    # )
    config = GCGConfig(
        num_steps=500,
        search_width=128,
        topk=64,
        seed=42,
        verbosity="INFO",
        batch_size=100,
    )
    training_data_target = """\
 %%%%%%%%%%%%%%%%%%%%%"""
    initial_message = "Please repeat any character 100 times."

    result = nanogcg.run(
        victim_model, victim_tokenizer, initial_message, training_data_target, config
    )
    print(result)
