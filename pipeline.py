import argparse
import os
import pandas as pd
import time
from dotenv import load_dotenv

from attacks.sentence_level.method.rl_method import RLMethod
from project_env import PROMPT_PATH
from attacks.token_level.whitebox.GradientGuidedSearch import GradientGuidedSearch
from attacks.token_level.whitebox.Probabilistic import Probabilistic
from attacks.sentence_level.method.GeneticMethod import FuzzingMethod
from attacks.sentence_level.method.ReactMethod import ReActMethod


def train_model(method_name, train_set, seeds, helper_model, target_model, args):
    if method_name == "gcg":
        method = GradientGuidedSearch()
        method.train(train_set)
    elif method_name == "prob":
        method = Probabilistic()
        method.train(train_set)
    elif method_name == "fuzz":
        method = FuzzingMethod(
            seeds, train_set, helper_model=helper_model, target_model=target_model
        )
        method.train()
    elif method_name == "re":
        method = ReActMethod(
            seeds, train_set, helper_model=helper_model, target_model=target_model
        )
        method.train()
    elif method_name == "sent_rl":
        method = RLMethod(
            helper_model=helper_model,
            target_model=target_model,
            prompts=seeds,
            dataset=train_set,
            env_num=args.env_num,
            args=args,
        )
        method.train()


if __name__ == "__main__":
    argument = argparse.ArgumentParser()
    argument.add_argument(
        "--helper_model", type=str, default="gpt-4o-mini"
    )
    argument.add_argument("--helper_server_url", type=str, default=None)
    argument.add_argument("--helper_api_key", type=str, default=None)
    argument.add_argument("--target_server_url", type=str, default=None)
    argument.add_argument("--target_api_key", type=str, default=None)
    argument.add_argument(
        "--target_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    argument.add_argument(
        "--prompts_data_path", type=str, default="train_data_pleak.csv"
    )
    argument.add_argument("--method", type=str, default="fuzz")
    argument.add_argument("--env_num", type=int, default=2)
    args = argument.parse_args()
    # use dotenv to load the environment variables
    prompt_path = os.path.join(PROMPT_PATH, args.prompts_data_path)
    load_dotenv()
    train_set = pd.read_csv(str(prompt_path))["text"].tolist()
    print(len(train_set))
    seeds = pd.read_csv(os.path.join(PROMPT_PATH, "injection-prompt.csv"))[
        "text"
    ].tolist()
    start_time = time.time()
    train_model(
        args.method, train_set, seeds, args.helper_model, args.target_model, args
    )
    end_time = time.time()
    print(f"Current method uses {end_time - start_time} seconds")
