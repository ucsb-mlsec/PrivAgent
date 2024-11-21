import asyncio
import os

import argparse
from typing import Any

import torch
import numpy as np
import pandas as pd
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from itertools import repeat, chain

from project_env import PROMPT_PATH
from rewards.text_rewards import TextRewards


async def repeat_chat_claude(
    chat, n, system_message, user_messages, model, temperature, top_p, max_tokens
):
    async def single_chat():
        return await chat(
            system=system_message,
            messages=user_messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    tasks = [single_chat() for _ in range(n)]
    results = await asyncio.gather(*tasks)
    return results


async def chat_function(
    chat, model, messages, n_samples, temperature=0.6, top_p=0.9, max_tokens=128
):
    for i in range(5):
        # sleep for a while to avoid rate limit
        try:
            if "claude" in model:
                # extract system message
                system_message = [
                    message["content"]
                    for message in messages
                    if message["role"] == "system"
                ][0]
                user_messages = [
                    message for message in messages if message["role"] != "system"
                ]
                ret = await repeat_chat_claude(
                    chat,
                    n_samples,
                    system_message,
                    user_messages,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
            else:
                ret = await chat(
                    n=n_samples,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
            return ret
        except Exception as e:
            print(f"failed with error {e}, retrying")
            await asyncio.sleep(10)
            continue
    return None


async def openai_batch_async_chat_completion(
    messages_lst: list[list[dict[str, str]]], client, model, limiter, n_samples
) -> tuple[Any]:
    tasks = [
        rate_limited_api_call_precise(
            limiter, messages, model, client.chat.completions.create, n_samples
        )
        for messages in messages_lst
    ]
    return await tqdm_asyncio.gather(*tasks, disable=args.disable_tqdm)


async def claude_batch_async_chat_completion(
    messages_lst: list[list[dict[str, str]]], client, model, limiter, n_samples
) -> tuple[Any]:
    tasks = [
        rate_limited_api_call_precise(
            limiter, messages, model, client.messages.create, n_samples
        )
        for messages in messages_lst
    ]
    return await tqdm_asyncio.gather(*tasks, disable=args.disable_tqdm)


async def rate_limited_api_call_precise(limiter, messages, model, llm_func, n_samples):
    async with limiter:
        return await chat_function(
            chat=llm_func,
            model=model,
            messages=messages,
            max_tokens=128,
            temperature=0.6,
            top_p=0.9,
            n_samples=n_samples,
        )


# Need to be refined later if we are to make this repo public
def test_model(prompts, model_name, metrics, client, n_samples, dataset=None):
    textReward = TextRewards()
    rewards = np.zeros((len(dataset), len(prompts), len(metrics)))
    multi_index = pd.MultiIndex.from_product(
        [dataset, prompts], names=["Dataset", "Prompt"]
    )
    df = pd.DataFrame(
        rewards.reshape(len(dataset) * len(prompts), len(metrics)),
        index=multi_index,
        columns=metrics,
    )

    dataset = (
        pd.read_csv(os.path.join(PROMPT_PATH, "test_data_pleak.csv"))["text"].tolist()
        if dataset is None
        else dataset
    )
    repeated_dataset = list(
        chain.from_iterable(repeat(item, n_samples) for item in dataset)
    )
    for i, prompt in enumerate(tqdm(prompts, disable=args.disable_tqdm)):
        messages_list = []
        for text in dataset:
            messages = [
                {"role": "system", "content": text},
                {"role": "user", "content": prompt},
            ]
            messages_list.append(messages)

        if "gpt" in model_name:
            output = asyncio.run(
                openai_batch_async_chat_completion(
                    messages_list,
                    client=client,
                    model=model_name,
                    limiter=limiter,
                    n_samples=n_samples,
                )
            )
            resp = [
                [
                    output_sentence.message.content
                    for output_sentence in data_out.choices
                ]
                for data_out in output
            ]
            resp = list(chain(*resp))
        elif "claude" in model_name:
            output = asyncio.run(
                claude_batch_async_chat_completion(
                    messages_list,
                    client=client,
                    model=model_name,
                    limiter=limiter,
                    n_samples=n_samples,
                )
            )
            resp = [
                [output_sentence.content[0].text for output_sentence in data_out]
                for data_out in output
            ]
            resp = list(chain(*resp))
        else:
            output = asyncio.run(
                openai_batch_async_chat_completion(
                    messages_list,
                    client=client,
                    model=model_name,
                    limiter=limiter,
                    n_samples=n_samples,
                )
            )
            resp = [
                [
                    output_sentence.message.content
                    for output_sentence in data_out.choices
                ]
                for data_out in output
            ]
            resp = list(chain(*resp))
            # torch.cuda.empty_cache()
        # resp = [tok.decode(sentence,
        # skip_special_tokens = True) for sentence in output]

        # print(f'for text {i}, prompt {j}, the response is: \n{resp}')
        lcs = textReward.distance_lcs(resp, repeated_dataset)
        df.loc[(slice(None), prompt), "lcs"] = [
            max(lcs[i : i + n_samples]) for i in range(0, len(lcs), n_samples)
        ]
        sim = textReward.embedding_similarity(resp, repeated_dataset, None, device)
        df.loc[(slice(None), prompt), "sim"] = [
            max(sim[i : i + n_samples]) for i in range(0, len(sim), n_samples)
        ]
        rouge = textReward.rouge(resp, repeated_dataset)
        df.loc[(slice(None), prompt), "rouge"] = [
            max(rouge[i : i + n_samples]) for i in range(0, len(rouge), n_samples)
        ]
        # for k, metric in enumerate(metrics):
        #     if metric == 'lcs':
        #         rewards[:, i, k] = textReward.distance_lcs(resp, dataset)
        #     elif metric == 'sim':
        #         rewards[:, i, k] = textReward.embedding_similarity(resp, dataset, None, device)
        #     elif metric == 'rouge':
        #         rewards[:, i, k] = textReward.rouge(resp, dataset)

    # for i in range(len(metrics)): print( f'the rewards in {metrics[i]}: avg = {np.mean(rewards[:, :, i])},
    # max = {np.mean(np.max(rewards[:, :, i], axis = 1))}')
    return df


if __name__ == "__main__":
    load_dotenv()
    argument = argparse.ArgumentParser()
    argument.add_argument(
        "--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    argument.add_argument("--prompts_data_path", type=str, required=True)
    argument.add_argument("--n_samples", type=int, default=5)
    argument.add_argument("--only_eval", action="store_true")
    argument.add_argument("--disable_tqdm", action="store_true")
    argument.add_argument("--server_url", type=str, default="")
    argument.add_argument("--api_key", type=str, default="")
    argument.add_argument("--dataset_path", type=str, default="test_data_pleak.csv")
    args = argument.parse_args()
    model_name = args.model_name
    assert args.prompts_data_path.endswith(".csv")
    short_model_name = model_name.split("/")[-1]
    output_path = args.prompts_data_path.replace(
        ".csv", f"_eval_top_{args.n_samples}_{short_model_name}.csv"
    )
    if args.only_eval and os.path.exists(output_path):
        print(f"loading from {output_path}")
        df = pd.read_csv(output_path).set_index(["Dataset", "Prompt"])
        max_values = df.groupby(level="Dataset").max().mean()
        print(max_values)
        exit(0)
    client = None
    if "gpt" in model_name:
        import openai

        client = openai.AsyncOpenAI()
        limiter = AsyncLimiter(100, 60)
    elif "claude" in model_name:
        import anthropic

        client = anthropic.AsyncAnthropic()
        limiter = AsyncLimiter(60, 60)
    else:
        assert args.server_url != "" and args.api_key != ""
        import openai

        client = openai.AsyncOpenAI(base_url=args.server_url, api_key=args.api_key)
        limiter = AsyncLimiter(100, 60)
    dataset = pd.read_csv(os.path.join(PROMPT_PATH, args.dataset_path))[
        "text"
    ].tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # A list of prompts is needed
    prompts_data = pd.read_csv(args.prompts_data_path)
    # select with the highest scores
    if "reward" in prompts_data.columns:
        prompts_data = prompts_data.sort_values(by="reward", ascending=False)[:5]
    prompts_data = (
        prompts_data.sample(5, random_state=42)
        if len(prompts_data) > 5
        else prompts_data
    )
    prompts = prompts_data["text"].tolist()

    df = test_model(
        prompts, model_name, ["lcs", "sim", "rouge"], client, args.n_samples, dataset
    )
    max_values = df.groupby(level="Dataset").mean().mean()
    df.to_csv(output_path)
    print(max_values)
