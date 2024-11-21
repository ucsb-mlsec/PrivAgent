import asyncio
from typing import Any

import numpy as np
import openai
import anthropic
import torch
from aiolimiter import AsyncLimiter
import random
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from attacks.sentence_level.action_mutator.different_mutators import MutationHelper
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


async def openai_batch_async_chat_completion(
    messages_lst: list[list[dict[str, str]]], client, model, limiter
) -> tuple[Any]:
    tasks = [
        rate_limited_api_call_precise(
            limiter, messages, model, client.chat.completions.create
        )
        for messages in messages_lst
    ]
    return await tqdm_asyncio.gather(*tasks)


async def claude_batch_async_chat_completion(
    messages_lst: list[list[dict[str, str]]], client, model, limiter
) -> tuple[Any]:
    tasks = [
        rate_limited_api_call_precise(limiter, messages, model, client.messages.create)
        for messages in messages_lst
    ]
    return await tqdm_asyncio.gather(*tasks)


async def rate_limited_api_call_precise(limiter, messages, model, llm_func):
    async with limiter:
        return await chat_function(
            chat=llm_func,
            model=model,
            messages=messages,
            max_tokens=128,
            temperature=0.6,
            top_p=0.9,
        )


async def chat_function(
    chat, model, messages, temperature=0.6, top_p=0.9, max_tokens=128
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
                # import pdb; pdb.set_trace()
                ret = await chat(
                    system=system_message,
                    messages=user_messages,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
            else:
                ret = await chat(
                    n=1,
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


class MCTSNode:
    def __init__(self, prompt, reward, parent):
        self.prompt, self.parent, self.reward = prompt, parent, reward
        self.visits, self.children = 0, []

    def get_score(self, total_visits, const):
        return self.reward + const * np.sqrt(
            2 * np.log(total_visits) / (self.visits + 1)
        )

    def update(self, new_reward, update_path, bias):
        if self.parent is not None:
            self.reward = (self.visits * self.reward + new_reward - bias) / (
                self.visits + 1
            )
            self.visits += 1
            if update_path:
                self.parent.update(new_reward, update_path)


"""
    seeds: list of initial prompts injection prompts
    dataset: list of system prompts in the training set
    method: method used to select a seed from the pool
    budget: maximum number of queries to the target model
    prob: when using MCTS, the probability of choosing non-leaf nodes
    const: when using UCB or MCTS, the coefficient for balancing exploitation and exploration

    To generate prompts, run 'self.train()', and the generated prompts are stored as a tree with 'self.root' as the root node

    During testing and debugging, we may use FuzzingMethod(seeds, target, target_model='gpt-3.5-turbo', helper_model='gpt-3.5-turbo')
"""


class FuzzingMethod:
    def __init__(
        self,
        seeds,
        dataset,
        method="ucb",
        budget=100,
        prob=0.2,
        const=0.2,
        target_model="gpt-4o-mini",
        helper_model="gpt-4o-mini",
    ):
        assert method in ["random", "ucb", "mcts"], "Invalid method!"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_model, self.target_tok = self.get_model(target_model)
        self.helper_model, self.helper_tok = self.get_model(helper_model, True)
        (
            self.prob,
            self.const,
            self.method,
            self.budget,
            self.dataset,
            self.num_seeds,
        ) = prob, const, method, budget, dataset, len(seeds)
        self.helper = MutationHelper(self.helper_model, self.helper_tok)
        self.terminator = (
            [
                self.target_tok.eos_token_id,
                self.target_tok.convert_tokens_to_ids("<|eot_id|>"),
            ]
            if target_model == "llama3"
            else None
        )
        self.target_name = target_model

        self.root = MCTSNode(None, None, None)
        initial_reward = 0
        for seed in tqdm(seeds):
            current_reward = self.evaluate_seed(seed)
            initial_reward += current_reward
            node = MCTSNode(seed, current_reward, self.root)
            self.root.children.append(node)
        self.avg = initial_reward / len(seeds)
        print("Initialization completed, average rewards: ", self.avg)

    def get_model(self, model, helper=False):
        model_to_name = dict(
            zip(
                [
                    "gptj",
                    "opt",
                    "llama",
                    "llama-70b",
                    "llama-chat",
                    "llama3",
                    "llama3-80b",
                    "falcon",
                    "vicuna",
                    "mixtral",
                ],
                [
                    "EleutherAI/gpt-j-6b",
                    "facebook/opt-6.7B",
                    "meta-llama/Llama-2-7b-hf",
                    "meta-llama/Llama-2-70b-chat-hf",
                    "meta-llama/Llama-2-7b-chat-hf",
                    "meta-llama/Llama-3.1-8B-Instruct",
                    "meta-llama/Llama-3.1-70B-Instruct",
                    "tiiuae/falcon-7b",
                    "lmsys/vicuna-7b-v1.5",
                    'mistralai/Mistral-7B-Instruct-v0.2',
                ],
            )
        )
        model_name = model_to_name.get(model, None)

        if model_name:
            quant_config = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_compute_dtype = torch.float16,
                bnb_4bit_quant_type = "nf4",
                bnb_4bit_use_double_quant = True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", quantization_config=quant_config,
            ).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
            if model == "llama":
                chat_template = r"""{% for message in messages %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + message['content'] + ' [/INST]' }}
                                {% elif message['role'] == 'system' %}\{{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}
                                {% elif message['role'] == 'assistant' %}{{ ' '  + message['content'] + ' ' + eos_token }}{% endif %}{% endfor %}"""
                tokenizer.chat_template = chat_template
            tokenizer.pad_token_id = tokenizer.eos_token_id

        else:
            limiter = AsyncLimiter(30, 60)
            if "gpt" in model:
                if helper:
                    client = openai.OpenAI()
                else:
                    client = openai.AsyncOpenAI()
            elif "claude" in model:
                client = anthropic.AsyncAnthropic()
            model = {"api": client, "name": model, "limiter": limiter}
            tokenizer = None

        return model, tokenizer

    def select_seed(self, node, visits, idx=None, best_score=None, best_node=None):
        if self.method == "random":
            new_node = node if node == self.root else best_node
            new_idx = (
                np.random.randint(0, self.num_seeds) if node == self.root else idx - 1
            )
            if new_idx == 0:
                new_node = node
            elif new_idx > 0:
                for child in node.children:
                    new_node, new_idx = self.select_seed(
                        child, visits, new_idx, None, new_node
                    )
            return new_node, new_idx

        elif self.method == "ucb":
            if node == self.root:
                best_score, this_score = -np.inf, -np.inf
            else:
                this_score = node.get_score(visits, self.const)
            new_best_node = node if this_score > best_score else best_node
            new_score = max(best_score, this_score)
            for child in node.children:
                new_best_node, new_score = self.select_seed(
                    child, visits, None, new_score, new_best_node
                )
            return new_best_node, new_score

        else:
            while len(node.children) > 0:
                if np.random.rand() < self.prob:
                    break
                best_score, best_node = 0, node
                for child in node.children:
                    score = child.get_score(visits, self.const)
                    if score > best_score:
                        best_score = score
                        best_node = child
                node = best_node
            return node, None

    def evaluate_seed(self, prompt):
        resps = []
        messages = []
        random_data = random.sample(self.dataset, 20)
        for text in random_data:
            messages.append(
                [
                    {"role": "system", "content": text},
                    {"role": "user", "content": prompt},
                ]
            )
        if self.target_tok:
            inputs = self.target_tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.target_model.device)
            inputs_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            terminator = [
                self.target_tok.eos_token_id,
                self.target_tok.convert_tokens_to_ids("<|eot_id|>"),
            ]
            output = self.target_model.generate(
                input_ids=inputs_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                eos_token_id=terminator,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )[:, inputs_ids.shape[1] :].cpu()
            resps = [
                self.target_tok.decode(output[i, :], skip_special_tokens=True)
                for i in range(output.shape[0])
            ]
        else:
            client = self.target_model["api"]
            limiter = self.target_model["limiter"]
            if 'gpt' in self.target_name:
                resp = asyncio.run(
                    openai_batch_async_chat_completion(
                        messages, client=client, model=self.target_name, limiter=limiter
                    )
                )
                resps = [
                    output.choices[0].message.content if output else "" for output in resp
                ]
            elif 'claude' in self.target_name:
                resp = asyncio.run(
                    claude_batch_async_chat_completion(
                        messages, client=client, model=self.target_name, limiter=limiter
                    )
                )
                resps = [output.content[0].text if output else "" for output in resp]
        rewards = TextRewards.distance_lcs(resps, random_data)
        return sum(rewards) / len(rewards)

    def train(self):
        total_reward = 0
        for i in tqdm(range(self.budget)):
            seed, _ = self.select_seed(self.root, i + 1)
            action = np.random.choice(self.helper.actions)
            print(f"Seed: {seed.prompt}, action: {action}")
            new_seed = self.helper.execute(seed.prompt, action)
            print("New seed generated:", new_seed)
            reward = self.evaluate_seed(new_seed)
            print("Reward: ", reward)
            total_reward += reward
            seed.update(reward, (self.method == "mcts"), self.avg)
            if reward > self.avg:
                new_leaf = MCTSNode(new_seed, 0, seed)
                seed.children.append(new_leaf)
                self.num_seeds += 1
                print("The seed has been added to the pool.")

        print(
            f"Training finished, now we have {self.num_seeds} useful prompts in total."
        )
        print(f"Average reward of generated prompts is {total_reward / self.budget}.")
