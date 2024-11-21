import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from rewards.text_rewards import TextRewards


def local_chat(msg_lst, model, tok):
    inputs = tok.apply_chat_template(
        msg_lst,
        add_generation_prompt=True,
        padding=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    inputs_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    terminator = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]
    output = model.generate(
        input_ids=inputs_ids,
        attention_mask=attention_mask,
        max_new_tokens=64,
        eos_token_id=terminator,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )[:, inputs_ids.shape[1] :].cpu()
    resps = [
        tok.decode(output[i, :], skip_special_tokens=True)
        for i in range(output.shape[0])
    ]

    return resps


# meta-llama/Meta-Llama-3-8B-Instruct
def test_model(
    atk_path="./step_60",
    dfs_path="./methods/token_level/blackbox/runs/step_15_new",
    num=10,
    dataset=None,
):
    test_data = (
        dataset if dataset else pd.read_csv("./dataset/test_data_pleak.csv")["text"]
    )
    reward = np.zeros((num, len(test_data)))
    atk_model = AutoModelForCausalLM.from_pretrained(atk_path, device_map="auto")
    atk_tok = AutoTokenizer.from_pretrained(atk_path, padding_side="left")
    atk_tok.pad_token = atk_tok.eos_token

    message = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Please generate a prompt for me: "},
        ]
        for _ in range(num)
    ]
    prompts = local_chat(message, atk_model, atk_tok)
    del atk_model
    torch.cuda.empty_cache()

    dfs_model = AutoModelForCausalLM.from_pretrained(dfs_path, device_map="auto")
    dfs_tok = AutoTokenizer.from_pretrained(dfs_path, padding_side="left")
    dfs_tok.pad_token = dfs_tok.eos_token

    for i, prompt in enumerate(tqdm(prompts)):
        query = []
        for text in test_data:
            query.append(
                [
                    {"role": "system", "content": text},
                    {"role": "user", "content": prompt},
                ]
            )
        resps = local_chat(query, dfs_model, dfs_tok)
        reward[i, :] = TextRewards.distance_lcs(resps, test_data)

    print(f"average score: {np.mean(reward)}")
    print(f"max score: {np.mean(np.max(reward, axis=0))}")


test_model()
