import numpy as np
import warnings
import torch
import re
from tqdm import tqdm
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


"""
    optim_token_length: numbers of tokens to be optimized first
    append_token_length: numbers of additional tokens to be appended and optimized
    num_trigger: numbers of triggers (prompts) to generate
    input_trigger: handcrafted initial seed whose length should be less than optim_token_length
    use_english_vocab: limiting generate tokens in English or not
    top_k: parameter for tradeoff between performance and efficiency
    temperature: parameter for diversity

    To generate prompts, run 'self.train()', and generated prompts are stored in self.generated_trigger
"""


class GradientGuidedSearch:
    def __init__(
        self,
        model="llama3",
        optim_token_length=12,
        append_token_length=0,
        num_trigger=5,
        input_trigger="",
        use_english_vocab=False,
        top_k=30,
        temperature=0.05,
        random=False,
    ):
        assert append_token_length >= 0 and temperature > 0, "Invalid parameter!"

        model_to_name = dict(
            zip(
                [
                    "gptj",
                    "opt",
                    "llama",
                    "llama-70b",
                    "llama-chat",
                    "llama3",
                    "falcon",
                    "vicuna",
                ],
                [
                    "EleutherAI/gpt-j-6b",
                    "facebook/opt-6.7B",
                    "meta-llama/Llama-2-7b-hf",
                    "meta-llama/Llama-2-70b-chat-hf",
                    "meta-llama/Llama-2-7b-chat-hf",
                    "meta-llama/Meta-Llama-3-8B-Instruct",
                    "tiiuae/falcon-7b",
                    "lmsys/vicuna-7b-v1.5",
                ],
            )
        )
        model_name = model_to_name.get(model)
        self.model_name = model
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", quantization_config=quant_config
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model == "llama":
            chat_template = r"""{% for message in messages %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + message['content'] + ' [/INST]' }}
                            {% elif message['role'] == 'system' %}\{{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}
                            {% elif message['role'] == 'assistant' %}{{ ' '  + message['content'] + ' ' + eos_token }}{% endif %}{% endfor %}"""
            self.tokenizer.chat_template = chat_template
        if model == "llama3":
            self.tokenizer.pad_token_id = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optim_token_length, self.append_token_length, self.input_trigger = (
            optim_token_length,
            append_token_length,
            input_trigger,
        )
        self.use_english_vocab, self.top_k, self.temperature, self.random = (
            use_english_vocab,
            top_k,
            temperature,
            random,
        )
        self.trigger, self.num_trigger, self.generated_trigger = None, num_trigger, []
        self.repetition_record = [
            {} for _ in range(optim_token_length + append_token_length)
        ]
        model_to_vocab = dict(
            zip(["gptj", "opt", "falcon", "llama3"], [50400, 50272, 65024, 128256])
        )
        self.vocab_size = model_to_vocab.get(model, 32000)

    def init_triggers(self, trigger_token_length):
        input_tokens = self.tokenizer.encode(self.input_trigger)
        input_len = len(input_tokens)
        if self.model_name in ["opt", "vicuna", "llama", "llama-chat", "llama-70b"]:
            input_tokens = input_tokens[1:]
            input_len -= 1
        if input_len > trigger_token_length:
            warnings.warn(
                "The initial token is too long and may lead to inadequate optimization",
                UserWarning,
            )
            return np.asarray(input_tokens)
        else:
            trigger = np.array(input_tokens, dtype=int)
            while input_len < trigger_token_length:
                t = np.random.randint(self.vocab_size)
                while (
                    re.search(r"[^a-zA-Z0-9s\s]", self.tokenizer.decode(t))
                    and self.use_english_vocab
                ):
                    t = np.random.randint(self.vocab_size)
                trigger = np.append(trigger, t)
                input_len += 1
            return trigger

    def get_triggers_grad(self):
        return self.model.model.embed_tokens.weight.grad[self.trigger]
        # for module in self.model.modules():
        #     if not isinstance(module, torch.nn.Embedding): continue
        #     if module.weight.shape[0] != self.vocab_size: continue
        #     return module.weight.grad[self.trigger]

    def make_target_chat(self, text, trigger):
        prompt = [
            {"role": "system", "content": text},
            {"role": "user", "content": self.tokenizer.decode(trigger)},
            {"role": "assistant", "content": text},
        ]
        prompt = self.tokenizer.apply_chat_template(prompt)
        non_label = [
            {"role": "system", "content": text},
            {"role": "user", "content": self.tokenizer.decode(trigger)},
        ]
        non_label = self.tokenizer.apply_chat_template(non_label)
        label = [-100] * len(non_label) + prompt[len(non_label) :]

        label = torch.tensor([label], device=self.device, dtype=torch.long)
        lm_input = torch.tensor([prompt], device=self.device, dtype=torch.long)
        return lm_input, label

    def compute_loss(self, target, trigger, require_grad=False):
        total_loss = 0
        for text in target:
            lm_input, label = self.make_target_chat(text, trigger)
            loss = self.model(lm_input, labels=label)[0] / len(target)
            if require_grad:
                loss.backward()
            total_loss += loss.item()
        return total_loss

    def hotflip_attack(self, grad):
        for module in self.model.modules():
            if not isinstance(module, torch.nn.Embedding):
                continue
            if module.weight.shape[0] != self.vocab_size:
                continue
            module.weight.requires_grad = True
            embedding_matrix = module.weight.detach()
            break
        averaged_grad = grad.unsqueeze(0)

        gradient_dot_embedding_matrix = -torch.einsum(
            "bij,kj->bik", (averaged_grad, embedding_matrix)
        )
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, self.top_k, dim=2)
        return best_k_ids.detach().squeeze().cpu().numpy()

    def replace_triggers(self, target, begin_idx=0):
        token_flipped = True
        while token_flipped:
            token_flipped = False
            with torch.set_grad_enabled(True):
                self.model.zero_grad()
                best_loss = self.compute_loss(target, self.trigger, True)

            candidates = self.hotflip_attack(self.get_triggers_grad())
            best_trigger = deepcopy(self.trigger)
            logits = np.zeros((len(self.trigger), self.top_k))
            for i in tqdm(range(len(self.trigger))):
                for j, cand in enumerate(candidates[i]):
                    if i < begin_idx:
                        logits[i][j] = -np.inf
                        continue
                    if (
                        re.search(r"[^a-zA-Z0-9s\s]", self.tokenizer.decode(cand))
                        and self.use_english_vocab
                    ):
                        logits[i][j] = -np.inf
                        continue
                    candidate_trigger = deepcopy(self.trigger)
                    candidate_trigger[i] = cand
                    self.model.zero_grad()
                    with torch.no_grad():
                        loss = self.compute_loss(
                            target, candidate_trigger, require_grad=False
                        )
                    logits[i][j] = best_loss - loss

            if not np.all(logits <= 0):
                prob = np.exp((logits - np.max(logits)) / self.temperature)
                for i in range(len(self.trigger) - begin_idx):
                    for j in range(self.top_k):
                        prob[i][j] /= (
                            self.repetition_record[i].get(candidates[i][j], 0) + 1
                        )
                prob = prob / prob.sum()
                if self.random:
                    prob_1d = prob.flatten()
                    idx_1d = np.random.choice(np.arange(prob_1d.size), p=prob_1d)
                    idx = np.unravel_index(idx_1d, prob.shape)
                else:
                    idx = np.unravel_index(np.argmax(prob), prob.shape)
                best_trigger[idx[0]] = candidates[idx[0]][idx[1]]
                best_loss -= logits[idx[0]][idx[1]]
                token_flipped = True
            self.trigger = deepcopy(best_trigger)
            print(
                f"Loss: {best_loss}, trigger:{self.tokenizer.decode(self.trigger, skip_special_tokens=True)}"
            )

    def train(self, target):
        tmp = self.input_trigger
        for _ in range(self.num_trigger):
            print(f"Generating the {len(self.generated_trigger) + 1}th trigger... ")
            self.trigger = self.init_triggers(self.optim_token_length)
            print(
                f"The {len(self.generated_trigger) + 1}th initial trigger is: {self.tokenizer.decode(self.trigger, skip_special_tokens=True)}"
            )
            self.replace_triggers(target)

            if self.append_token_length > 0:
                print(
                    f"The intermediate trigger is: {self.tokenizer.decode(self.trigger, skip_special_tokens=True)}"
                )
                begin_idx = len(self.trigger)
                self.input_trigger = self.tokenizer.decode(self.trigger)
                self.trigger = self.init_triggers(
                    self.optim_token_length + self.append_token_length
                )
                self.replace_triggers(target, begin_idx)
                self.input_trigger = tmp

            print(
                f"The {len(self.generated_trigger) + 1}th optimized trigger is: {self.tokenizer.decode(self.trigger, skip_special_tokens=True)}"
            )
            for i in range(len(self.trigger)):
                if self.trigger[i] in self.repetition_record[i]:
                    self.repetition_record[i][self.trigger[i]] += 1
                else:
                    self.repetition_record[i][self.trigger[i]] = 1
            self.generated_trigger.append(self.tokenizer.decode(self.trigger))
