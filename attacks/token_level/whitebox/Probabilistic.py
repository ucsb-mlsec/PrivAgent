import numpy as np
import warnings
import torch
import re
from tqdm import tqdm
import time
import random
import os
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torchmetrics.functional.pairwise import pairwise_cosine_similarity


class Probabilistic:
    def __init__(
        self,
        model="llama3",
        optim_token_length=12,
        learning_rate=3e-1,
        batch_size=10,
        initial_coeff=15,
        num_iters=200,
        max_seq_len=256,
        num_trigger=10,
        n_samples=10,
        lam_sim=0.0,
        vocabulary=None,
        log_dir=None,
    ):
        assert optim_token_length >= 0 and learning_rate > 0, "Invalid parameter!"

        model_to_name = {
            "llama": "meta-llama/Llama-2-7b-hf",
            "llama-70b": "meta-llama/Llama-2-70b-chat-hf",
            "llama-chat": "meta-llama/Llama-2-7b-chat-hf",
            "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
        }

        model_name = model_to_name.get(model, "default_model")
        self.model_name = model_name
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", quantization_config=quant_config
        ).eval()
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quant_config,
            output_hidden_states=True,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output_dir = (
            log_dir
            + "/lr_"
            + str(learning_rate)
            + "_lam_sim_"
            + str(lam_sim)
            + "_num_iters_"
            + str(num_iters)
            + "_token_length_"
            + str(optim_token_length)
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.output_dir = output_dir
        self.writer = SummaryWriter(log_dir=output_dir)

        self.vocabulary = vocabulary
        # Set vocabulary size based on model
        model_to_vocab = {
            "gptj": 50400,
            "opt": 50272,
            "falcon": 65024,
            "llama3": 128256,
        }
        self.vocab_size = model_to_vocab.get(model, 32000)

        if vocabulary != None:
            self.vocab_size = len(vocabulary)

        self.learning_rate = learning_rate
        self.optim_token_length = optim_token_length
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.num_trigger = num_trigger

        self.generated_trigger = []
        self.sentence_embeddings = None
        self.n_samples = n_samples
        self.lam_sim = lam_sim

        # Initialize embeddings with no gradient calculations
        with torch.no_grad():
            if vocabulary == None:
                self.embeddings = self.model.get_input_embeddings()(
                    torch.arange(0, self.vocab_size).to(self.device).long()
                )
                self.ref_embeddings = self.ref_model.get_input_embeddings()(
                    torch.arange(0, self.vocab_size).to(self.device).long()
                )
            else:
                self.token_id_mapping, vocabulary_id = self.map_custom_voc(vocabulary)
                self.embeddings = self.model.get_input_embeddings()(
                    torch.from_numpy(vocabulary_id).to(self.device).long()
                )
                self.ref_embeddings = self.ref_model.get_input_embeddings()(
                    torch.from_numpy(vocabulary_id).to(self.device).long()
                )

        self.embed_sz = self.embeddings.size(-1)

        # Initialize coefficients
        self.log_coeffs = torch.zeros(
            optim_token_length, self.vocab_size, dtype=torch.float16, device=self.device
        )
        indices = torch.arange(self.log_coeffs.size(0), device=self.device).long()
        idx = torch.randint(
            0, self.vocab_size, (optim_token_length,), device=self.device
        )
        self.log_coeffs[indices, idx] = initial_coeff
        self.log_coeffs.requires_grad = True

        self.optimizer = torch.optim.Adam(
            [self.log_coeffs], lr=self.learning_rate, eps=1e-3
        )

    def map_custom_voc(self, vocabulary):
        token_id_mapping = {}
        voc_id = []
        # get the id of each token in vocabulary
        for token in vocabulary:
            token_id = self.tokenizer.convert_tokens_to_ids([token])[0]
            token_id_mapping[token] = token_id
            voc_id.append(token_id)

        return token_id_mapping, np.array(voc_id)

    def make_target_chat(self, text):
        system_prompt = [{"role": "system", "content": text}]
        dummy_prompt = [
            {"role": "user", "content": ""}
        ]  # Conversation roles must alternate user/assistant/user/assistant/...
        assistant_prompt = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": text},
        ]

        system_id = self.tokenizer.apply_chat_template(
            system_prompt, return_tensors="pt"
        ).to(self.device)
        dummy_id = self.tokenizer.apply_chat_template(
            dummy_prompt, return_tensors="pt"
        ).to(self.device)
        assistant_id = self.tokenizer.apply_chat_template(
            assistant_prompt, return_tensors="pt"
        ).to(self.device)[:, dummy_id.shape[1] :]

        system_onehot = torch.nn.functional.one_hot(
            system_id.long(), num_classes=self.vocab_size
        ).to(torch.float16)
        dummy_onehot = torch.nn.functional.one_hot(
            dummy_id.long(), num_classes=self.vocab_size
        ).to(torch.float16)
        assistant_onehot = torch.nn.functional.one_hot(
            assistant_id.long(), num_classes=self.vocab_size
        ).to(torch.float16)

        system_embed = torch.matmul(
            system_onehot, self.embeddings.to(system_onehot.device)[None, :, :]
        )
        dummy_embed = torch.matmul(
            dummy_onehot, self.embeddings.to(dummy_onehot.device)[None, :, :]
        )
        assistant_embed = torch.matmul(
            assistant_onehot, self.embeddings.to(assistant_onehot.device)[None, :, :]
        )

        return (
            system_embed.squeeze(0),
            dummy_embed.squeeze(0),
            assistant_embed.squeeze(0),
            assistant_id,
        )

    def train(self, target):
        total_num = len(target)
        # Training phrase
        for iteration in tqdm(range(self.num_iters)):
            random.shuffle(target)
            for i in range(0, total_num, self.batch_size):
                start = time.time()
                b_sz = min(self.batch_size, total_num - i)

                inputs_embeds = torch.zeros(
                    (b_sz, self.max_seq_len, self.embed_sz),
                    dtype=torch.float16,
                    device=self.device,
                )
                labels = torch.zeros(
                    (b_sz, self.max_seq_len), dtype=torch.long, device=self.device
                )

                coeffs = torch.nn.functional.gumbel_softmax(
                    self.log_coeffs.unsqueeze(0).repeat(b_sz, 1, 1), hard=False
                )
                user_embeds = torch.matmul(
                    coeffs, self.embeddings.to(coeffs.device)[None, :, :]
                )

                for idx, j in enumerate(range(i, min(total_num, i + self.batch_size))):
                    system_embed, dummy_embed, assistant_embed, assistant_id = (
                        self.make_target_chat(target[j])
                    )
                    system_len, assistant_len = (
                        system_embed.size(0),
                        assistant_embed.size(0),
                    )

                    shift = system_len
                    inputs_embeds[idx, :shift, :] = system_embed
                    inputs_embeds[idx, shift : shift + 5, :] = dummy_embed[
                        0:5, :
                    ]  # first part of the user's template
                    inputs_embeds[
                        idx, shift + 5 : shift + self.optim_token_length + 5, :
                    ] = user_embeds[idx]
                    inputs_embeds[idx, shift + self.optim_token_length + 5, :] = (
                        dummy_embed[5, :]
                    )  # second part
                    shift += self.optim_token_length + dummy_embed.shape[0]
                    labels[idx, :shift] = -100
                    inputs_embeds[idx, shift : shift + assistant_len, :] = (
                        assistant_embed
                    )
                    labels[idx, shift : shift + assistant_len] = assistant_id
                    shift += assistant_len
                    labels[idx, shift:] = -100

                self.optimizer.zero_grad()
                target_loss = self.model(inputs_embeds=inputs_embeds, labels=labels)[0]

                if self.lam_sim > 0:
                    ref_embeds = (
                        coeffs @ self.ref_embeddings.to(coeffs.device)[None, :, :]
                    )
                    pred = self.ref_model(inputs_embeds=ref_embeds)
                    output = pred.hidden_states[-1]
                    output = output.mean(1)

                    if self.sentence_embeddings != None:
                        sample_size = min(len(self.sentence_embeddings), self.n_samples)
                        sample_indices = random.sample(
                            range(self.sentence_embeddings.shape[0]), k=sample_size
                        )
                        ref_sentence_embeddings = self.sentence_embeddings[
                            sample_indices, :
                        ]

                        sims_loss = pairwise_cosine_similarity(
                            output, ref_sentence_embeddings
                        ).mean()
                        ref_loss = self.lam_sim * sims_loss

                        self.sentence_embeddings = torch.cat(
                            [self.sentence_embeddings, output.detach()], dim=0
                        )

                    else:
                        self.sentence_embeddings = output.detach()
                        ref_loss = torch.Tensor([0]).cuda()
                else:
                    ref_loss = torch.Tensor([0]).cuda()

                total_loss = target_loss + ref_loss
                total_loss.backward()

                self.writer.add_scalar(
                    "Loss/train", total_loss.item(), iteration * total_num + i
                )
                self.writer.add_scalar(
                    "Target_loss/train", target_loss.item(), iteration * total_num + i
                )
                self.writer.add_scalar(
                    "Ref_loss/train", ref_loss.item(), iteration * total_num + i
                )

                if i % 5 == 0:
                    print(
                        "Iteration %d: loss = %.4f, target_loss = %.4f, ref_loss = %.4f, time=%.2f"
                        % (
                            iteration * total_num + i,
                            total_loss.item(),
                            target_loss.item(),
                            ref_loss,
                            time.time() - start,
                        )
                    )

                self.optimizer.step()

        # Testing phrase
        for _ in range(self.num_trigger):
            adv_ids = torch.nn.functional.gumbel_softmax(
                self.log_coeffs, hard=True
            ).argmax(1)
            adv_ids = adv_ids.cpu().detach().numpy()
            if self.vocabulary is not None:
                # Map generated indices back to tokens in the custom vocabulary
                adv_ids = [
                    self.token_id_mapping[self.vocabulary[idx]]
                    for idx in adv_ids.cpu().tolist()
                ]
                adv_ids = np.array(adv_ids)

            self.generated_trigger.append(adv_ids)

        print(self.generated_trigger)
        np.savez(
            self.output_dir + "/results.npz", trigger=np.array(self.generated_trigger)
        )
