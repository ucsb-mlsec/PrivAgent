import evaluate
import numpy as np
import sacrebleu
import torch
from nltk.translate.bleu_score import sentence_bleu
from torch.nn import functional as F
import nltk
from nltk.tokenize import word_tokenize
from nltk.data import find
from sentence_transformers import SentenceTransformer


def calculate_selfbleu(sentences, n_gram=4):
    bleu_scores = []
    for i, sentence in enumerate(sentences):
        references = [word_tokenize(s) for j, s in enumerate(sentences) if j != i]
        candidate = word_tokenize(sentence)

        weights = [1.0 / n_gram] * n_gram
        bleu_score = sentence_bleu(references, candidate, weights=weights)
        bleu_scores.append(bleu_score)

    return np.mean(bleu_scores)


class TextRewards(object):
    def __init__(self):
        self.model = None

    @staticmethod
    def keyword_matching(preds, keywords):
        """
        Keyword matching: check if given keywords in a list of generated responses
        :param preds: a list of generated responses
        :param keywords: a list of keywords to be matched, e.g., ["Sorry", "I am sorry"]
        :return: matching rate, num_matched_responses
        """
        uppercased_test_prefixes = [word.upper() for word in keywords]
        num_matched_responses = 0
        num_responses = len(preds)
        if len(preds) == 0:
            return num_matched_responses, num_matched_responses
        for pred in preds:
            if any([prefix in pred for prefix in keywords + uppercased_test_prefixes]):
                num_matched_responses += 1
        return num_matched_responses / num_responses, num_matched_responses

    def embedding_similarity(
        self, preds, refs, embedding_model_func, device, sim_func="cosine"
    ):
        """
        Embedding similarity: calculate the similarity between generated responses and reference responses in
        the embedding space of an embedding model
        :param device: device to run the model
        :param preds: a list of generated responses
        :param refs: a list of reference responses with the same size as preds
        :param embedding_model_func: the call function of the embedding model, output should be n \times dim
        :param sim_func: similarity function, e.g., cosine, Euclidean
        :return: similarity score
        """
        if self.model is None:
            self.model = SentenceTransformer(
                "BAAI/bge-large-en-v1.5", device=device
            ).eval()

        def defalut_embedding_model_func(texts):
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            return embeddings

        if embedding_model_func is None:
            embedding_model_func = defalut_embedding_model_func
        pred_embeddings = embedding_model_func(preds)
        ref_embeddings = embedding_model_func(refs)
        if pred_embeddings.shape != ref_embeddings.shape:
            raise ValueError(
                "The shape of the response and reference embeddings should be the same."
            )
        if sim_func == "cosine":
            similarity = F.cosine_similarity(pred_embeddings, ref_embeddings, dim=1)
        elif sim_func == "euclidean":
            similarity = torch.cdist(pred_embeddings, ref_embeddings, p=2)
        else:
            raise ValueError("Unsupported similarity function")

        return similarity.cpu().numpy()

    @staticmethod
    def distance_lcs(preds, refs):
        """
        Editing distance: calculate the editing needed to change the generated responses to reference responses
        :param preds: a list of generated responses
        :param refs: a list of reference responses with the same size as preds
        :return: normalized score
        """

        try:
            find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        def sigmoid(x, k=5, x0=0.6):
            # want to have a large diff when x is close to 0.6
            return 1 / (1 + np.exp(-k * (x - x0)))

        # def substring_match(text1, text2):
        #     return text2 in text1
        #
        # def lcs_length(s1, s2):
        #     m, n = len(s1), len(s2)
        #     dp = np.zeros((m + 1, n + 1), dtype=int)
        #     for i in range(1, m + 1):
        #         for j in range(1, n + 1):
        #             if s1[i - 1] == s2[j - 1]:
        #                 dp[i][j] = dp[i - 1][j - 1] + 1
        #             else:
        #                 dp[i][j] = np.max([dp[i - 1][j], dp[i][j - 1]])
        #     return dp[m][n]

        def word_levenshtein_distance(words1, words2):
            """
            Edit distance between two lists of words
            """
            len1, len2 = len(words1), len(words2)
            dp = np.zeros((len1 + 1, len2 + 1), dtype=float)

            for i in range(len1 + 1):
                dp[i][0] = i
            for j in range(len2 + 1):
                dp[0][j] = j

            for i in range(1, len2 + 1):
                for j in range(1, len1 + 1):
                    if words1[j - 1] == words2[i - 1]:
                        dp[j][i] = dp[j - 1][i - 1]
                    else:
                        dp[j][i] = np.min(
                            [
                                dp[j - 1][i] + 1,
                                dp[j][i - 1] + 1,
                                dp[j - 1][i - 1] + 1,
                            ]
                        )

            # for i in range(1, len1 + 1):
            #     dp[i][len2] /= i
            if len1 < len2:
                return (dp[len1, len2]) / len2
            else:
                return (np.min(dp[len2:, len2])) / len2

        scores = []
        for pred, ref in zip(preds, refs):
            response_words, instruction_words = word_tokenize(pred), word_tokenize(ref)
            len_response, len_instruction = len(response_words), len(instruction_words)
            if len_response < len_instruction:
                max_score = np.log(
                    1 / word_levenshtein_distance(response_words, instruction_words)
                )
            else:
                max_score = 0
                for start in range(0, len_response - len_instruction + 1):
                    edit_dis = word_levenshtein_distance(
                        response_words[start : (start + len_instruction)],
                        instruction_words,
                    )
                    if edit_dis == 0:
                        max_score = np.inf
                        break
                    else:
                        new_score = np.log(1 / edit_dis)
                        if new_score > max_score:
                            max_score = new_score
            scores.append(sigmoid(max_score))
        return scores

    @staticmethod
    def rouge(preds, refs, option="rouge1"):
        """
        Calculate the ROUGE score between generated responses and reference responses
        :param preds: a list of generated responses
        :param refs: a list of reference responses with the same size as preds
        :param option: rouge option, e.g., rouge1 (unigram word matching), rouge2, rougeL (longest common subsequence)
        :return: rouge scores
        """
        rouge = evaluate.load("rouge")
        rouge_scores = []

        for pred, ref in zip(preds, refs):
            if option == "rouge1":
                rouge_scores.append(
                    rouge.compute(predictions=[pred], references=[ref])["rouge1"]
                )
            elif option == "rouge2":
                rouge_scores.append(
                    rouge.compute(predictions=[pred], references=[ref])["rouge2"]
                )
            elif option == "rougeL":
                rouge_scores.append(
                    rouge.compute(predictions=[pred], references=[ref])["rougeL"]
                )
            else:
                raise ValueError("Unsupported option")

        return rouge_scores

    @staticmethod
    def LLM_judge(prompts, llm, device):
        """
        LLM judge: evaluate the generated responses based on the LLM judge
        :param prompts: a list of prompts
        :param device: device to run the model
        :param llm: the LLM model to evaluate the responses
        :return: evaluation scores
        """

        scores = llm.evaluate(prompts, device)
        return scores

    @staticmethod
    def BLEU(preds: list[str], refs: list[list[str]]):
        return (
            sacrebleu.corpus_bleu(
                preds,
                refs,
                smooth_method="exp",
                force=False,
                lowercase=False,
                use_effective_order=False,
            ).score
            / 100
        )

    @staticmethod
    def SELFBLEU(preds, max_n=4):
        total_selfbleu = 0
        for n in range(1, max_n + 1):
            selfbleu_n = calculate_selfbleu(preds, n_gram=n)
            total_selfbleu += selfbleu_n

        # BSelfBLEU = - sum(SelfBLEU)
        return total_selfbleu
