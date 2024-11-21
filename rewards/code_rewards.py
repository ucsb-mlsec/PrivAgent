from rewards.codebleu.codebleu import LANG_ALIASES, calc_codebleu
from text_rewards import TextRewards


class CodeRewards(TextRewards):
    def __init__(self):
        pass

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

    @staticmethod
    def codeBLUE(preds, refs, language="python", weights=(0.25, 0.25, 0.25, 0.25)):
        """
        Embedding similarity: calculate the similarity between generated responses and reference responses in
        the embedding space of an embedding model
        :param weights: weights of the ngram_match, weighted_ngram_match, syntax_match, and dataflow_match respectively
        :param preds: a list of generated responses
        :param refs: a list of reference responses with the same size as preds
        :param language: the language of the responses
        :return: codebleu score
        """

        lang = LANG_ALIASES.get(language, language)
        return calc_codebleu(
            references=[refs],
            predictions=preds,
            lang=lang,
            weights=weights,
        )["codebleu"]

    @staticmethod
    def LLM_judge(prompts, llm, device):
        """
        LLM judge: evaluate the generated responses based on the LLM judge
        :param device: device to run the model
        :param llm: the LLM judge model
        :param prompts: a list of prompts
        :return: evaluation scores
        """

        scores = llm.evaluate(prompts, device)
        return scores
