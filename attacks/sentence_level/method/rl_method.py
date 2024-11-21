import logging

import gymnasium as gym
from sentence_transformers import SentenceTransformer
from stable_baselines3 import PPO
from typing import List

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

logger = logging.getLogger(__name__)


gym.register(
    id="prompt-env-v0",
    entry_point="attacks.sentence_level.method.env.prompt_env:PromptEnv",
)


def make_env(
    helper_model,
    target_model,
    prompts,
    dataset,
    embedder,
    env_id: str,
    rank: int,
    helper_client,
    target_client,
    seed: int = 0,
):
    """
    Utility function for multiprocessed env.

    :param target_client: the target model
    :param helper_client: the helper model
    :param dataset: the training dataset
    :param prompts: the prompts
    :param target_model: the target model
    :param helper_model: the helper model
    :param embedder: the embedder model
    :param env_id: the environment ID
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """

    def _init():
        env = gym.make(
            env_id,
            max_episode_steps=128,  # max steps for each episode
            # render_mode="human",
            helper_model=helper_model,
            target_model=target_model,
            prompts=prompts,
            train_dataset=dataset,
            embedder=embedder,
            helper_client=helper_client,
            target_client=target_client,
        )
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def create_model(model_path, server_url=None, api_key=None):
    if "gpt" in model_path:
        import openai

        client = openai.AsyncOpenAI()
    elif "claude" in model_path:
        import anthropic

        client = anthropic.AsyncAnthropic()
    elif server_url:
        import openai

        client = openai.AsyncOpenAI(
            base_url=server_url,
            api_key=api_key,
        )
    else:
        raise ValueError("Please specify a valid target model.")
    return client


class RLMethod:
    def __init__(
        self,
        helper_model,
        target_model,
        prompts: List[str],
        dataset,
        args,
        env_num=2,
    ):
        # use spawn method to avoid the shared memory issue
        # torch.multiprocessing.set_start_method('spawn')
        # create environment
        self.embedder = SentenceTransformer(
            "BAAI/bge-large-en-v1.5", device="cuda", trust_remote_code=True
        )
        self.target_client = create_model(
            target_model, args.target_server_url, args.target_api_key
        )
        self.helper_client = create_model(
            helper_model, args.helper_server_url, args.helper_api_key
        )
        self.env = DummyVecEnv(
            [
                make_env(
                    helper_model,
                    target_model,
                    prompts,
                    dataset,
                    self.embedder,
                    "prompt-env-v0",
                    rank=i,
                    seed=42,
                    helper_client=self.helper_client,
                    target_client=self.target_client,
                )
                for i in range(env_num)
            ]
        )
        # self.env = PromptEnv(helper_model, target_model, prompts)
        # create PPO model
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=2,
            learning_rate=3e-4,
            n_steps=2048,  # update every 2048 steps
            batch_size=64,
            n_epochs=20,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="./tensorboard_logs",
        )

    def train(self, total_timesteps: int = 300000):
        """train the model"""
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path: str):
        """save the model"""
        self.model.save(path)

    def load(self, path: str):
        """load the model"""
        self.model = PPO.load(path, env=self.env)

    def optimize_new_prompt(self, prompt: str) -> str:
        """Use the trained model to optimize a single prompt"""
        pass
        # self.env.current_prompt = prompt
        # obs = self.env.get_observation()
        # action, _states = self.model.predict(obs, deterministic=True)
        # modified_prompt = self.env.step(action)
        #
        # return modified_prompt
