import ray

from ray.rllib.agents.qmix.qmix import QMixTrainer
from ray.tune import register_env
from ray.tune.logger import pretty_print
from env import RLlibStarCraft2Env
from gym.spaces import Tuple


def env_creator(smac_args):
        env = RLlibStarCraft2Env(**smac_args)
        agent_list = list(range(env._env.n_agents))
        grouping = {
            "group_1": agent_list,
        }
        obs_space = Tuple([env.observation_space for i in agent_list])
        act_space = Tuple([env.action_space for i in agent_list])
        return env.with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space)

ray.init()

# Register env with name
register_env("sc2_grouped", env_creator)

# Trainer config
config = {
        "num_workers": 3,
        "env_config": {
            "map_name": "8m",
        },
        "framework": "torch",
    }

# Make QMix Trainer
qmix_trainer = QMixTrainer(
    env="sc2_grouped",
    config=config
)

# Train 10 iterations
for i in range(10):
    result_qmix = qmix_trainer.train()
    print(pretty_print(result_qmix))

# Save Trainer
qmix_trainer.save(checkpoint_dir='./')

# Load Trainer
qmix_trainer.load_checkpoint(checkpoint_path='./checkpoint_10/checkpoint-10')

# Get RLlib Policy class
policy = qmix_trainer.get_policy()

# Get PyTorch nn.Module model
rnnmodel = policy.model
mixer = policy.mixer