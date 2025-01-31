import isaacgym

import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.envs.torch import load_isaacgym_env_preview4
from skrl.utils import set_seed

# Override skrl imports to use custom advantaged-mixed versions
from ppo_mix import PPO_Mix, PPO_DEFAULT_CONFIG
from sequential_mix import SequentialTrainerMix
from wrappers_mix import wrap_env


# set the seed for reproducibility
set_seed(42)


# Define the shared model (stochastic and deterministic models) for the agent using mixins.
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU())
        
        self.mean_layer = nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
        
        self.value_layer1 = nn.Linear(64, 1)
        self.value_layer2 = nn.Linear(64, 1)

    def act(self, states, taken_actions, role):
        if role == "value1" or role == "value2":
            return DeterministicMixin.act(self, states, taken_actions, role)
        elif role == "policy" or taken_actions is None:
            return GaussianMixin.act(self, states, taken_actions, role)
        elif role == "policy1":
            return GaussianMixin.act(self, states, taken_actions[:, :12], role)
        elif role == "policy2":
            return GaussianMixin.act(self, states, taken_actions[:, 12:], role)

    def compute(self, states, taken_actions, role):
        if role == "policy":
            return self.mean_layer(self.net(states)), self.log_std_parameter
        if role == "policy1":
            return self.mean_layer(self.net(states))[:, :12], self.log_std_parameter[:12]
        if role == "policy2":
            return self.mean_layer(self.net(states))[:, 12:], self.log_std_parameter[12:]
        elif role == "value1":
            return self.value_layer1(self.net(states))
        elif role == "value2":
            return self.value_layer2(self.net(states))


# Load and wrap the Isaac Gym environment
env = load_isaacgym_env_preview4(task_name="Pentapede")   # preview 3 and 4 use the same loader
env = wrap_env(env)

device = env.device


# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)


# Instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#spaces-and-models
models_ppo = {}
models_ppo["policy"] = Shared(env.observation_space, env.action_space, device)
models_ppo["value"] = models_ppo["policy"]  # same instance: shared model


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#configuration-and-hyperparameters
# cfg_ppo = PPO_DEFAULT_CONFIG.copy()
# cfg_ppo["random_timesteps"] = 0
# cfg_ppo["state_preprocessor"] = RunningStandardScaler
# cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# # logging to TensorBoard each 16 timesteps and ignore checkpoints
# cfg_ppo["experiment"]["write_interval"] = 16
# cfg_ppo["experiment"]["checkpoint_interval"] = 0
# cfg_ppo["value_preprocessor"] = RunningStandardScaler 
# cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device} 
cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["rollouts"] = 24  # memory_size
cfg_ppo["learning_epochs"] = 5
cfg_ppo["mini_batches"] = 6  # 24 * 4096 / 32768 but we use 24 * 2048 / 8192
cfg_ppo["discount_factor"] = 0.99
cfg_ppo["lambda"] = 0.95
cfg_ppo["learning_rate"] = 3e-4
cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg_ppo["random_timesteps"] = 0
cfg_ppo["learning_starts"] = 0
cfg_ppo["grad_norm_clip"] = 1.0
cfg_ppo["ratio_clip"] = 0.2
cfg_ppo["value_clip"] = 0.2
cfg_ppo["clip_predicted_values"] = True
cfg_ppo["entropy_loss_scale"] = 0.0
cfg_ppo["value_loss_scale"] = 1.0
cfg_ppo["kl_threshold"] = 0
cfg_ppo["rewards_shaper"] = None
cfg_ppo["state_preprocessor"] = RunningStandardScaler
cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg_ppo["value_preprocessor"] = RunningStandardScaler
cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints each 120 and 1200 timesteps respectively
cfg_ppo["experiment"]["write_interval"] = 120
cfg_ppo["experiment"]["checkpoint_interval"] = 1200

agent = PPO_Mix(models=models_ppo,
                memory=memory, 
                cfg=cfg_ppo, 
                observation_space=env.observation_space, 
                action_space=env.action_space,
                device=device)

agent.load('/home/frc-ag-3/harry_ws/courses/dlr/IsaacGymEnvs/isaacgymenvs/runs/22-12-14_13-59-11-335377_PPO_Mix/checkpoints/best_agent.pt')


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1600, "headless": True}
trainer = SequentialTrainerMix(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.eval()
