from typing import Union, Tuple, Dict, Any

import gym
import copy
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveRL

from skrl.agents.torch import Agent

TMIX = 0.5      # Fraction of training time to use advantage mixing

PPO_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch
    
    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages
    
    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor
    "value_loss_scale": 1.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": 250,      # TensorBoard writing interval (timesteps)

        "checkpoint_interval": 1000,        # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately
    }
}


class PPO_Mix(Agent):
    def __init__(self, 
                 models: Dict[str, Model], 
                 memory: Union[Memory, Tuple[Memory], None] = None, 
                 observation_space: Union[int, Tuple[int], gym.Space, None] = None, 
                 action_space: Union[int, Tuple[int], gym.Space, None] = None, 
                 device: Union[str, torch.device] = "cuda:0", 
                 cfg: dict = {}) -> None:
        """Proximal Policy Optimization (PPO)
        https://arxiv.org/abs/1707.06347
        
        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and 
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: None)
        :type observation_space: int, tuple or list of integers, gym.Space or None, optional
        :param action_space: Action space or shape (default: None)
        :type action_space: int, tuple or list of integers, gym.Space or None, optional
        :param device: Computing device (default: "cuda:0")
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict
        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)
        _cfg.update(cfg)
        super().__init__(models=models, 
                         memory=memory, 
                         observation_space=observation_space, 
                         action_space=action_space, 
                         device=device, 
                         cfg=_cfg)

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._clip_predicted_values = self.cfg["clip_predicted_values"]

        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]

        self._kl_threshold = self.cfg["kl_threshold"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]
        self._value_preprocessor = self.cfg["value_preprocessor"]

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None:
            if self.policy is self.value:
                self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
            else:
                self.optimizer = torch.optim.Adam(itertools.chain(self.policy.parameters(), self.value.parameters()), 
                                                  lr=self._learning_rate)
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"])

            self.checkpoint_modules["optimizer"] = self.optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(**self.cfg["value_preprocessor_kwargs"])
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

    def init(self) -> None:
        """Initialize the agent
        """
        super().init()
        self.set_mode("eval")
        
        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards1", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="rewards2", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="dones", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob1", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="log_prob2", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values1", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values2", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns1", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns2", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages1", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages2", size=1, dtype=torch.float32)

        self.tensors_names = ["states", "actions", "rewards1", "rewards2", "dones", "log_prob1", "log_prob2", "values1", "values2", "returns1", "returns2", "advantages1", "advantages2"]

        # create temporary variables needed for storage and computation
        self._current_log_prob1 = None
        self._current_log_prob2 = None
        self._current_next_states = None

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy
        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        :return: Actions
        :rtype: torch.Tensor
        """
        states = self._state_preprocessor(states)

        # sample random actions
        # TODO, check for stochasticity
        if timestep < self._random_timesteps:
            return self.policy.random_act(states, taken_actions=None, role="policy")

        # sample stochastic actions
        actions1, log_prob1, actions_mean1 = self.policy.act(states, taken_actions=None, role="policy1")
        self._current_log_prob1 = log_prob1
        actions2, log_prob2, actions_mean2 = self.policy.act(states, taken_actions=None, role="policy2")
        self._current_log_prob2 = log_prob2

        return torch.cat((actions1, actions2),axis=1), log_prob1 + log_prob2, torch.cat((actions_mean1, actions_mean2),axis=1)

    def record_transition(self, 
                          states: torch.Tensor, 
                          actions: torch.Tensor, 
                          rewards1: torch.Tensor, 
                          rewards2: torch.Tensor, 
                          next_states: torch.Tensor, 
                          dones: torch.Tensor, 
                          infos: Any, 
                          timestep: int, 
                          timesteps: int) -> None:
        """Record an environment transition in memory
        
        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param dones: Signals to indicate that episodes have ended
        :type dones: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        rewards = rewards1 + rewards2
        super().record_transition(states, actions, rewards, next_states, dones, infos, timestep, timesteps)

        # reward shaping
        if self._rewards_shaper is not None:
            rewards1 = self._rewards_shaper(rewards1, timestep, timesteps)
            rewards2 = self._rewards_shaper(rewards2, timestep, timesteps)

        self._current_next_states = next_states

        if self.memory is not None:
            with torch.no_grad():
                values1, _, _ = self.value.act(states=self._state_preprocessor(states), taken_actions=None, role="value1")
                values2, _, _ = self.value.act(states=self._state_preprocessor(states), taken_actions=None, role="value2")
            values1 = self._value_preprocessor(values1, inverse=True)
            values2 = self._value_preprocessor(values2, inverse=True)

            self.memory.add_samples(states=states, actions=actions, rewards1=rewards1, rewards2=rewards2, next_states=next_states, dones=dones, 
                                    log_prob1=self._current_log_prob1, log_prob2=self._current_log_prob2, values1=values1, values2=values2)
            for memory in self.secondary_memories:
                memory.add_samples(states=states, actions=actions, rewards1=rewards1, rewards2=rewards2, next_states=next_states, dones=dones, 
                                   log_prob1=self._current_log_prob1, log_prob2=self._current_log_prob2, values1=values1, values2=values2)

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        def compute_gae(rewards: torch.Tensor, 
                        dones: torch.Tensor, 
                        values: torch.Tensor, 
                        next_values: torch.Tensor, 
                        last_values: torch.Tensor, 
                        discount_factor: float = 0.99, 
                        lambda_coefficient: float = 0.95) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)
            :param rewards: Rewards obtained by the agent
            :type rewards: torch.Tensor
            :param dones: Signals to indicate that episodes have ended
            :type dones: torch.Tensor
            :param values: Values obtained by the agent
            :type values: torch.Tensor
            :param next_values: Next values obtained by the agent
            :type next_values: torch.Tensor
            :param discount_factor: Discount factor
            :type discount_factor: float
            :param lambda_coefficient: Lambda coefficient
            :type lambda_coefficient: float
            :return: Generalized Advantage Estimator
            :rtype: torch.Tensor
            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = rewards[i] - values[i] + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages
        
        tmix = timesteps*TMIX
        beta = min(timestep/tmix, 1)
        print("Beta: %.4f" % beta)
        print("Reward: %.4f" % (self.memory.get_tensor_by_name("rewards1") + self.memory.get_tensor_by_name("rewards2")).mean().item())
        print("\tLeg Reward: %.4f" % (self.memory.get_tensor_by_name("rewards1")).mean().item())
        print("\tArm Reward: %.4f" % (self.memory.get_tensor_by_name("rewards2")).mean().item())

        # compute returns and advantages
        with torch.no_grad():
            last_values1, _, _ = self.value.act(self._state_preprocessor(self._current_next_states.float()), taken_actions=None, role="value1")
            last_values2, _, _ = self.value.act(self._state_preprocessor(self._current_next_states.float()), taken_actions=None, role="value2")
        last_values1 = self._value_preprocessor(last_values1, inverse=True)
        last_values2 = self._value_preprocessor(last_values2, inverse=True)

        values1 = self.memory.get_tensor_by_name("values1")
        values2 = self.memory.get_tensor_by_name("values2")
        returns1, advantages1 = compute_gae(rewards=self.memory.get_tensor_by_name("rewards1"),
                                            dones=self.memory.get_tensor_by_name("dones"),
                                            values=values1,
                                            next_values=last_values1,
                                            last_values=last_values1,
                                            discount_factor=self._discount_factor,
                                            lambda_coefficient=self._lambda)
        returns2, advantages2 = compute_gae(rewards=self.memory.get_tensor_by_name("rewards2"),
                                            dones=self.memory.get_tensor_by_name("dones"),
                                            values=values2,
                                            next_values=last_values2,
                                            last_values=last_values2,
                                            discount_factor=self._discount_factor,
                                            lambda_coefficient=self._lambda)

        self.memory.set_tensor_by_name("values1", self._value_preprocessor(values1, train=True))
        self.memory.set_tensor_by_name("values2", self._value_preprocessor(values2, train=True))
        self.memory.set_tensor_by_name("returns1", self._value_preprocessor(returns1, train=True))
        self.memory.set_tensor_by_name("returns2", self._value_preprocessor(returns2, train=True))
        self.memory.set_tensor_by_name("advantages1", advantages1)
        self.memory.set_tensor_by_name("advantages2", advantages2)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self.tensors_names, mini_batches=self._mini_batches)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for sampled_states, sampled_actions, _, _, _, sampled_log_prob1, sampled_log_prob2, sampled_values1, sampled_values2, sampled_returns1, sampled_returns2, sampled_advantages1, sampled_advantages2 \
                in sampled_batches:

                sampled_states = self._state_preprocessor(sampled_states, train=not epoch)
                
                _, next_log_prob1, _ = self.policy.act(states=sampled_states, taken_actions=sampled_actions, role="policy1")
                _, next_log_prob2, _ = self.policy.act(states=sampled_states, taken_actions=sampled_actions, role="policy2")
                
                # Use combined action space for entropy calculation
                _, next_log_prob, _ = self.policy.act(states=sampled_states, taken_actions=sampled_actions, role="policy")

                # compute aproximate KL divergence
                with torch.no_grad():
                    ratio = next_log_prob - (sampled_log_prob1 + sampled_log_prob2)
                    kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                    kl_divergences.append(kl_divergence)

                # early stopping with KL divergence
                if self._kl_threshold and kl_divergence > self._kl_threshold:
                    break

                # compute entropy loss
                if self._entropy_loss_scale:
                    entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                else:
                    entropy_loss = 0
                
                # compute policy loss
                ratio1 = torch.exp(next_log_prob1 - sampled_log_prob1)
                ratio2 = torch.exp(next_log_prob2 - sampled_log_prob2)
                a1 = sampled_advantages1 + beta * sampled_advantages2
                a2 = sampled_advantages2 + beta * sampled_advantages1
                surrogate = a1 * ratio1 + a2 * ratio2
                surrogate_clipped = a1 * torch.clip(ratio1, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip) \
                                  + a2 * torch.clip(ratio2, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip)

                # Use the following code instead to turn off advantage mixing:
                # ratio = torch.exp(next_log_prob - (sampled_log_prob1 + sampled_log_prob2))
                # a = sampled_advantages1 + sampled_advantages2
                # surrogate = a * ratio
                # surrogate_clipped = a * torch.clip(ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip)
                
                policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                # compute value loss
                predicted_values1, _, _ = self.value.act(states=sampled_states, taken_actions=None, role="value1")
                predicted_values2, _, _ = self.value.act(states=sampled_states, taken_actions=None, role="value2")

                if self._clip_predicted_values:
                    predicted_values1 = sampled_values1 + torch.clip(predicted_values1 - sampled_values1, 
                                                                   min=-self._value_clip, 
                                                                   max=self._value_clip)
                    predicted_values2 = sampled_values2 + torch.clip(predicted_values2 - sampled_values2, 
                                                                   min=-self._value_clip, 
                                                                   max=self._value_clip)
                value_loss1 = self._value_loss_scale * F.mse_loss(sampled_returns1, predicted_values1)
                value_loss2 = self._value_loss_scale * F.mse_loss(sampled_returns2, predicted_values2)

                # optimization step
                self.optimizer.zero_grad()
                (policy_loss + entropy_loss + value_loss1 + value_loss2).backward()
                if self._grad_norm_clip > 0:
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(itertools.chain(self.policy.parameters(), self.value.parameters()), self._grad_norm_clip)
                self.optimizer.step()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += (value_loss1 + value_loss2).item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()
            
            # update learning rate
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveRL):                
                    self.scheduler.step(torch.tensor(kl_divergences).mean())
                else:
                    self.scheduler.step()

        # record data
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches))

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])