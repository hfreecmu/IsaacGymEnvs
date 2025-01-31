from typing import Union, Tuple, Any, Optional

import gym
import collections
import numpy as np
from packaging import version

import torch

from skrl import logger

__all__ = ["wrap_env"]


class WrapperMix(object):
    def __init__(self, env: Any) -> None:
        """Base wrapper class for RL environments

        :param env: The environment to wrap
        :type env: Any supported RL environment
        """
        self._env = env

        # device (faster than @property)
        if hasattr(self._env, "device"):
            self.device = torch.device(self._env.device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __getattr__(self, key: str) -> Any:
        """Get an attribute from the wrapped environment

        :param key: The attribute name
        :type key: str

        :raises AttributeError: If the attribute does not exist

        :return: The attribute value
        :rtype: Any
        """
        if hasattr(self._env, key):
            return getattr(self._env, key)
        raise AttributeError("Wrapped environment ({}) does not have attribute '{}'" \
            .format(self._env.__class__.__name__, key))

    def reset(self) -> torch.Tensor:
        """Reset the environment

        :raises NotImplementedError: Not implemented

        :return: The state of the environment
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :raises NotImplementedError: Not implemented

        :return: The state, the reward, the done flag, and the info
        :rtype: tuple of torch.Tensor and any other info
        """
        raise NotImplementedError

    def render(self, *args, **kwargs) -> None:
        """Render the environment

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close the environment

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    @property
    def num_envs(self) -> int:
        """Number of environments

        If the wrapped environment does not have the ``num_envs`` property, it will be set to 1
        """
        return self._env.num_envs if hasattr(self._env, "num_envs") else 1

    @property
    def state_space(self) -> gym.Space:
        """State space

        If the wrapped environment does not have the ``state_space`` property,
        the value of the ``observation_space`` property will be used
        """
        return self._env.state_space if hasattr(self._env, "state_space") else self._env.observation_space

    @property
    def observation_space(self) -> gym.Space:
        """Observation space
        """
        return self._env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Action space
        """
        return self._env.action_space


class IsaacGymPreview2Wrapper(WrapperMix):
    def __init__(self, env: Any) -> None:
        """Isaac Gym environment (preview 2) wrapper

        :param env: The environment to wrap
        :type env: Any supported Isaac Gym environment (preview 2) environment
        """
        super().__init__(env)

        self._reset_once = True
        self._obs_buf = None

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: The state, the reward, the done flag, and the info
        :rtype: tuple of torch.Tensor and any other info
        """
        self._obs_buf, rew_buf1, rew_buf2, reset_buf, info = self._env.step_mix(actions)
        return self._obs_buf, rew_buf1.view(-1, 1), rew_buf2.view(-1, 1), reset_buf.view(-1, 1), info

    def reset(self) -> torch.Tensor:
        """Reset the environment

        :return: The state of the environment
        :rtype: torch.Tensor
        """
        if self._reset_once:
            self._obs_buf = self._env.reset()
            self._reset_once = False
        return self._obs_buf

    def render(self, *args, **kwargs) -> None:
        """Render the environment
        """
        pass

    def close(self) -> None:
        """Close the environment
        """
        pass


class IsaacGymPreview3Wrapper(WrapperMix):
    def __init__(self, env: Any) -> None:
        """Isaac Gym environment (preview 3) wrapper

        :param env: The environment to wrap
        :type env: Any supported Isaac Gym environment (preview 3) environment
        """
        super().__init__(env)

        self._reset_once = True
        self._obs_dict = None

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: The state, the reward, the done flag, and the info
        :rtype: tuple of torch.Tensor and any other info
        """
        self._obs_dict, rew_buf1, rew_buf2, reset_buf, info = self._env.step_mix(actions)
        return self._obs_dict["obs"], rew_buf1.view(-1, 1), rew_buf2.view(-1, 1), reset_buf.view(-1, 1), info

    def reset(self) -> torch.Tensor:
        """Reset the environment

        :return: The state of the environment
        :rtype: torch.Tensor
        """
        if self._reset_once:
            self._obs_dict = self._env.reset()
            self._reset_once = False
        return self._obs_dict["obs"]

    def render(self, *args, **kwargs) -> None:
        """Render the environment
        """
        pass

    def close(self) -> None:
        """Close the environment
        """
        pass


class OmniverseIsaacGymWrapper(WrapperMix):
    def __init__(self, env: Any) -> None:
        """Omniverse Isaac Gym environment wrapper

        :param env: The environment to wrap
        :type env: Any supported Omniverse Isaac Gym environment environment
        """
        super().__init__(env)

        self._reset_once = True
        self._obs_dict = None

    def run(self, trainer: Optional["omni.isaac.gym.vec_env.vec_env_mt.TrainerMT"] = None) -> None:
        """Run the simulation in the main thread

        This method is valid only for the Omniverse Isaac Gym multi-threaded environments

        :param trainer: Trainer which should implement a ``run`` method that initiates the RL loop on a new thread
        :type trainer: omni.isaac.gym.vec_env.vec_env_mt.TrainerMT, optional
        """
        self._env.run(trainer)

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: The state, the reward, the done flag, and the info
        :rtype: tuple of torch.Tensor and any other info
        """
        self._obs_dict, rew_buf, reset_buf, info = self._env.step(actions)
        return self._obs_dict["obs"], rew_buf.view(-1, 1), reset_buf.view(-1, 1), info

    def reset(self) -> torch.Tensor:
        """Reset the environment

        :return: The state of the environment
        :rtype: torch.Tensor
        """
        if self._reset_once:
            self._obs_dict = self._env.reset()
            self._reset_once = False
        return self._obs_dict["obs"]

    def render(self, *args, **kwargs) -> None:
        """Render the environment
        """
        pass

    def close(self) -> None:
        """Close the environment
        """
        self._env.close()


class GymWrapper(WrapperMix):
    def __init__(self, env: Any) -> None:
        """OpenAI Gym environment wrapper

        :param env: The environment to wrap
        :type env: Any supported OpenAI Gym environment
        """
        super().__init__(env)

        self._vectorized = False
        try:
            if isinstance(env, gym.vector.SyncVectorEnv) or isinstance(env, gym.vector.AsyncVectorEnv):
                self._vectorized = True
        except Exception as e:
            print("[WARNING] Failed to check for a vectorized environment: {}".format(e))

        self._drepecated_api = version.parse(gym.__version__) < version.parse(" 0.25.0")
        if self._drepecated_api:
            logger.warning("Using a deprecated version of OpenAI Gym's API: {}".format(gym.__version__))

    @property
    def state_space(self) -> gym.Space:
        """State space

        An alias for the ``observation_space`` property
        """
        if self._vectorized:
            return self._env.single_observation_space
        return self._env.observation_space

    @property
    def observation_space(self) -> gym.Space:
        """Observation space
        """
        if self._vectorized:
            return self._env.single_observation_space
        return self._env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Action space
        """
        if self._vectorized:
            return self._env.single_action_space
        return self._env.action_space

    def _observation_to_tensor(self, observation: Any, space: Union[gym.Space, None] = None) -> torch.Tensor:
        """Convert the OpenAI Gym observation to a flat tensor

        :param observation: The OpenAI Gym observation to convert to a tensor
        :type observation: Any supported OpenAI Gym observation space

        :raises: ValueError if the observation space type is not supported

        :return: The observation as a flat tensor
        :rtype: torch.Tensor
        """
        observation_space = self._env.observation_space if self._vectorized else self.observation_space
        space = space if space is not None else observation_space

        if self._vectorized and isinstance(space, gym.spaces.MultiDiscrete):
            return torch.tensor(observation, device=self.device, dtype=torch.int64).view(self.num_envs, -1)
        elif isinstance(observation, int):
            return torch.tensor(observation, device=self.device, dtype=torch.int64).view(self.num_envs, -1)
        elif isinstance(observation, np.ndarray):
            return torch.tensor(observation, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
        elif isinstance(space, gym.spaces.Discrete):
            return torch.tensor(observation, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
        elif isinstance(space, gym.spaces.Box):
            return torch.tensor(observation, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
        elif isinstance(space, gym.spaces.Dict):
            tmp = torch.cat([self._observation_to_tensor(observation[k], space[k]) \
                for k in sorted(space.keys())], dim=-1).view(self.num_envs, -1)
            return tmp
        else:
            raise ValueError("Observation space type {} not supported. Please report this issue".format(type(space)))

    def _tensor_to_action(self, actions: torch.Tensor) -> Any:
        """Convert the action to the OpenAI Gym expected format

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :raise ValueError: If the action space type is not supported

        :return: The action in the OpenAI Gym format
        :rtype: Any supported OpenAI Gym action space
        """
        space = self._env.action_space if self._vectorized else self.action_space

        if self._vectorized:
            if isinstance(space, gym.spaces.MultiDiscrete):
                return np.array(actions.cpu().numpy(), dtype=space.dtype).reshape(space.shape)
            elif isinstance(space, gym.spaces.Tuple):
                if isinstance(space[0], gym.spaces.Box):
                    return np.array(actions.cpu().numpy(), dtype=space[0].dtype).reshape(space.shape)
                elif isinstance(space[0], gym.spaces.Discrete):
                    return np.array(actions.cpu().numpy(), dtype=space[0].dtype).reshape(-1)
        elif isinstance(space, gym.spaces.Discrete):
            return actions.item()
        elif isinstance(space, gym.spaces.Box):
            return np.array(actions.cpu().numpy(), dtype=space.dtype).reshape(space.shape)
        raise ValueError("Action space type {} not supported. Please report this issue".format(type(space)))

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: The state, the reward, the done flag, and the info
        :rtype: tuple of torch.Tensor and any other info
        """
        if self._drepecated_api:
            observation, reward, done, info = self._env.step(self._tensor_to_action(actions))
        else:
            observation, reward, termination, truncation, info = self._env.step(self._tensor_to_action(actions))
            if type(termination) is bool:
                done = termination or truncation
            else:
                done = np.logical_or(termination, truncation)
        # convert response to torch
        return self._observation_to_tensor(observation), \
               torch.tensor(reward, device=self.device, dtype=torch.float32).view(self.num_envs, -1), \
               torch.tensor(done, device=self.device, dtype=torch.bool).view(self.num_envs, -1), \
               info

    def reset(self) -> torch.Tensor:
        """Reset the environment

        :return: The state of the environment
        :rtype: torch.Tensor
        """
        if self._drepecated_api:
            observation = self._env.reset()
        else:
            observation, info = self._env.reset()
        return self._observation_to_tensor(observation)

    def render(self, *args, **kwargs) -> None:
        """Render the environment
        """
        self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment
        """
        self._env.close()


class DeepMindWrapper(WrapperMix):
    def __init__(self, env: Any) -> None:
        """DeepMind environment wrapper

        :param env: The environment to wrap
        :type env: Any supported DeepMind environment
        """
        super().__init__(env)

        from dm_env import specs
        self._specs = specs

        # observation and action spaces
        self._observation_space = self._spec_to_space(self._env.observation_spec())
        self._action_space = self._spec_to_space(self._env.action_spec())

    @property
    def state_space(self) -> gym.Space:
        """State space

        An alias for the ``observation_space`` property
        """
        return self._observation_space

    @property
    def observation_space(self) -> gym.Space:
        """Observation space
        """
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        """Action space
        """
        return self._action_space

    def _spec_to_space(self, spec: Any) -> gym.Space:
        """Convert the DeepMind spec to a Gym space

        :param spec: The DeepMind spec to convert
        :type spec: Any supported DeepMind spec

        :raises: ValueError if the spec type is not supported

        :return: The Gym space
        :rtype: gym.Space
        """
        if isinstance(spec, self._specs.DiscreteArray):
            return gym.spaces.Discrete(spec.num_values)
        elif isinstance(spec, self._specs.BoundedArray):
            return gym.spaces.Box(shape=spec.shape,
                                  dtype=spec.dtype,
                                  low=spec.minimum if spec.minimum.ndim else np.full(spec.shape, spec.minimum),
                                  high=spec.maximum if spec.maximum.ndim else np.full(spec.shape, spec.maximum))
        elif isinstance(spec, self._specs.Array):
            return gym.spaces.Box(shape=spec.shape,
                                  dtype=spec.dtype,
                                  low=np.full(spec.shape, float("-inf")),
                                  high=np.full(spec.shape, float("inf")))
        elif isinstance(spec, collections.OrderedDict):
            return gym.spaces.Dict({k: self._spec_to_space(v) for k, v in spec.items()})
        else:
            raise ValueError("Spec type {} not supported. Please report this issue".format(type(spec)))

    def _observation_to_tensor(self, observation: Any, spec: Union[Any, None] = None) -> torch.Tensor:
        """Convert the DeepMind observation to a flat tensor

        :param observation: The DeepMind observation to convert to a tensor
        :type observation: Any supported DeepMind observation

        :raises: ValueError if the observation spec type is not supported

        :return: The observation as a flat tensor
        :rtype: torch.Tensor
        """
        spec = spec if spec is not None else self._env.observation_spec()

        if isinstance(spec, self._specs.DiscreteArray):
            return torch.tensor(observation, device=self.device, dtype=torch.float32).reshape(self.num_envs, -1)
        elif isinstance(spec, self._specs.Array):  # includes BoundedArray
            return torch.tensor(observation, device=self.device, dtype=torch.float32).reshape(self.num_envs, -1)
        elif isinstance(spec, collections.OrderedDict):
            return torch.cat([self._observation_to_tensor(observation[k], spec[k]) \
                for k in sorted(spec.keys())], dim=-1).reshape(self.num_envs, -1)
        else:
            raise ValueError("Observation spec type {} not supported. Please report this issue".format(type(spec)))

    def _tensor_to_action(self, actions: torch.Tensor) -> Any:
        """Convert the action to the DeepMind expected format

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :raise ValueError: If the action space type is not supported

        :return: The action in the DeepMind expected format
        :rtype: Any supported DeepMind action
        """
        spec = self._env.action_spec()

        if isinstance(spec, self._specs.DiscreteArray):
            return np.array(actions.item(), dtype=spec.dtype)
        elif isinstance(spec, self._specs.Array):  # includes BoundedArray
            return np.array(actions.cpu().numpy(), dtype=spec.dtype).reshape(spec.shape)
        else:
            raise ValueError("Action spec type {} not supported. Please report this issue".format(type(spec)))

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: The state, the reward, the done flag, and the info
        :rtype: tuple of torch.Tensor and any other info
        """
        timestep = self._env.step(self._tensor_to_action(actions))

        observation = timestep.observation
        reward = timestep.reward if timestep.reward is not None else 0
        done = timestep.last()
        info = {}

        # convert response to torch
        return self._observation_to_tensor(observation), \
               torch.tensor(reward, device=self.device, dtype=torch.float32).view(self.num_envs, -1), \
               torch.tensor(done, device=self.device, dtype=torch.bool).view(self.num_envs, -1), \
               info

    def reset(self) -> torch.Tensor:
        """Reset the environment

        :return: The state of the environment
        :rtype: torch.Tensor
        """
        timestep = self._env.reset()
        return self._observation_to_tensor(timestep.observation)

    def render(self, *args, **kwargs) -> None:
        """Render the environment

        OpenCV is used to render the environment.
        Install OpenCV with ``pip install opencv-python``
        """
        frame = self._env.physics.render(480, 640, camera_id=0)

        # render the frame using OpenCV
        import cv2
        cv2.imshow("env", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

    def close(self) -> None:
        """Close the environment
        """
        self._env.close()


def wrap_env(env: Any, wrapper: str = "auto", verbose: bool = True) -> WrapperMix:
    """Wrap an environment to use a common interface

    Example::

        >>> from skrl.envs.torch import wrap_env
        >>>
        >>> # assuming that there is an environment called "env"
        >>> env = wrap_env(env)

    :param env: The environment to be wrapped
    :type env: gym.Env, dm_env.Environment or VecTask
    :param wrapper: The type of wrapper to use (default: "auto").
                    If ``"auto"``, the wrapper will be automatically selected based on the environment class.
                    The supported wrappers are described in the following table:

                    .. raw:: html

                        <br>

                    +--------------------+-------------------------+
                    |Environment         |Wrapper tag              |
                    +====================+=========================+
                    |OpenAI Gym          |``"gym"``                |
                    +--------------------+-------------------------+
                    |DeepMind            |``"dm"``                 |
                    +--------------------+-------------------------+
                    |Isaac Gym preview 2 |``"isaacgym-preview2"``  |
                    +--------------------+-------------------------+
                    |Isaac Gym preview 3 |``"isaacgym-preview3"``  |
                    +--------------------+-------------------------+
                    |Isaac Gym preview 4 |``"isaacgym-preview4"``  |
                    +--------------------+-------------------------+
                    |Omniverse Isaac Gym |``"omniverse-isaacgym"`` |
                    +--------------------+-------------------------+
    :type wrapper: str, optional
    :param verbose: Whether to print the wrapper type (default: True)
    :type verbose: bool, optional

    :raises ValueError: Unknow wrapper type

    :return: Wrapped environment
    :rtype: Wrapper
    """
    if verbose:
        logger.info("Environment class: {}".format(", ".join([str(base).replace("<class '", "").replace("'>", "") \
            for base in env.__class__.__bases__])))
    if wrapper == "auto":
        base_classes = [str(base) for base in env.__class__.__bases__]
        if "<class 'omni.isaac.gym.vec_env.vec_env_base.VecEnvBase'>" in base_classes or \
            "<class 'omni.isaac.gym.vec_env.vec_env_mt.VecEnvMT'>" in base_classes:
            if verbose:
                logger.info("Environment wrapper: Omniverse Isaac Gym")
            return OmniverseIsaacGymWrapper(env)
        elif isinstance(env, gym.core.Env) or isinstance(env, gym.core.Wrapper):
            if verbose:
                logger.info("Environment wrapper: Gym")
            return GymWrapper(env)
        elif "<class 'dm_env._environment.Environment'>" in base_classes:
            if verbose:
                logger.info("Environment wrapper: DeepMind")
            return DeepMindWrapper(env)
        elif "<class 'rlgpu.tasks.base.vec_task.VecTask'>" in base_classes:
            if verbose:
                logger.info("Environment wrapper: Isaac Gym (preview 2)")
            return IsaacGymPreview2Wrapper(env)
        if verbose:
            logger.info("Environment wrapper: Isaac Gym (preview 3/4)")
        return IsaacGymPreview3Wrapper(env)  # preview 4 is the same as 3
    elif wrapper == "gym":
        if verbose:
            logger.info("Environment wrapper: Gym")
        return GymWrapper(env)
    elif wrapper == "dm":
        if verbose:
            logger.info("Environment wrapper: DeepMind")
        return DeepMindWrapper(env)
    elif wrapper == "isaacgym-preview2":
        if verbose:
            logger.info("Environment wrapper: Isaac Gym (preview 2)")
        return IsaacGymPreview2Wrapper(env)
    elif wrapper == "isaacgym-preview3":
        if verbose:
            logger.info("Environment wrapper: Isaac Gym (preview 3)")
        return IsaacGymPreview3Wrapper(env)
    elif wrapper == "isaacgym-preview4":
        if verbose:
            logger.info("Environment wrapper: Isaac Gym (preview 4)")
        return IsaacGymPreview3Wrapper(env)  # preview 4 is the same as 3
    elif wrapper == "omniverse-isaacgym":
        if verbose:
            logger.info("Environment wrapper: Omniverse Isaac Gym")
        return OmniverseIsaacGymWrapper(env)
    else:
        raise ValueError("Unknown {} wrapper type".format(wrapper))
