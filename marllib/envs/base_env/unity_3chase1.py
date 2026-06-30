from pathlib import Path
from typing import Any, Dict

import numpy as np
from gym.spaces import Box, Dict as GymDict
from ray.rllib.env.multi_agent_env import MultiAgentEnv


def _resolve_binary_path(path: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (Path.cwd() / raw).resolve()


def _validate_binary_path(path: str) -> str:
    resolved = _resolve_binary_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Unity binary not found: {path} (resolved to {resolved})")
    if not resolved.is_file():
        raise FileNotFoundError(f"Unity binary path is not a file: {resolved}")
    if not resolved.stat().st_mode & 0o111:
        raise PermissionError(f"Unity binary is not executable: {resolved}")
    if resolved.suffix == ".x86_64":
        data_dir = resolved.with_name(f"{resolved.stem}_Data")
        player = resolved.with_name("UnityPlayer.so")
        missing = [str(p) for p in (data_dir, player) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Unity Linux build is incomplete. Missing companion files: "
                + ", ".join(missing)
            )
    return str(resolved)


def _flatten_obs(obs: Any, target_dim: int | None = None) -> np.ndarray:
    if isinstance(obs, dict) and "observation" in obs:
        flat = np.concatenate([np.asarray(o, dtype=np.float32).reshape(-1) for o in obs["observation"]])
    elif isinstance(obs, (list, tuple)):
        flat = np.concatenate([np.asarray(o, dtype=np.float32).reshape(-1) for o in obs])
    else:
        flat = np.asarray(obs, dtype=np.float32).reshape(-1)
    if target_dim is None:
        return flat.astype(np.float32, copy=False)
    if flat.shape[0] < target_dim:
        flat = np.pad(flat, (0, target_dim - flat.shape[0]), constant_values=0.0)
    elif flat.shape[0] > target_dim:
        flat = flat[:target_dim]
    return flat.astype(np.float32, copy=False)


def _space_size(space: Any) -> int:
    if hasattr(space, "shape") and space.shape is not None:
        return int(np.prod(space.shape))
    if hasattr(space, "spaces"):
        return sum(_space_size(s) for s in space.spaces)
    if hasattr(space, "n"):
        return int(space.n)
    return 1


class RLlibUnity3Chase1(MultiAgentEnv):
    """Ray-compatible wrapper for the 3Chase1 Unity ML-Agents build."""

    ROLE_PREFIXES = ("Herder", "Netter", "Prey")
    LEARNING_PREFIXES = ("Herder", "Netter")

    def __init__(self, env_config: Dict[str, Any]):
        from mlagents_envs.environment import UnityEnvironment
        from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
        from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

        cfg = dict(env_config)
        binary_path = _validate_binary_path(
            cfg.get(
                "unity_env_binary_path",
                "artifacts/unity_builds/3Chase1/RLChase_FishEscapeHybrid_NewNet.x86_64",
            )
        )
        self.episode_limit = int(cfg.get("episode_limit", 900))
        self.worker_id = int(cfg.get("worker_id", 0))
        self.base_port = int(cfg.get("env_base_port", 5005))
        self._step_count = 0

        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale=float(cfg.get("time_scale", 10.0)))
        unity_env = None
        last_error = None
        max_retries = int(cfg.get("max_port_retries", 16))
        for attempt in range(max_retries):
            try:
                unity_env = UnityEnvironment(
                    file_name=binary_path,
                    side_channels=[channel],
                    no_graphics=bool(cfg.get("no_graphics", True)),
                    seed=int(cfg.get("seed", 321)) + self.worker_id,
                    base_port=self.base_port + attempt,
                    worker_id=self.worker_id,
                )
                break
            except Exception as exc:
                last_error = exc
                channel = EngineConfigurationChannel()
                channel.set_configuration_parameters(time_scale=float(cfg.get("time_scale", 10.0)))
        if unity_env is None:
            raise RuntimeError(
                f"Failed to start Unity environment after {max_retries} port attempts "
                f"from base_port={self.base_port}, worker_id={self.worker_id}: {last_error}"
            ) from last_error
        self.env = UnityParallelEnv(unity_env)
        self.possible_agents = list(self.env.possible_agents)
        self.learning_agents = [
            agent for agent in self.possible_agents
            if any(prefix in agent for prefix in self.LEARNING_PREFIXES)
        ]
        self.prey_agents = [agent for agent in self.possible_agents if "Prey" in agent]
        self.agents = list(self.learning_agents)

        obs_dims = [_space_size(self.env.observation_space(agent)) for agent in self.learning_agents]
        act_dims = [_space_size(self.env.action_space(agent)) for agent in self.possible_agents]
        self._obs_dim = max(obs_dims) if obs_dims else 1
        self._action_dims = dict(zip(self.possible_agents, act_dims))
        self._action_dim = max(self._action_dims[a] for a in self.learning_agents) if self.learning_agents else 1

        self.observation_space = GymDict({
            "obs": Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)
        })
        self.action_space = Box(low=-1.0, high=1.0, shape=(self._action_dim,), dtype=np.float32)
        self._last_raw_obs = None

    def _format_obs(self, obs_dict: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        return {
            agent: {"obs": _flatten_obs(obs_dict[agent], self._obs_dim)}
            for agent in self.learning_agents
            if agent in obs_dict
        }

    def _fit_action(self, agent: str, action: Any) -> np.ndarray:
        target_dim = self._action_dims.get(agent, self._action_dim)
        flat = np.asarray(action, dtype=np.float32).reshape(-1)
        if flat.shape[0] < target_dim:
            flat = np.pad(flat, (0, target_dim - flat.shape[0]), constant_values=0.0)
        elif flat.shape[0] > target_dim:
            flat = flat[:target_dim]
        return flat.astype(np.float32, copy=False)

    def _prey_action(self, agent: str) -> np.ndarray:
        return np.zeros(self._action_dims.get(agent, self._action_dim), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None and hasattr(self.env, "seed"):
            self.env.seed(seed)
        result = self.env.reset()
        obs_dict = result[0] if isinstance(result, tuple) else result
        self._last_raw_obs = obs_dict
        self._step_count = 0
        self.agents = list(self.learning_agents)
        return self._format_obs(obs_dict)

    def step(self, action_dict):
        unity_actions = {}
        for agent in self.learning_agents:
            unity_actions[agent] = self._fit_action(agent, action_dict.get(agent, np.zeros(self._action_dim)))
        for agent in self.prey_agents:
            unity_actions[agent] = self._prey_action(agent)

        result = self.env.step(unity_actions)
        if len(result) == 5:
            obs_dict, rewards, terminated, truncated, infos = result
            done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
        else:
            obs_dict, rewards, dones, infos = result
            done = bool(dones.get("__all__", False))

        self._last_raw_obs = obs_dict
        self._step_count += 1
        if self._step_count >= self.episode_limit:
            done = True

        obs = self._format_obs(obs_dict)
        reward_out = {agent: float(rewards.get(agent, 0.0)) for agent in self.learning_agents}
        done_out = {agent: done for agent in self.learning_agents}
        done_out["__all__"] = done
        info_out = {agent: infos.get(agent, {}) if isinstance(infos, dict) else {} for agent in self.learning_agents}
        return obs, reward_out, done_out, info_out

    def close(self):
        self.env.close()

    def get_env_info(self):
        return {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": len(self.learning_agents),
            "episode_limit": self.episode_limit,
            "policy_mapping_info": {
                "3Chase1": {
                    "description": "Herder and Netters learn against wrapper-controlled Prey",
                    "team_prefix": ("Herder", "Netter"),
                    "all_agents_one_policy": False,
                    "one_agent_one_policy": True,
                }
            },
        }
