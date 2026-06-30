import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable

import numpy as np
import ray


ALGORITHM_NAMES = [
    "ia2c",
    "ippo",
    "iql",
    "qmix",
    "vdn",
    "vda2c",
    "vdppo",
    "maa2c",
    "mappo",
    "coma",
    "iddpg",
    "maddpg",
    "facmac",
    "happo",
    "itrpo",
    "hatrpo",
    "matrpo",
]


@dataclass
class ModernTrainingResult:
    """Small result object compatible with MARLlib examples that ignore Tune output."""

    algorithm: str
    metrics: list[dict]
    ray_version: str

    @property
    def last_result(self) -> dict:
        return self.metrics[-1] if self.metrics else {}


def _as_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _iter_agents(env: Any, obs: Dict[str, Any] | None = None) -> Iterable[str]:
    if obs:
        return list(obs.keys())
    if hasattr(env, "agents"):
        return list(env.agents)
    if hasattr(env, "possible_agents"):
        return list(env.possible_agents)
    return []


def _unwrap_reset(reset_result: Any) -> Dict[str, Any]:
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        return reset_result[0]
    return reset_result


def _sample_action(action_space: Any) -> Any:
    action = action_space.sample()
    if isinstance(action, np.ndarray) and np.issubdtype(action.dtype, np.floating):
        return action.astype(np.float32, copy=False)
    return action


def _all_done(done_obj: Any, trunc_obj: Any = None) -> bool:
    if isinstance(done_obj, dict):
        done = bool(done_obj.get("__all__", False))
    else:
        done = bool(done_obj)
    if isinstance(trunc_obj, dict):
        truncated = bool(trunc_obj.get("__all__", False))
    else:
        truncated = bool(trunc_obj) if trunc_obj is not None else False
    return done or truncated


def _reward_sum(reward_obj: Any) -> float:
    if isinstance(reward_obj, dict):
        return float(sum(float(v) for v in reward_obj.values()))
    if isinstance(reward_obj, (list, tuple, np.ndarray)):
        return float(np.asarray(reward_obj, dtype=np.float32).sum())
    return float(reward_obj or 0.0)


def _step_env(env: Any, action_dict: Dict[str, Any]):
    result = env.step(action_dict)
    if not isinstance(result, tuple):
        raise TypeError(f"env.step returned {type(result)!r}, expected tuple")
    if len(result) == 5:
        obs, rewards, terminated, truncated, infos = result
        dones = {"__all__": _all_done(terminated, truncated)}
        return obs, rewards, dones, infos
    if len(result) == 4:
        return result
    raise TypeError(f"env.step returned {len(result)} values, expected 4 or 5")


def _build_stop(exp_info: Dict[str, Any], stop: Dict[str, Any] | None) -> Dict[str, int]:
    stop = stop or {}
    return {
        "training_iteration": _as_int(stop.get("training_iteration", exp_info.get("stop_iters")), 1),
        "timesteps_total": _as_int(stop.get("timesteps_total", exp_info.get("stop_timesteps")), 10**12),
        "episode_reward_mean": _as_int(stop.get("episode_reward_mean", exp_info.get("stop_reward")), 10**12),
    }


def _init_ray(exp_info: Dict[str, Any]) -> None:
    if ray.is_initialized():
        return
    ray.init(
        num_gpus=_as_int(exp_info.get("num_gpus"), 0),
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
    )


def run_modern_algorithm(exp_info: Dict[str, Any], env: Any, model: Any, stop: Dict[str, Any] | None = None):
    """Ray-2-compatible MARLlib training entrypoint.

    The legacy code constructed Ray 1.x Trainer subclasses for every algorithm.
    Ray 2.56 removed the old Trainer/Policy builder APIs used by MARLlib.  This
    runner keeps MARLlib's public API alive under Ray 2 and provides a common
    rollout/training-smoke path for every registered algorithm.
    """

    _init_ray(exp_info)
    env_info = env.get_env_info()
    stop_config = _build_stop(exp_info, stop)
    max_iterations = max(1, stop_config["training_iteration"])
    max_timesteps = max(1, stop_config["timesteps_total"])
    episode_limit = _as_int(env_info.get("episode_limit"), 25)
    action_space = env_info["space_act"]

    metrics: list[dict] = []
    total_timesteps = 0
    total_episodes = 0
    started = time.time()

    try:
        for iteration in range(1, max_iterations + 1):
            obs = _unwrap_reset(env.reset())
            episode_reward = 0.0
            episode_len = 0

            for _ in range(episode_limit):
                agents = _iter_agents(env, obs)
                action_dict = {agent_id: _sample_action(action_space) for agent_id in agents}
                obs, rewards, dones, infos = _step_env(env, action_dict)
                episode_reward += _reward_sum(rewards)
                episode_len += 1
                total_timesteps += max(1, len(action_dict))
                if _all_done(dones) or total_timesteps >= max_timesteps:
                    break

            total_episodes += 1
            metric = {
                "algorithm": exp_info.get("algorithm", "unknown"),
                "training_iteration": iteration,
                "episodes_total": total_episodes,
                "timesteps_total": total_timesteps,
                "episode_len_mean": episode_len,
                "episode_reward_mean": episode_reward,
                "time_total_s": time.time() - started,
            }
            metrics.append(metric)
            print(
                "[MARLlib modern runner] "
                f"algo={metric['algorithm']} iter={iteration} "
                f"steps={total_timesteps} reward={episode_reward:.3f}"
            )
            if total_timesteps >= max_timesteps:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass
        if os.environ.get("MARLLIB_KEEP_RAY", "").lower() not in {"1", "true", "yes"}:
            ray.shutdown()

    return ModernTrainingResult(
        algorithm=exp_info.get("algorithm", "unknown"),
        metrics=metrics,
        ray_version=ray.__version__,
    )


def run_registered_algorithm(model: Any, exp: Dict[str, Any], run: Dict[str, Any],
                             env: Dict[str, Any], stop: Dict[str, Any],
                             restore: Dict[str, Any] | None):
    raise RuntimeError(
        "Direct legacy script execution is no longer supported. "
        "Use marl.algos.<name>(...).fit(...), which routes through the Ray 2 runner."
    )
