import envpool
import flax
import jax.numpy as jnp
import os
import importlib.util
import gymnasium as gym
from icrl.common.venv_wrappers import VectorEnvWrapper
from icrl.common.DummyVecEnv import DummyVecEnv

@flax.struct.dataclass
class EpisodeStatistics:
    episode_costs: jnp.array
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_costs: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array



def load_cost_function(env_id):

    # Get the directory of the current file
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    # Set the path to the envs directory relative to the current file's location
    envs_dir = os.path.join(current_file_dir, "envs")

    # Find all files in the envs directory
    files = os.listdir(envs_dir)

    # Look for the Python module that matches the envs_id
    for file in files:
        if file.endswith(".py") and file[:-3] == env_id:
            module_name = f"envs.{file[:-3]}"
            module_path = os.path.join(envs_dir, file)

            # Dynamically load the module
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Check if the cost_function function exists in the module
            if hasattr(module, "cost_function"):
                return module
            else:
                raise AttributeError(f"Module '{module_name}' does not have 'cost_function' function.")
    
    # If the envs_id does not match any module, raise an error
    raise ValueError(f"No module found for envs_id '{env_id}' in the 'envs' directory.")

def make_cost_env(env_id,num_envs,seed,get_wrappers,get_module=False):    
    try:
        module = load_cost_function(env_id)
        cost_function=module.cost_function
        env_id=module.env_id
        # Now you can use the cost_function
        # For example: result = cost_function(some_arguments)
    except Exception:
        print("Not a custom env, assigning cost function 0")
        def cost_function(next_obs, reward, next_done, next_truncated, info):
            return jnp.zeros_like(reward)

    envs = envpool.make(
        env_id,
        env_type="gymnasium",
        num_envs=num_envs,
        seed=seed,
    )
    num_envs = num_envs
    envs.is_vector_env = True
    episode_stats = EpisodeStatistics(
        episode_costs=jnp.zeros(num_envs, dtype=jnp.float32),
        episode_returns=jnp.zeros(num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(num_envs, dtype=jnp.int32),
        returned_episode_costs=jnp.zeros(num_envs, dtype=jnp.float32),
        returned_episode_returns=jnp.zeros(num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(num_envs, dtype=jnp.int32),
    )
    wrappers=get_wrappers(envs)
    if hasattr(module, 'get_wrappers'):
        wrappers += module.get_wrappers(envs)
    envs = VectorEnvWrapper(envs, wrappers)

    handle, recv, send, step_env = envs.xla()
    if get_module:
        return envs, handle, step_env, episode_stats,cost_function,module
    else:
        return envs, handle, step_env, episode_stats,cost_function

def make_cost_env_no_xla(env_id,num_envs,seed,get_module=False):    
    try:
        module = load_cost_function(env_id)
        cost_function=module.cost_function
        env_id=module.env_id
        # Now you can use the cost_function
        # For example: result = cost_function(some_arguments)
    except Exception:
        print("Not a custom env, assigning cost function 0")
        def cost_function(next_obs, reward, next_done, next_truncated, info):
            return jnp.zeros_like(reward)

    try:
        envs = envpool.make(
            env_id,
            env_type="gymnasium",
            num_envs=num_envs,
            seed=seed,
        )
    except Exception:
        envs = DummyVecEnv([lambda : gym.make(env_id) for i in range(num_envs)])    
    if get_module:
        return envs, cost_function,module
    else:
        return envs, cost_function


def step_env_wrappeed_factory(step_env,cost_function):
    def step_env_wrappeed(episode_stats, handle, action):
        handle, (next_obs, reward, next_done, next_truncated, info) = step_env(
            handle, action
        )
        cost = cost_function(next_obs, reward, next_done, next_truncated, info).astype(
            jnp.float32
        )
        new_episode_cost = episode_stats.episode_costs + cost
        new_episode_return = episode_stats.episode_returns + reward
        new_episode_length = episode_stats.episode_lengths + 1
        episode_stats = episode_stats.replace(
            episode_costs=(new_episode_cost) * (1 - next_done) * (1 - next_truncated),
            episode_returns=(new_episode_return)
            * (1 - next_done)
            * (1 - next_truncated),
            episode_lengths=(new_episode_length)
            * (1 - next_done)
            * (1 - next_truncated),
            # only update the `returned_episode_returns` if the episode is done
            returned_episode_costs=jnp.where(
                next_done + next_truncated,
                new_episode_cost,
                episode_stats.returned_episode_costs,
            ),
            returned_episode_returns=jnp.where(
                next_done + next_truncated,
                new_episode_return,
                episode_stats.returned_episode_returns,
            ),
            returned_episode_lengths=jnp.where(
                next_done + next_truncated,
                new_episode_length,
                episode_stats.returned_episode_lengths,
            ),
        )
        return (
            episode_stats,
            handle,
            (next_obs, reward, cost, next_done, next_truncated, info),
        )
    return step_env_wrappeed
