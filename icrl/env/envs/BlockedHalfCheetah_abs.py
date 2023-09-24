import jax.numpy as jnp
import jax
import numpy as np
import gym as ogym
import d4rl
from flax.training import orbax_utils
import orbax.checkpoint
env_id="HalfCheetah-v4"

def cost_function(next_obs, reward, next_done, next_truncated, info):
    if 'true_obs' in info.keys():
        return  jnp.abs(info['true_obs'][:,5])+jnp.abs(info['true_obs'][:,6])+jnp.abs(info['true_obs'][:,7])
    else:
        return jnp.abs(next_obs[:,5])+jnp.abs(next_obs[:,6])+jnp.abs(next_obs[:,7])
def reward_function(next_obs, reward, next_done, next_truncated, info):
    return (info['reward_ctrl']+(info['reward_run'] if info['reward_run']>0 else -0.75*info['reward_run']) ).astype(jnp.float32)
def offline_dataset():
    optimal_data=ogym.make('halfcheetah-medium-replay-v2')
    dataset=optimal_data.get_dataset()
    dataset['rewards']
    return dataset