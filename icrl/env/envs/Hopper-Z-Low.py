import jax.numpy as jnp
import jax
import numpy as np
import gym as ogym
import d4rl
env_id="Hopper-v3"


def cost_function(next_obs, reward, next_done, next_truncated, info):
    if 'true_obs' in info.keys():
        return  (info['true_obs'][:,0]>1.3)
    else:
        return (next_obs[:,0]>1.3)
def offline_dataset():
    optimal_data=ogym.make('hopper-medium-v2')
    return optimal_data.get_dataset()