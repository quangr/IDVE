import jax.numpy as jnp
import jax
import numpy as np
import gym as ogym
from flax.training import orbax_utils
import orbax.checkpoint
env_id="HalfCheetah-v4"

def cost_function(next_obs, reward, next_done, next_truncated, info):
    if 'true_obs' in info.keys():
        return  jnp.abs(info['true_obs'][:,5])+jnp.abs(info['true_obs'][:,6])+jnp.abs(info['true_obs'][:,7])
    else:
        return jnp.abs(next_obs[:,5])+jnp.abs(next_obs[:,6])+jnp.abs(next_obs[:,7])

single_mask = np.ones(17)
# single_mask = np.zeros(17)
# single_mask[5:8] = 1

def offline_dataset():
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    data=checkpointer.restore("tmp/buffer/BlockedHalfCheetah_abs/BlockedHalfCheetah_abs__new_sac_sample__1__1695352002/")
    dataset={}
    dataset["observations"]=data[0][:100000,0]
    dataset["next_observations"]=data[1][:100000,0]
    dataset["actions"]=data[2][:100000,0]
    dataset["rewards"]=(data[5]['reward_ctrl']+data[5]['x_velocity'])[1:,0]
    dataset["terminals"]=data[3][:100000,0]
    dataset["timeouts"]=jnp.zeros_like(dataset["terminals"])
    return dataset