import jax.numpy as jnp
import jax
import numpy as np
import gym as ogym
import d4rl
from flax.training import orbax_utils
import orbax.checkpoint
env_id="Ant-v4"

def cost_function(next_obs, reward, next_done, next_truncated, info):
    if 'true_obs' in info.keys():
        return  (info['true_obs'][:,13]>0.2)
    else:
        return (next_obs[:,13]>0.2)


def offline_dataset():
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    data=checkpointer.restore("/home/guorui/jax-rl/tmp/buffer/Ant_ls/Ant_ls__new_sac_sample__1__1695367454/")
    dataset={}
    dataset["observations"]=data[0][:100000,0]
    dataset["next_observations"]=data[1][:100000,0]
    dataset["actions"]=data[2][:100000,0]
    dataset["rewards"]=data[4][:100000,0]
    # dataset["rewards"]=(data[5]['reward_ctrl']+data[5]['x_velocity'])[1:,0]
    dataset["terminals"]=data[3][:100000,0]
    dataset["timeouts"]=jnp.zeros_like(dataset["terminals"])
    return dataset
# def offline_dataset():
#     optimal_data=ogym.make('ant-medium-replay-v2')
#     return optimal_data.get_dataset()