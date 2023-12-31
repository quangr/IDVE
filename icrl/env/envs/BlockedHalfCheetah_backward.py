import jax.numpy as jnp
import jax
import numpy as np
import gym as ogym
import d4rl
from flax.training import orbax_utils
import orbax.checkpoint
import flax
from icrl.common.venv_wrappers import EnvWrapper
env_id="HalfCheetah-v4"
@flax.struct.dataclass
class RewardWrapper(EnvWrapper):
    def recv(self,ret) :
        next_obs, reward, next_done,next_truncated, info= ret
        reward=reward_function(next_obs, reward, next_done, next_truncated, info)
        return self, (next_obs, reward, next_done,next_truncated, info)

def get_wrappers(envs):
    return [RewardWrapper()]

def cost_function(next_obs, reward, next_done, next_truncated, info):
    if 'true_obs' in info.keys():
        return  jnp.abs(info['true_obs'][:,5])+jnp.abs(info['true_obs'][:,6])+jnp.abs(info['true_obs'][:,7])
    else:
        return jnp.abs(next_obs[:,5])+jnp.abs(next_obs[:,6])+jnp.abs(next_obs[:,7])
def reward_function(next_obs, reward, next_done, next_truncated, info):
    return (info['reward_ctrl']-info['reward_run']).astype(jnp.float32)

single_mask = np.ones(17)
single_mask[8:] = 0
# single_mask = np.zeros(17)
# single_mask[5:8] = 1

# def offline_dataset():
#     checkpointer = orbax.checkpoint.PyTreeCheckpointer()
#     data=checkpointer.restore("tmp/buffer/BlockedHalfCheetah/BlockedHalfCheetah__new_sac__1__1694673157/")
#     dataset={}
#     dataset["observations"]=data[0][:100000,0]
#     dataset["next_observations"]=data[1][:100000,0]
#     dataset["actions"]=data[2][:100000,0]
#     dataset["rewards"]=data[4][:100000,0]
#     dataset["terminals"]=data[3][:100000,0]
#     dataset["timeouts"]=jnp.zeros_like(dataset["terminals"])
#     return dataset
def offline_dataset():
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    expert_indice = np.random.permutation(100000)[:50000]
    data=checkpointer.restore("/home/guorui/jax-rl/tmp/buffer/BlockedHalfCheetah_backward/BlockedHalfCheetah_backward__new_sac_sample__1__1695446461/")
    dataset_extra={}
    dataset_extra["observations"]=data[0][:100000,0]
    dataset_extra["next_observations"]=data[1][:100000,0]
    dataset_extra["actions"]=data[2][:100000,0]
    dataset_extra["rewards"]=(data[5]['reward_ctrl']-data[5]['x_velocity'])[1:,0]
    dataset_extra["terminals"]=data[3][:100000,0]
    dataset_extra["timeouts"]=jnp.zeros_like(dataset_extra["terminals"])
    dataset_extra=jax.tree_map(lambda x:x[expert_indice],dataset_extra)

    data=checkpointer.restore("/home/guorui/jax-rl/tmp/buffer/BlockedHalfCheetah_abs/BlockedHalfCheetah_abs__new_sac_sample__1__1695352002/")
    dataset={}
    dataset["observations"]=data[0][:100000,0]
    dataset["next_observations"]=data[1][:100000,0]
    dataset["actions"]=data[2][:100000,0]
    dataset["rewards"]=(data[5]['reward_ctrl']-data[5]['x_velocity'])[1:,0]
    dataset["terminals"]=data[3][:100000,0]
    dataset["timeouts"]=jnp.zeros_like(dataset["terminals"])
    dataset=jax.tree_map(lambda x,y:jnp.concatenate([x,y]),dataset,dataset_extra)
    return dataset