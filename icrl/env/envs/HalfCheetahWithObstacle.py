import jax.numpy as jnp
import jax
import numpy as np
import gym as ogym
import flax
import jax.numpy as jnp
from flax.training import orbax_utils
import orbax.checkpoint

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

def reward_function(next_obs, reward, next_done, next_truncated, info):
    return (info['reward_ctrl']+jnp.abs(info['reward_run'])).astype(jnp.float32)

def cost_function(next_obs, reward, next_done, next_truncated, info):
    return info['x_position']<-3

single_mask = np.ones(17)
def offline_dataset():
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    
    data=checkpointer.restore("tmp/buffer/HalfCheetahWithObstacle/HalfCheetahWithObstacle__new_sac__1__1695722356")
    dataset={}
    dataset["observations"]=data[0][:100000,0]
    dataset["next_observations"]=data[1][:100000,0]
    dataset["actions"]=data[2][:100000,0]
    dataset["rewards"]=data[4][:100000,0]
    dataset["terminals"]=data[3][:100000,0]
    dataset["infos"]=data[5]
    dataset["timeouts"]=jnp.zeros_like(dataset["terminals"])
    return dataset