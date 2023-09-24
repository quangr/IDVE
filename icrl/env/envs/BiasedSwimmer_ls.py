env_id="Swimmer-v4"
import jax.numpy as jnp
from flax.training import orbax_utils
import orbax.checkpoint
import flax
from icrl.common.venv_wrappers import EnvWrapper
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
        return (info['true_obs'][:,3]>0.15)
    else:
        return (next_obs[:,3]>0.15)
def reward_function(next_obs, reward, next_done, next_truncated, info):
    return (info['reward_ctrl']+info['reward_fwd']).astype(jnp.float32)
def offline_dataset():
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    data=checkpointer.restore("/home/guorui/jax-rl/tmp/buffer/BiasedSwimmer_ls/BiasedSwimmer_ls__new_sac__1__1695197265")
    dataset={}
    dataset["observations"]=data[0][:100000,0]
    dataset["next_observations"]=data[1][:100000,0]
    dataset["actions"]=data[2][:100000,0]
    dataset["rewards"]=(data[5]['reward_ctrl']+data[5]['x_velocity'])[1:,0]
    dataset["terminals"]=data[3][:100000,0]
    dataset["timeouts"]=jnp.zeros_like(dataset["terminals"])
    return dataset