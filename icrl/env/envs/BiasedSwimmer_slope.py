env_id="Swimmer-v4"
import jax.numpy as jnp
from flax.training import orbax_utils
import orbax.checkpoint
def cost_function(next_obs, reward, next_done, next_truncated, info):
    if 'true_obs' in info.keys():
        return (info['true_obs'][:,3]>0.15)
    else:
        return (next_obs[:,3]>0.15)
theta=-jnp.pi/6
def reward_function(next_obs, reward, next_done, next_truncated, info):
    return (info['reward_ctrl']+(jnp.cos(theta)*info['x_velocity']+jnp.sin(theta)*(info['y_velocity']))).astype(jnp.float32)
def offline_dataset():
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    data=checkpointer.restore("/home/guorui/jax-rl/tmp/buffer/BiasedSwimmer_ls/BiasedSwimmer_ls__test_actor__1__169519784108")
    dataset={}
    dataset["observations"]=data[0][:100000,0]
    dataset["next_observations"]=data[1][:100000,0]
    dataset["actions"]=data[2][:100000,0]
    reward=(jnp.cos(theta)*data[5]['x_velocity']+jnp.sin(theta)*(data[5]['y_velocity']))
    dataset["rewards"]=(data[5]['reward_ctrl']+reward)[1:,0]
    dataset["terminals"]=data[3][:100000,0]
    dataset["timeouts"]=jnp.zeros_like(dataset["terminals"])
    return dataset