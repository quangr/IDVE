env_id="Reacher-v4"
import jax.numpy as jnp
from flax.training import orbax_utils
import orbax.checkpoint
def cost_function(next_obs, reward, next_done, next_truncated, info):
    if 'true_obs' in info.keys():
        return (info['true_obs'][:,7]<0 )
    else:
        return (next_obs[:,7]<0 )

def offline_dataset():
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    dataset=checkpointer.restore("offlinedata/BiasedReacher")
    return dataset

