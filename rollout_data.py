import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import envpool
import flax
from jax.config import config
import gym
import optax
import orbax.checkpoint
from icrl.common.Lagrange import Lagrange
from icrl.common.RunningMeanStd import RunningMeanStd
from flax.training import orbax_utils

from icrl.constraint_rl.ppo_lag_mujoco_envpool_xla_jax import (
    Actor,
    Critic,
    PPOTrainState,
    AgentParams,
    RolloutState,
)
from icrl.common.venv_wrappers import (
    EnvWrapper,
    MojocoEnvDtypeAct,
    VectorEnvClipAct,
    VectorEnvNormObs,
    VectorEnvWrapper,
)
from icrl.env.cost_env import EpisodeStatistics, make_cost_env


@flax.struct.dataclass
class FakeVectorEnvNormObs(EnvWrapper):
    obs_rms:RunningMeanStd=RunningMeanStd()
    def recv(self,ret) :
        next_obs, reward, next_done,next_truncated, info= ret
        obs_rms=self.obs_rms
        return self.replace(obs_rms=obs_rms), (obs_rms.norm(next_obs), reward, next_done,next_truncated, info)

    def reset(self,ret):
        obs,info = ret
        obs_rms=self.obs_rms
        obs=obs_rms.norm(obs).astype(jnp.float32)
        return self.replace(obs_rms=obs_rms), (obs,info)

seed=0
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
import orbax.checkpoint

env_id = "BlockedWalker"

num_envs = 64
def get_wrappers(envs):
    wrappers = [
        VectorEnvNormObs(),
        VectorEnvClipAct(envs.action_space.low, envs.action_space.high),
    ]
    if envs.action_space.dtype == np.float64:
        config.update("jax_enable_x64", True)
        wrappers=wrappers + [MojocoEnvDtypeAct()]
    return wrappers
# wrappers=[]
envs, handle, step_env, episode_stats,cost_function =make_cost_env(env_id,num_envs,seed,get_wrappers)
handle, (next_obs, info) = envs.reset()
next_obs = next_obs.astype(jnp.float32)
next_done = jnp.zeros(num_envs, dtype=jax.numpy.bool_)
next_truncated = jnp.zeros(num_envs, dtype=jax.numpy.bool_)
single_action_space = envs.action_space
single_observation_space = envs.observation_space
actor = Actor(
    action_dim=np.prod(single_action_space.shape),
)
critic = Critic()
cost_critic = Critic()
key = jax.random.PRNGKey(0)

agent_state = PPOTrainState.create(
    apply_fn=None,
    params=AgentParams(
        actor.init(
            key,
            np.array([single_observation_space.sample()], dtype=jnp.float32),
        ),
        critic.init(
            key,
            np.array([single_observation_space.sample()], dtype=jnp.float32),
        ),
        cost_critic.init(
            key,
            np.array([single_observation_space.sample()], dtype=jnp.float32),
        ),
    ),
    tx=optax.chain(
        optax.clip_by_global_norm(0.1),
        optax.inject_hyperparams(optax.adam)(learning_rate=0.1),
    ),
    ret_rms=RunningMeanStd(),
    cost_ret_rms=RunningMeanStd(),
    lagrange=Lagrange.create(100.0, 0.035, 0.0, 0.01),
)

rollout_state = RolloutState(
    agent_state, episode_stats, next_obs, next_done, next_truncated, key, handle
)

@jax.jit
def get_action_and_value(
    agent_state: TrainState,
    next_obs: np.ndarray,
    key: jax.random.PRNGKey,
):
    """sample action, calculate value, logprob, entropy, and update storage"""
    logits, stdlog = actor.apply(agent_state.params.actor_params, next_obs)
    key, subkey = jax.random.split(key)
    u = jax.random.normal(subkey, shape=logits.shape)
    action = logits + u * jnp.exp(stdlog)
    var = jnp.exp(2 * stdlog)
    logprob = (
        -((action - logits) ** 2) / (2 * var) - stdlog - jnp.log(jnp.sqrt(2 * jnp.pi))
    ).sum(
        -1
    )  # need gradient information
    value = critic.apply(agent_state.params.critic_params, next_obs)
    cost_value = critic.apply(agent_state.params.cost_critic_params, next_obs)
    return action, logprob, value, cost_value, key




def gen_traj(agent_state, agent_handle):
    key = jax.random.PRNGKey(0)
    num_envs=10
    def get_wrappers(envs):
        wrappers = [
            FakeVectorEnvNormObs().replace(obs_rms=agent_handle[1].obs_rms),
            VectorEnvClipAct(envs.action_space.low, envs.action_space.high),
        ]
        if envs.action_space.dtype == np.float64:
            config.update("jax_enable_x64", True)
            wrappers=wrappers + [MojocoEnvDtypeAct()]
        return wrappers
    # wrappers=[]
    envs, handle, step_env, episode_stats,cost_function,env_module =make_cost_env(env_id,num_envs,seed,get_wrappers,True)
    handle, (obs, info) = envs.reset()
    obs_traj = []
    next_obs_traj = []
    traj_info=[]
    actions = []
    rewards = []
    dones = []
    r=0
    next_done=np.zeros(num_envs,dtype=bool)
    next_truncated=np.zeros(num_envs,dtype=bool)
    for i in range(envs.env.config['max_episode_steps']):
        mask=~(next_done | next_truncated)
        obs_traj.append((obs*jnp.sqrt(handle[1].obs_rms.var)+handle[1].obs_rms.mean)[mask])
        action, logprob, value, cost_value, key = get_action_and_value(
        agent_state, obs, key
        )
        actions.append(action[mask].clip(-1.0, 1.0))
        handle, (obs, reward, next_done, next_truncated, info) = step_env(
            handle, action
        )
        next_obs_traj.append((obs*jnp.sqrt(handle[1].obs_rms.var)+handle[1].obs_rms.mean)[mask])
        dones.append((next_done*(1-next_truncated))[mask])
        traj_info.append(jax.tree_map(
            lambda x: x[mask],info
        ))
        rewards.append(reward[mask])
        r+=reward
    return np.concatenate(obs_traj),np.concatenate(next_obs_traj),np.concatenate(actions),np.concatenate(dones),np.concatenate(rewards)

checkpointer = orbax.checkpoint.PyTreeCheckpointer()
import os
# Get all files in tmp/
    # Restore the save point

dirname=env_id
# dirname='BiasedSwimmer__ppo_lag_mujoco_envpool_xla_jax__1__169476728724'

rollout_state = checkpointer.restore(f'expert/{dirname}', item=rollout_state)
# save_args = orbax_utils.save_args_from_target(rollout_state.agent_state)
# Generate trajectory using gen_traj
traj_obs,traj_next_obs,action,dones,rewards = gen_traj(rollout_state.agent_state, rollout_state.handle)
def save_data():
    expert_data={'traj_obs':traj_obs,'action':action,'traj_next_obs': traj_next_obs,"dones":dones,"rewards":rewards}
    save_args = orbax_utils.save_args_from_target(expert_data)
    checkpointer.save(f"expert_data/{dirname}",expert_data, save_args=save_args)

save_data()
        # Plot the trajectory
#     def plottraj(traj_info):
#         x,y= np.array([info['x_position'] for info in traj_info]),np.array([info['y_position'] for info in traj_info])#x,y both shaped (1000,500)
#         plt.plot(x.mean(1), y.mean(1),label=file)
#         plt.legend()
#     plottraj(traj_info)
# plt.savefig("a.jpg")
