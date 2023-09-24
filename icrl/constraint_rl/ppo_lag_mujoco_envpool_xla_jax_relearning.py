# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy
import argparse
import os
os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
] = "0.1"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991
import random
import time
from collections import namedtuple
from distutils.util import strtobool
from functools import partial
from typing import Sequence

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jax.config import config
from icrl.common.Lagrange import Lagrange
import orbax.checkpoint
from flax.training import orbax_utils

from icrl.constraint_rl.sac import (
    Actor as sac_Actor,
    RLTrainState,
    VectorCritic
)

from icrl.common.RunningMeanStd import RunningMeanStd
from icrl.common.venv_wrappers import (
    MojocoEnvDtypeAct,
    VectorEnvClipAct,
    VectorEnvNormObs,
)
from orbax.checkpoint import CheckpointManagerOptions

from icrl.env.cost_env import load_cost_function
from icrl.common.venv_wrappers import VectorEnvWrapper
import envpool
# config.update("jax_default_matmul_precision", jax.lax.Precision.HIGHEST)
# config.update("jax_disable_jit", True)



def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="relearning",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="quangr",
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    # Algorithm specific arguments
    parser.add_argument("--rew-norm", type=int, default=False)
    parser.add_argument("--env-id", type=str, default="Ant-Z-Low",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--lagrange-learning-rate", type=float, default=3e-2,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=64,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=32,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.25,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--cost-limit", type=float, default=-500.0,
        help="the target KL divergence threshold")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_updates = args.total_timesteps // args.batch_size
    # fmt: on
    return args



RolloutState = namedtuple(
    "RolloutState",
    [
        "agent_state",
        "episode_stats",
        "next_obs",
        "next_done",
        "next_truncated",
        "key",
        "handle",
    ],
)


class Critic(nn.Module):
    dtype = jnp.float32
    param_dtype = jnp.float32
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            64, kernel_init=orthogonal(jnp.sqrt(2).astype(jnp.float32)), bias_init=constant(0.0)
        )(x)
        x = nn.tanh(x)
        x = nn.Dense(
            64, kernel_init=orthogonal(jnp.sqrt(2).astype(jnp.float32)), bias_init=constant(0.0)
        )(x)
        x = nn.tanh(x)
        return nn.Dense(
            1, kernel_init=orthogonal(jnp.sqrt(2).astype(jnp.float32)), bias_init=constant(0.0)
        )(x)


class Actor(nn.Module):
    action_dim: Sequence[int]
    dtype = jnp.float32
    param_dtype = jnp.float32
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            64, kernel_init=orthogonal(jnp.sqrt(2).astype(jnp.float32)), bias_init=constant(0.0)
        )(x)
        x = nn.tanh(x)
        x = nn.Dense(
            64, kernel_init=orthogonal(jnp.sqrt(2).astype(jnp.float32)), bias_init=constant(0.0)
        )(x)
        x = nn.tanh(x)
        x = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal((0.01) * jnp.sqrt(2).astype(jnp.float32)),
            bias_init=constant(0.0),
        )(x)
        # stdlog = -jnp.ones((self.action_dim,))/2
        stdlog = self.param(
            "stdlog", lambda rng, shape: -jnp.ones(shape,dtype=jnp.float32) / 2, (self.action_dim,)
        )
        return x, stdlog


class PPOTrainState(TrainState):
    ret_rms: RunningMeanStd
    cost_ret_rms: RunningMeanStd
    lagrange: Lagrange


@flax.struct.dataclass
class AgentParams:
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict
    cost_critic_params: flax.core.FrozenDict


@flax.struct.dataclass
class RewardMetrics:
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array


@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    truncated: jnp.array
    reward: RewardMetrics
    cost: RewardMetrics



orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

class ValueCritic(nn.Module):
    n_units: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


def read_expert_state():
  env_id = "Ant-v4"
  env = gym.make(env_id)
  obs = jnp.array([env.observation_space.sample()])
  action = jnp.array([env.action_space.sample()])

  actor = sac_Actor(action_dim=np.prod(env.action_space.shape))
  vf=ValueCritic()
  vf_state = TrainState.create(
      apply_fn=vf.apply,
      params=vf.init(actor_key, obs),
      tx=optax.adam(learning_rate=0.01),
  )

  actor_state = TrainState.create(
      apply_fn=actor.apply,
      params=actor.init(actor_key, obs),
      tx=optax.adam(learning_rate=0.01),
  )

  checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  qf = VectorCritic(n_critics=1)

  qf_state = RLTrainState.create(
      apply_fn=qf.apply,
      params=qf.init({"params": jax.random.PRNGKey(args.seed)}, obs, action),
      target_params=qf.init({"params": jax.random.PRNGKey(args.seed)}, obs, action),
      tx=optax.adam(learning_rate=0.01),
  )

  state={"qf_state":qf_state,"vf_state":vf_state,"actor_state":actor_state}
  state=checkpointer.restore("/home/guorui/jax-rl/sacpolicy/Ant-Z-Low/Ant-Z-Low__test__2__16938585444", item=state)
  return vf,qf,state


@flax.struct.dataclass
class EpisodeStatistics:
    episode_real_costs: jnp.array
    episode_costs: jnp.array
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_real_costs: jnp.array
    returned_episode_costs: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array

def make_cost_env(env_id,num_envs,seed,get_wrappers):    
    try:
        module = load_cost_function(env_id)
        cost_function=module.cost_function
        env_id=module.env_id
        # Now you can use the cost_function
        # For example: result = cost_function(some_arguments)
    except Exception:
        print("Not a custom env, assigning cost function 0")
        def cost_function(next_obs, reward, next_done, next_truncated, info):
            return jnp.zeros_like(reward)
    envs = envpool.make(
        env_id,
        env_type="gymnasium",
        num_envs=num_envs,
        seed=seed,
    )
    num_envs = num_envs
    envs.is_vector_env = True
    episode_stats = EpisodeStatistics(
        episode_real_costs=jnp.zeros(num_envs, dtype=jnp.float32),
        episode_costs=jnp.zeros(num_envs, dtype=jnp.float32),
        episode_returns=jnp.zeros(num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(num_envs, dtype=jnp.int32),
        returned_episode_real_costs=jnp.zeros(num_envs, dtype=jnp.float32),
        returned_episode_costs=jnp.zeros(num_envs, dtype=jnp.float32),
        returned_episode_returns=jnp.zeros(num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(num_envs, dtype=jnp.int32),
    )
    wrappers = get_wrappers(envs)
    envs = VectorEnvWrapper(envs, wrappers)

    handle, recv, send, step_env = envs.xla()
    return envs, handle, step_env, episode_stats,cost_function


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)

    # env setup
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
    envs, handle, step_env, episode_stats,old_cost_function =make_cost_env(args.env_id,args.num_envs,args.seed,get_wrappers)
    num_envs = args.num_envs
    single_action_space = envs.action_space
    single_observation_space = envs.observation_space
    envs.is_vector_env = True

    vf,qf,state=read_expert_state()
    def cost_function(action, next_obs, reward, next_done, next_truncated, info):
        return 0.99*qf.apply(state['qf_state'].target_params,info['true_obs'],action).reshape(-1)-vf.apply(state['vf_state'].params,info['true_obs']).reshape(-1)

    def step_env_wrappeed_factory(step_env,cost_function,real_cost_function):
        def step_env_wrappeed(episode_stats, handle, action):
            handle, (next_obs, reward, next_done, next_truncated, info) = step_env(
                handle, action
            )
            cost = cost_function(action,next_obs, reward, next_done, next_truncated, info).astype(
                jnp.float32
            )
            real_cost = real_cost_function(next_obs, reward, next_done, next_truncated, info).astype(
                jnp.float32
            )
            new_episode_real_cost = episode_stats.episode_real_costs + real_cost
            new_episode_cost = episode_stats.episode_costs + cost
            new_episode_return = episode_stats.episode_returns + reward
            new_episode_length = episode_stats.episode_lengths + 1
            episode_stats = episode_stats.replace(
                episode_real_costs=(new_episode_real_cost) * (1 - next_done) * (1 - next_truncated),
                episode_costs=(new_episode_cost) * (1 - next_done) * (1 - next_truncated),
                episode_returns=(new_episode_return)
                * (1 - next_done)
                * (1 - next_truncated),
                episode_lengths=(new_episode_length)
                * (1 - next_done)
                * (1 - next_truncated),
                # only update the `returned_episode_returns` if the episode is done
                returned_episode_costs=jnp.where(
                    next_done + next_truncated,
                    new_episode_cost,
                    episode_stats.returned_episode_costs,
                ),
                returned_episode_real_costs=jnp.where(
                    next_done + next_truncated,
                    new_episode_real_cost,
                    episode_stats.returned_episode_real_costs,
                ),
                returned_episode_returns=jnp.where(
                    next_done + next_truncated,
                    new_episode_return,
                    episode_stats.returned_episode_returns,
                ),
                returned_episode_lengths=jnp.where(
                    next_done + next_truncated,
                    new_episode_length,
                    episode_stats.returned_episode_lengths,
                ),
            )
            return (
                episode_stats,
                handle,
                (next_obs, reward, cost, next_done, next_truncated, info),
            )
        return step_env_wrappeed


    step_env_wrappeed=step_env_wrappeed_factory(step_env,cost_function,old_cost_function)

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches * args.update_epochs) gradient updates
        frac = (
            1.0
            - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
        )
        return args.learning_rate * frac

    actor = Actor(
        action_dim=np.prod(single_action_space.shape),
    )
    critic = Critic()
    cost_critic = Critic()
    agent_state = PPOTrainState.create(
        apply_fn=None,
        params=AgentParams(
            actor.init(
                actor_key,
                np.array([single_observation_space.sample()], dtype=jnp.float32),
            ),
            critic.init(
                critic_key,
                np.array([single_observation_space.sample()], dtype=jnp.float32),
            ),
            cost_critic.init(
                critic_key,
                np.array([single_observation_space.sample()], dtype=jnp.float32),
            ),
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if args.anneal_lr else args.learning_rate
            ),
        ),
        ret_rms=RunningMeanStd(),
        cost_ret_rms=RunningMeanStd(),
        lagrange=Lagrange.create(args.cost_limit, args.lagrange_learning_rate, 0.0, 0.),
    )
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        """sample action, calculate value, logprob, entropy, and update storage"""
        logits, stdlog = actor.apply(agent_state.params.actor_params, next_obs)
        key, subkey = jax.random.split(key)
        u = jax.random.normal(subkey, shape=logits.shape,dtype=jnp.float32)
        action = logits + u * jnp.exp(stdlog)
        var = jnp.exp(2 * stdlog)
        logprob = (
            -((action - logits) ** 2) / (2 * var)
            - stdlog
            - jnp.log(jnp.sqrt(2 * jnp.pi))
        ).sum(
            -1
        )  # need gradient information
        value = critic.apply(agent_state.params.critic_params, next_obs)
        cost_value = critic.apply(agent_state.params.cost_critic_params, next_obs)
        return action, logprob, value.squeeze(1), cost_value.squeeze(1), key

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
    ):
        """calculate value, logprob of supplied `action`, and entropy"""
        logits, stdlog = actor.apply(params.actor_params, x)
        var = jnp.exp(2 * stdlog)
        logprob = (
            -((action - logits) ** 2) / (2 * var)
            - stdlog
            - jnp.log(jnp.sqrt(2 * jnp.pi))
        ).sum(
            -1
        )  # need gradient information
        # normalize the logits https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        entropy = (2 * stdlog + jnp.log(2 * jnp.pi) + 1) / 2
        value = critic.apply(params.critic_params, x).squeeze()
        cost_value = cost_critic.apply(params.cost_critic_params, x).squeeze()
        return logprob, entropy.sum(-1), value, cost_value

    def compute_gae_once(carry, inp, gamma, gae_lambda):
        advantages = carry
        nextdone, nexttruncated, nextvalues, curvalues, reward = inp
        nextnonterminal = (1.0 - nextdone) * (1.0 - nexttruncated)

        delta = reward + gamma * nextvalues - curvalues  # mask done state
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    compute_gae_once = partial(
        compute_gae_once, gamma=args.gamma, gae_lambda=args.gae_lambda
    )

    def compute_advantages_and_returns(
        critic_params, ret_rms, next_obs, next_done, next_truncated, storage, rewards
    ):
        values = critic.apply(
            critic_params,
            jnp.concatenate([storage.obs, next_obs[None, :]], axis=0),
        ).squeeze()
        if args.rew_norm:
            values = values * jnp.sqrt(ret_rms.var).astype(jnp.float32)
        advantages = jnp.zeros((args.num_envs,),dtype=jnp.float32)
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        truncated = jnp.concatenate(
            [storage.truncated, next_truncated[None, :]], axis=0
        )
        _, advantages = jax.lax.scan(
            compute_gae_once,
            advantages,
            (
                dones[1:],
                truncated[1:],
                values[1:] * (1.0 - dones[1:]),
                values[:-1],
                rewards,
            ),
            reverse=True,
        )
        returns = advantages + values[:-1]
        return advantages, returns

    @jax.jit
    def compute_gae(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        next_truncated: np.ndarray,
        storage: Storage,
    ):
        advantages, returns = compute_advantages_and_returns(
            agent_state.params.critic_params,
            agent_state.ret_rms,
            next_obs,
            next_done,
            next_truncated,
            storage,
            storage.reward.rewards,
        )
        cost_advantages, cost_returns = compute_advantages_and_returns(
            agent_state.params.cost_critic_params,
            agent_state.cost_ret_rms,
            next_obs,
            next_done,
            next_truncated,
            storage,
            storage.cost.rewards,
        )
        if args.rew_norm:
            returns = (returns / jnp.sqrt(agent_state.ret_rms.var + 10e-8)).astype(
                jnp.float32
            )
            cost_returns = (
                cost_returns / jnp.sqrt(agent_state.cost_ret_rms.var + 10e-8)
            ).astype(jnp.float32)
            agent_state = agent_state.replace(
                ret_rms=agent_state.ret_rms.update(returns.flatten()),
                cost_ret_rms=agent_state.cost_ret_rms.update(cost_returns.flatten()),
            )

        storage = storage.replace(
            reward=storage.reward.replace(
                advantages=advantages,
                returns=returns,
            ),
            cost=storage.cost.replace(
                advantages=cost_advantages,
                returns=cost_returns,
            ),
        )
        return storage, agent_state

    def ppo_loss(params, x, a, logp, reward, cost, truncated, penalty):
        # (adv_r - penalty * adv_c) / (1 + penalty)
        mb_advantages = (reward.advantages - cost.advantages * penalty) / (1 + penalty)
        # mb_advantages = reward.advantages
        mb_returns = reward.returns
        mb_costreturns = cost.returns
        newlogprob, entropy, newvalue, newcostvalue = get_action_and_value2(
            params, x, a
        )
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                mb_advantages.std() + 1e-8
            )

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(
            ratio, 1 - args.clip_coef, 1 + args.clip_coef
        )
        pg_loss = (jnp.maximum(pg_loss1, pg_loss2) * (1 - truncated)).sum() / (
            1 - truncated
        ).sum()

        # Value loss
        v_loss = (((newvalue - mb_returns) * (1 - truncated)) ** 2).sum() / (
            1 - truncated
        ).sum()
        cost_v_loss = (
            ((newcostvalue - mb_costreturns) * (1 - truncated)) ** 2
        ).sum() / (1 - truncated).sum()

        entropy_loss = entropy.mean()
        loss = pg_loss + (v_loss + cost_v_loss) * args.vf_coef
        return loss, (
            pg_loss,
            v_loss,
            cost_v_loss,
            entropy_loss,
            jax.lax.stop_gradient(approx_kl),
        )

    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    @partial(jax.jit, donate_argnums=(0,1))
    def update_ppo(
        rollout_state: RolloutState,
        storage: Storage,
    ):
        agent_state = rollout_state.agent_state
        next_obs: np.ndarray = rollout_state.next_obs
        next_done: np.ndarray = rollout_state.next_done
        next_truncated: np.ndarray = rollout_state.next_truncated
        key: jax.random.PRNGKey = rollout_state.key
        penalty = agent_state.lagrange.state.params.astype(jnp.float32)

        def update_epoch(carry, unused_inp):
            agent_state, key = carry
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
                return x

            newstorage, agent_state = compute_gae(
                agent_state, next_obs, next_done, next_truncated, storage
            )
            flatten_storage = jax.tree_map(flatten, newstorage)  # seem uneffcient
            shuffled_storage = jax.tree_map(convert_data, flatten_storage)

            def update_minibatch(agent_state, minibatch):
                (
                    loss,
                    (pg_loss, v_loss, cost_v_loss, entropy_loss, approx_kl),
                ), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.reward,
                    minibatch.cost,
                    minibatch.truncated,
                    penalty,
                )
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (
                    loss,
                    pg_loss,
                    v_loss,
                    cost_v_loss,
                    entropy_loss,
                    approx_kl,
                    grads,
                )

            agent_state, (
                loss,
                pg_loss,
                v_loss,
                cost_v_loss,
                entropy_loss,
                approx_kl,
                grads,
            ) = jax.lax.scan(update_minibatch, agent_state, shuffled_storage)
            return (agent_state, key), (
                loss,
                pg_loss,
                v_loss,
                cost_v_loss,
                entropy_loss,
                approx_kl,
                grads,
            )

        (agent_state, key), (
            loss,
            pg_loss,
            v_loss,
            cost_v_loss,
            entropy_loss,
            approx_kl,
            grads,
        ) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=args.update_epochs
        )
        return (
            rollout_state._replace(agent_state=agent_state, key=key),
            loss,
            pg_loss,
            v_loss,
            cost_v_loss,
            entropy_loss,
            approx_kl,
        )

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    handle, (next_obs, info) = envs.reset()
    next_obs = next_obs.astype(jnp.float32)
    next_done = jnp.zeros(args.num_envs, dtype=jax.numpy.bool_)
    next_truncated = jnp.zeros(args.num_envs, dtype=jax.numpy.bool_)
    # based on https://github.dev/google/evojax/blob/0625d875262011d8e1b6aa32566b236f44b4da66/evojax/sim_mgr.py

    options = CheckpointManagerOptions(keep_period=1)


    def step_once(carry, step, env_step_fn):
        agent_state, episode_stats, obs, done, truncated, key, handle = carry
        obs = obs.astype(jnp.float32)
        action, logprob, value, cost_value, key = get_action_and_value(
            agent_state, obs, key
        )
        (
            episode_stats,
            handle,
            (next_obs, reward, cost, next_done, next_truncated, _),
        ) = env_step_fn(episode_stats, handle, action)
        next_obs = next_obs.astype(jnp.float32)
        action = action.astype(jnp.float32)
        reward=reward.astype(jnp.float32)
        storage = Storage(
            obs=obs,
            actions=action,
            logprobs=logprob,
            dones=done,
            truncated=truncated,
            reward=RewardMetrics(
                values=value,
                rewards=reward,
                returns=jnp.zeros_like(reward),
                advantages=jnp.zeros_like(reward),
            ),
            cost=RewardMetrics(
                values=cost_value,
                rewards=cost,
                returns=jnp.zeros_like(reward),
                advantages=jnp.zeros_like(reward),
            ),
        )
        return (
            (
                agent_state,
                episode_stats,
                next_obs,
                next_done,
                next_truncated,
                key,
                handle,
            ),
            storage,
        )

    def rollout(
        agent_state,
        episode_stats,
        next_obs,
        next_done,
        next_truncated,
        key,
        handle,
        step_once_fn,
        max_steps,
    ):
        (
            agent_state,
            episode_stats,
            next_obs,
            next_done,
            next_truncated,
            key,
            handle,
        ), storage = jax.lax.scan(
            step_once_fn,
            (
                agent_state,
                episode_stats,
                next_obs,
                next_done,
                next_truncated,
                key,
                handle,
            ),
            (),
            max_steps,
        )
        return (
            RolloutState(
                agent_state,
                episode_stats,
                next_obs,
                next_done,
                next_truncated,
                key,
                handle,
            ),
            storage,
        )

    rollout = partial(
        rollout,
        step_once_fn=partial(step_once, env_step_fn=step_env_wrappeed),
        max_steps=args.num_steps,
    )
    rollout_state = RolloutState(
        agent_state, episode_stats, next_obs, next_done, next_truncated, key, handle
    )

    for update in range(1, args.num_updates + 1):
        update_time_start = time.time()
        # with jax.disable_jit():
        rollout_state, storage = rollout(*rollout_state)
        global_step += args.num_steps * args.num_envs
        avg_episodic_cost = np.mean(
            jax.device_get(rollout_state.episode_stats.returned_episode_costs)
        )
        rollout_state = rollout_state._replace(
            agent_state=rollout_state.agent_state.replace(
                lagrange=rollout_state.agent_state.lagrange.update_lagrange_multiplier(
                    avg_episodic_cost
                )
            )
        )
        # with jax.disable_jit():
        (
            rollout_state,
            loss,
            pg_loss,
            v_loss,
            cost_v_loss,
            entropy_loss,
            approx_kl,
        ) = update_ppo(rollout_state, storage)
        avg_episodic_return = np.mean(
            jax.device_get(rollout_state.episode_stats.returned_episode_returns)
        )
        avg_episodic_real_costs = np.mean(
            jax.device_get(rollout_state.episode_stats.returned_episode_real_costs)
        )
        avg_episodic_length = np.mean(
            jax.device_get(rollout_state.episode_stats.returned_episode_lengths)
        )
        log_data={
                    "global_step": global_step,
                    "avg_episodic_return": avg_episodic_return,
                    "avg_episodic_cost": avg_episodic_cost,
                    "v_loss": v_loss.mean(),
                    "cost_v_loss": cost_v_loss.mean(),
                    "pg_loss": pg_loss.mean(),
                    "real_cost": avg_episodic_real_costs,
                    "approx_kl": approx_kl.mean(),
                    "entropy_loss": entropy_loss.mean(),
                    "avg_episodic_length": avg_episodic_length,
                    "lagrange": rollout_state.agent_state.lagrange.state.params,
                    "SPS":int(args.num_steps * args.num_envs / (time.time() - update_time_start))
                }
        for k, v in log_data.items():
            print(k, v, global_step)
        # Log the data if args.track is True
        if args.track:
            # Assuming you have imported the required libraries (e.g., wandb)
            wandb.log({"global_step": global_step, **log_data})
    ckpt = rollout_state
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(f"tmp/{run_name}", ckpt, save_args=save_args)
    # envs.close()
