# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
# Implementation adapted from https://github.com/araffin/sbx
import argparse
import os

# os.environ["JAX_PLATFORM_NAME"] = "cpu"
import random
import time
from dataclasses import dataclass
from distutils.util import strtobool
from functools import partial
from typing import Sequence
import flax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
from jax import config
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
import optax
from flax.training import orbax_utils
import orbax.checkpoint
import collections

# import pybullet_envs  # noqa
from jax.experimental import checkify  # noqa
from flax.training.train_state import TrainState
import dejax

from icrl.env.cost_env import make_cost_env_no_xla

# Add progress bar if available
try:
    from tqdm.rich import tqdm
except ImportError:
    tqdm = None


@flax.struct.dataclass
class BatchData:
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="icrl-benchmark",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--eval-freq", type=int, default=-1,
        help="evaluate the agent every `eval_freq` steps (if negative, no evaluation)")
    parser.add_argument("--n-eval-episodes", type=int, default=10,
        help="number of episodes to use for evaluation")
    parser.add_argument("--n-eval-envs", type=int, default=5,
        help="number of environments for evaluation")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="AntWithObstacle",
        help="the id of the environment to be used")
    parser.add_argument("--expert-data", type=str,
        help="the name of the expert data file")
    parser.add_argument("--update-period", type=int, default=500)
    parser.add_argument("--expert-size", type=int, default=5000)
    parser.add_argument("--decay-schedule", type=str, default="linear",
        help="the reward alpha parameter")
    parser.add_argument("--anti-cof", type=float, default=-jnp.inf,
        help="the reward alpha parameter")
    parser.add_argument("--reward-alpha", type=float, default=0.0,
        help="the reward alpha parameter")
    parser.add_argument("--total-timesteps", type=int, default=100000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--l1-ratio", type=float, default=0.1,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-4,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--n-critics", type=int, default=2,
        help="the number of critic networks")
    parser.add_argument("--policy-frequency", type=int, default=1,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--alpha", type=float, default=0.001,
        help="entropy regularization coefficient")
    parser.add_argument("--cost-max", type=float, default=0,
        help="entropy regularization coefficient")
    parser.add_argument("--cost-min", type=float, default=-jnp.inf,
        help="entropy regularization coefficient")
    parser.add_argument("--lamb", type=float, default=0.6,
        help="entropy regularization coefficient")
    parser.add_argument("--reg-cof", type=float, default=0.05,
        help="entropy regularization coefficient")
    parser.add_argument('--mean-V-update', type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=False,)
    parser.add_argument('--update-buffer', type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=False,)
    parser.add_argument('--update-V', type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=False,)
    parser.add_argument('--stable-update', type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=False,)
    parser.add_argument('--use-q', type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=False,)
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=False,
        help="automatic tuning of the entropy coefficient")
    args = parser.parse_args()
    # fmt: on
    if args.expert_data is None:
        args.expert_data = "expert_data/" + args.env_id
    return args


class Critic(nn.Module):
    n_units: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([x, action], -1)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(
            1,
            1,
            kernel_init=orthogonal((0.01)),
            bias_init=constant(0.0),
        )(x)
        return x


class ValueCritic(nn.Module):
    n_units: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(
            1,
            kernel_init=orthogonal((0.01)),
            bias_init=constant(0.0),
        )(x)
        return x


class VectorCritic(nn.Module):
    n_units: int = 256
    n_critics: int = 2

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        # Idea taken from https://github.com/perrin-isir/xpag
        # Similar to https://github.com/tinkoff-ai/CORL for PyTorch
        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            n_units=self.n_units,
        )(obs, action)
        return q_values


class Actor(nn.Module):
    action_dim: Sequence[int]
    n_units: int = 256
    log_std_min: float = -20
    log_std_max: float = -5

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


class RLTrainState(TrainState):
    target_params: flax.core.FrozenDict = None


@partial(jax.jit, static_argnames="actor")
def sample_action(
    actor: Actor,
    actor_state: TrainState,
    observations: jnp.ndarray,
    key: jax.Array,
) -> jnp.array:
    key, subkey = jax.random.split(key, 2)
    mean, log_std = actor.apply(actor_state.params, observations)
    action_std = jnp.exp(log_std)
    gaussian_action = mean + action_std * jax.random.normal(subkey, shape=mean.shape)
    action = jnp.tanh(gaussian_action)
    return action, key


@jax.jit
def sample_action_and_log_prob(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    subkey: jax.Array,
):
    action_std = jnp.exp(log_std)
    gaussian_action = mean + action_std * jax.random.normal(
        subkey, shape=mean.shape, dtype=jnp.float32
    )
    log_prob = (
        -0.5 * ((gaussian_action - mean) / action_std) ** 2
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - log_std
    )
    log_prob = log_prob.sum(axis=1)
    action = jnp.tanh(gaussian_action)
    log_prob -= jnp.sum(jnp.log((1 - action**2) + 1e-6), 1)
    return action, log_prob


@jax.jit
def action_log_prob(
    action: jnp.ndarray,
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
):
    action = action.clip(-1 + 1e-6, 1 - 1e-6)
    action_std = jnp.exp(log_std)
    gaussian_action = jnp.arctanh(action)
    log_prob = (
        -0.5 * ((gaussian_action - mean) / action_std) ** 2
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - log_std
    )
    log_prob -= jnp.log((1 - action**2) + 1e-6)
    return log_prob.sum(-1)


@partial(jax.jit, static_argnames="actor")
def select_action(
    actor: Actor, actor_state: TrainState, observations: jnp.ndarray
) -> jnp.array:
    return actor.apply(actor_state.params, observations)[0]


def scale_action(action_space, action) -> np.ndarray:
    """
    Rescale the action from [low, high] to [-1, 1]
    (no need for symmetric action space)

    :param action: Action to scale
    :return: Scaled action
    """
    low, high = action_space.low, action_space.high
    return 2.0 * ((action - low) / (high - low)) - 1.0


def unscale_action(action_space, scaled_action) -> np.ndarray:
    """
    Rescale the action from [-1, 1] to [low, high]
    (no need for symmetric action space)

    :param scaled_action: Action to un-scale
    """
    low, high = action_space.low, action_space.high
    return low + (0.5 * (scaled_action + 1.0) * (high - low))


class EntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_ent_coef = self.param(
            "log_ent_coef",
            init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)),
        )
        return jnp.exp(log_ent_coef)


@jax.jit
def soft_update(tau: float, qf_state: RLTrainState) -> RLTrainState:
    qf_state = qf_state.replace(
        target_params=optax.incremental_update(
            qf_state.params, qf_state.target_params, tau
        )
    )
    return qf_state


def l2_loss(x):
    return (x**2).mean()

def plot_vf_state(vf_state, obs,position):
    xs = position
    y = vf.apply(vf_state.params, obs * single_mask)
    plt.scatter(xs, y, color="b", alpha=0.1)
def plot_state_whole(vf_state, qf_state, obs, action):
    plots_num = obs.shape[-1]
    fig, axes = plt.subplots(nrows=3, ncols=(plots_num + 2) // 3, figsize=(30, 10))
    y = qf.apply(qf_state.target_params, obs, action).reshape(-1) - vf.apply(
        vf_state.params, obs * single_mask
    ).reshape(-1)
    y = y.clip(args.cost_min, args.cost_max)
    for i in range(plots_num):
        xs = obs[:, i]
        ax = axes.flatten()[i]
        ax.scatter(xs, y, color="b", alpha=0.1)
        bins = np.linspace(xs.min(), xs.max(), 10)
        inds = np.digitize(xs, bins)

        # Calculate mean for each section
        clip_means = [y[inds == i].mean() for i in range(1, len(bins))]
        # means = [y[inds == i].mean() for i in range(1, len(bins))]
        # Plot graph
        ax.plot(bins[1:], clip_means, color="r", alpha=0.9)
        # ax.plot(bins[1:], means, color="r", alpha=0.9,label="unclip")
        ax.set_title("Subplot " + str(i + 1))


def plot_vf_state_whole(vf_state, obs):
    plots_num = obs.shape[-1]
    fig, axes = plt.subplots(nrows=3, ncols=(plots_num + 2) // 3, figsize=(30, 10))
    y = vf.apply(vf_state.params, obs * single_mask)
    for i in range(plots_num):
        xs = obs[:, i]
        ax = axes.flatten()[i]
        ax.scatter(xs, y, color="b", alpha=0.1)
        bins = np.linspace(xs.min(), xs.max(), 10)
        inds = np.digitize(xs, bins)

        # Calculate mean for each section
        means = [y[inds == i].mean() for i in range(1, len(bins))]
        # Plot graph
        ax.plot(bins[1:], means, color="r", alpha=0.9)
        ax.set_title("Subplot " + str(i + 1))


def plot_vf_state_and_save(vf_state, save_path):
    plot_vf_state(vf_state)
    dirname = os.path.dirname(save_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    plt.savefig(save_path)
    plt.close()


@jax.jit
def get_log_prob(params, actions, observations):
    mean, log_std = actor.apply(params, observations)
    log_prob = action_log_prob(actions, mean, log_std)
    return log_prob.mean()


@checkify.checkify
def reward_train_step(agent_state):
    callback_log = {}

    def update_reward_critic(
        qf_state: RLTrainState,
        vf_state: RLTrainState,
        Expert_Batch: BatchData,
        Policy_Batch: BatchData,
        reward_qf_state: RLTrainState,
        reward_vf_state: RLTrainState,
        key: jax.Array,
    ):
        key, subkey = jax.random.split(key, 2)
        Batch = jax.tree_map(
            lambda x, y: jnp.concatenate([x, y]), Policy_Batch, Expert_Batch
        )
        next_values = vf.apply(reward_vf_state.params, Batch.next_observations).reshape(
            -1
        )

        def mse_loss(params):
            current_reward_Q = qf.apply(
                params, Batch.observations, Batch.actions
            ).reshape(-1)
            discounted_rewards = Batch.rewards + (1 - Batch.dones) * args.gamma * (
                next_values
            )
            loss = ((current_reward_Q - discounted_rewards) ** 2).mean()
            return loss, current_reward_Q.mean()

        (reward_qf_loss_value, reward_qf_values), grads = jax.value_and_grad(
            mse_loss, has_aux=True
        )(reward_qf_state.params)
        reward_qf_state = reward_qf_state.apply_gradients(grads=grads)
        callback_log["reward_qf_loss_value"] = reward_qf_loss_value
        callback_log["reward_qf_values"] = reward_qf_values
        return reward_qf_state, key

    def update_reward_value_critic(
        Expert_Batch: BatchData,
        Policy_Batch: BatchData,
        reward_qf_state: RLTrainState,
        reward_vf_state: RLTrainState,
        key: jax.Array,
    ):
        Batch = jax.tree_map(
            lambda x, y: jnp.concatenate([x, y]), Policy_Batch, Expert_Batch
        )
        current_reward_Q = qf.apply(
            reward_qf_state.target_params, Batch.observations, Batch.actions
        ).reshape(-1)
        lamb = args.lamb
        current_Q = qf.apply(
            qf_state.params, Batch.observations, Batch.actions
        ).reshape(-1)
        current_V = vf.apply(vf_state.params, Batch.observations * single_mask).reshape(
            -1
        )
        next_V = (1 - Batch.dones) * vf.apply(
            vf_state.params, Batch.next_observations * single_mask
        ).reshape(-1)
        policy_length = len(Policy_Batch.observations)

        def mse_loss(params):
            current_reward_V = vf.apply(params, Batch.observations).reshape(-1)
            if args.use_q:
                y = (
                    current_reward_Q
                    - current_reward_V
                    + (current_Q - current_V).clip(args.cost_min, args.cost_max)
                )
            else:
                y = (
                    current_reward_Q
                    - current_reward_V
                    + (next_V - current_V).clip(args.cost_min, args.cost_max)
                )

            loss = (1 - lamb) * (current_reward_V).mean() + lamb * ((y).clip(0)).mean()
            return loss, (
                current_reward_V[:policy_length].mean(),
                current_reward_V[policy_length:].mean(),
                current_reward_Q[:policy_length].mean(),
                current_reward_Q[policy_length:].mean(),
            )

        (
            rewad_vf_loss_value,
            (
                policy_reward_vf_values,
                expert_reward_vf_values,
                policy_reward_qf_values,
                expert_reward_qf_values,
            ),
        ), grads = jax.value_and_grad(mse_loss, has_aux=True)(reward_vf_state.params)
        reward_vf_state = reward_vf_state.apply_gradients(grads=grads)
        callback_log["reward_vf_loss_value"] = rewad_vf_loss_value
        callback_log["policy_reward_vf_values"] = policy_reward_vf_values
        callback_log["expert_reward_vf_values"] = expert_reward_vf_values
        callback_log["policy_reward_qf_values"] = policy_reward_qf_values
        callback_log["expert_reward_qf_values"] = expert_reward_qf_values
        return reward_vf_state, key

    def update_actor(
        actor_state: RLTrainState,
        qf_state: RLTrainState,
        vf_state: RLTrainState,
        Policy_Batch: BatchData,
        reward_qf_state: RLTrainState,
        reward_vf_state: RLTrainState,
        key: jax.Array,
    ):
        Batch = jax.tree_map(
            lambda x, y: jnp.concatenate([x, y]), Policy_Batch, Expert_Batch
        )
        current_reward_V = vf.apply(reward_vf_state.params, Batch.observations).reshape(
            -1
        )
        current_reward_Q = qf.apply(
            reward_qf_state.target_params, Batch.observations, Batch.actions
        ).reshape(-1)

        vf_pi = vf.apply(vf_state.params, Batch.observations * single_mask).reshape(-1)
        next_vf_pi = (1 - Batch.dones) * vf.apply(
            vf_state.params, Batch.next_observations * single_mask
        ).reshape(-1)
        qf_pi = qf.apply(qf_state.params, Batch.observations, Batch.actions).reshape(-1)
        # Take min among all critics
        min_qf_pi = jnp.min(qf_pi, axis=0).reshape(-1)
        key, subkey = jax.random.split(key, 2)

        def actor_loss(params):
            mean, log_std = actor.apply(params, Batch.observations)
            log_prob = action_log_prob(Batch.actions, mean, log_std)
            if args.use_q:
                y = (
                    current_reward_Q
                    - current_reward_V
                    + (min_qf_pi - vf_pi).clip(args.cost_min, args.cost_max)
                )
            else:
                y = (
                    current_reward_Q
                    - current_reward_V
                    + (next_vf_pi - vf_pi).clip(args.cost_min, args.cost_max)
                )

            callback_log["policy_td_mean"] = jax.lax.stop_gradient(
                y[: len(Policy_Batch.observations)].mean()
            )
            callback_log["expert_td_mean"] = jax.lax.stop_gradient(
                y[len(Policy_Batch.observations) :].mean()
            )
            callback_log["policy_log_std"] = jax.lax.stop_gradient(
                log_std[: len(Policy_Batch.observations)].mean()
            )
            callback_log["expert_log_std"] = jax.lax.stop_gradient(
                log_std[len(Policy_Batch.observations) :].mean()
            )
            w = omega_star(y)
            actor_loss = -(
                # w.clip(max=100.0).flatten() * log_prob
                # w.flatten()
                jnp.exp(y * 10.0).clip(max=100.0)
                * log_prob
            ).mean()
            return actor_loss, -log_prob.mean()

        (actor_loss_value, entropy), grads = jax.value_and_grad(
            actor_loss, has_aux=True
        )(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        callback_log["actor_loss_value"] = actor_loss_value
        callback_log["entropy"] = entropy
        return actor_state

    (
        buffer_state,
        actor_state,
        vf_state,
        qf_state,
        reward_qf_state,
        reward_vf_state,
        key,
    ) = agent_state
    sample_key, key = jax.random.split(key, 2)
    observations, next_observations, actions, rewards, dones = buffer.sample_fn(
        buffer_state, sample_key, args.batch_size
    )
    sample_key, key = jax.random.split(key, 2)
    Policy_Batch = BatchData(
        observations,
        actions,
        next_observations,
        rewards,
        dones,
    )
    expert_indice = jax.random.randint(
        sample_key, minval=0, maxval=len(expert_obs), shape=(args.batch_size,)
    )
    Expert_Batch = BatchData(
        expert_obs[expert_indice],
        expert_action[expert_indice],
        expert_next_obs[expert_indice],
        expert_rewards[expert_indice],
        expert_dones[expert_indice],
    )
    reward_qf_state, key = update_reward_critic(
        qf_state,
        vf_state,
        Expert_Batch,
        Policy_Batch,
        reward_qf_state,
        reward_vf_state,
        key,
    )
    reward_vf_state, key = update_reward_value_critic(
        Expert_Batch,
        Policy_Batch,
        reward_qf_state,
        reward_vf_state,
        key,
    )
    actor_state = update_actor(
        actor_state,
        qf_state,
        vf_state,
        Policy_Batch,
        reward_qf_state,
        reward_vf_state,
        key,
    )

    reward_qf_state = soft_update(args.tau, reward_qf_state)

    return (
        AgentState(
            buffer_state,
            actor_state,
            vf_state,
            qf_state,
            reward_qf_state,
            reward_vf_state,
            key,
        ),
        callback_log,
    )


@checkify.checkify
def train_step(agent_state, anti_buffer_state):
    callback_log = {}

    def update_value_critic(
        qf_state: RLTrainState,
        vf_state: RLTrainState,
        Policy_Batch: BatchData,
        Expert_Batch: BatchData,
        key: jax.Array,
    ):
        key, subkey = jax.random.split(key, 2)
        Batch = jax.tree_map(
            lambda x, y: jnp.concatenate([x, y]), Policy_Batch, Expert_Batch
        )

        current_Q = qf.apply(
            qf_state.target_params, Batch.observations, Batch.actions
        ).clip(max=0)

        def mse_loss(params):
            expert_V = vf.apply(
                params, Expert_Batch.observations * single_mask
            ).reshape(-1)
            next_V = vf.apply(params, Batch.next_observations * single_mask).reshape(-1)

            discounted_rewards = (1 - Batch.dones) * args.gamma * next_V
            if args.mean_V_update:
                loss = ((current_Q.reshape(-1) - discounted_rewards) ** 2).mean()
            else:
                loss = (
                    ((discounted_rewards - current_Q.reshape(-1)).clip(0)).mean()
                    - next_V.clip(max=0).mean() * 0.01
                    + ((expert_V - 0) ** 2).mean()
                )
            return loss, current_Q.mean()

        (vf_loss_value, qf_values), grads = jax.value_and_grad(mse_loss, has_aux=True)(
            vf_state.params
        )
        vf_state = vf_state.apply_gradients(grads=grads)

        callback_log["vf_loss_value"] = vf_loss_value
        callback_log["qf_values"] = qf_values

        return vf_state, key

    def update_critic(
        qf_state: RLTrainState,
        vf_state: RLTrainState,
        Policy_Batch: BatchData,
        Expert_Batch: BatchData,
        reward_qf_state: RLTrainState,
        reward_vf_state: RLTrainState,
        key: jax.Array,
    ):
        Batch = jax.tree_map(
            lambda x, y: jnp.concatenate([x, y]), Policy_Batch, Expert_Batch
        )

        current_expert_reward_V = vf.apply(
            reward_vf_state.params, Expert_Batch.observations
        ).reshape(-1)
        current_expert_reward_Q = qf.apply(
            reward_qf_state.params,
            Expert_Batch.observations,
            Expert_Batch.actions,
        ).reshape(-1)

        current_anti_reward_Q = qf.apply(
            reward_qf_state.params,
            anti_observations,
            anti_actions,
        ).reshape(-1)
        current_anti_reward_V = vf.apply(
            reward_vf_state.params, anti_observations
        ).reshape(-1)

        current_V = vf.apply(vf_state.params, Batch.observations * single_mask).reshape(
            -1
        )
        anti_current_V = vf.apply(
            vf_state.params, anti_observations * single_mask
        ).reshape(-1)
        next_V = vf.apply(
            vf_state.params, Batch.next_observations * single_mask
        ).reshape(-1)
        expert_next_V = vf.apply(
            vf_state.params, Expert_Batch.next_observations * single_mask
        ).reshape(-1)
        expert_current_V = vf.apply(
            vf_state.params, Expert_Batch.observations * single_mask
        ).reshape(-1)
        anti_Q_target = qf.apply(
            qf_state.target_params, anti_observations, anti_actions
        ).reshape(-1)
        current_expert_Q_target = qf.apply(
            qf_state.target_params, Expert_Batch.observations, Expert_Batch.actions
        ).reshape(-1)

        def mse_loss(params):
            current_Q = qf.apply(params, Batch.observations, Batch.actions).reshape(-1)
            anti_Q = qf.apply(params, anti_observations, anti_actions).reshape(-1)
            current_expert_Q = (1 - Expert_Batch.dones) * qf.apply(
                params, Expert_Batch.observations, Expert_Batch.actions
            ).reshape(-1)

            w = (
                current_expert_reward_Q
                - current_expert_reward_V
                + (
                    current_expert_Q - expert_current_V
                ).clip(args.cost_min, args.cost_max)
            ).mean()
            if args.update_buffer:
                neg_w = - (
                        current_anti_reward_Q
                        - current_anti_reward_V
                        + (
                            anti_Q - anti_current_V
                        )  # .clip(args.cost_min, args.cost_max)
                    ).clip(-args.anti_cof).mean()
                w=w+neg_w
                callback_log['neg_omega_qr']=jax.lax.stop_gradient((current_anti_reward_Q- current_anti_reward_V).mean())
                callback_log['omega_qr']=jax.lax.stop_gradient((current_expert_reward_Q- current_expert_reward_V).mean())
                callback_log['neg_omega_vr']=jax.lax.stop_gradient((current_anti_reward_V).mean())
                callback_log['omega_vr']=jax.lax.stop_gradient((current_expert_reward_V).mean())
                callback_log['neg_omega_qc']=jax.lax.stop_gradient((anti_Q - anti_current_V).mean())
                callback_log['omega_qc']=jax.lax.stop_gradient((current_expert_Q - expert_current_V).mean())
                callback_log['neg_omega_vc']=jax.lax.stop_gradient((anti_current_V).mean())
                callback_log['omega_vc']=jax.lax.stop_gradient((expert_current_V).mean())
                callback_log['neg_omega']=jax.lax.stop_gradient(neg_w.mean())
                callback_log['omega']=jax.lax.stop_gradient(w.mean())
            coeff = args.l1_ratio
            loss = (
                coeff * (-w)
                + (1 - coeff) * (((anti_Q - 0) ** 2).mean())
                # +10*current_Q.clip(0).mean()
                + args.reg_cof * (((current_expert_Q - 0) ** 2).mean())
            )

            return loss, (current_V.mean(), current_Q.mean())

        (qf_loss_value, (vf_values, qf_values)), grads = jax.value_and_grad(
            mse_loss, has_aux=True
        )(qf_state.params)
        qf_state = qf_state.apply_gradients(grads=grads)
        callback_log["qf_loss_value"] = qf_loss_value
        callback_log["qf_values"] = qf_values
        callback_log["vf_values"] = vf_values

        return qf_state, key

    (
        buffer_state,
        actor_state,
        vf_state,
        qf_state,
        reward_qf_state,
        reward_vf_state,
        key,
    ) = agent_state
    sample_key, key = jax.random.split(key, 2)
    observations, next_observations, actions, rewards, dones = buffer.sample_fn(
        buffer_state, sample_key, args.batch_size
    )
    sample_key, key = jax.random.split(key, 2)
    anti_observations, anti_next_observations, anti_actions = anti_buffer.sample_fn(
        anti_buffer_state, sample_key, args.batch_size
    )

    sample_key, key = jax.random.split(key, 2)
    Policy_Batch = BatchData(
        observations,
        actions,
        next_observations,
        rewards,
        dones,
    )
    expert_indice = jax.random.randint(
        sample_key, minval=0, maxval=len(expert_obs), shape=(args.batch_size,)
    )
    Expert_Batch = BatchData(
        expert_obs[expert_indice],
        expert_action[expert_indice],
        expert_next_obs[expert_indice],
        expert_rewards[expert_indice],
        expert_dones[expert_indice],
    )
    qf_state, key = update_critic(
        qf_state,
        vf_state,
        Policy_Batch,
        Expert_Batch,
        reward_qf_state,
        reward_vf_state,
        key,
    )
    vf_state, key = update_value_critic(
        qf_state,
        vf_state,
        Policy_Batch,
        Expert_Batch,
        key,
    )
    qf_state = soft_update(args.tau, qf_state)

    return (
        AgentState(
            buffer_state,
            actor_state,
            vf_state,
            qf_state,
            reward_qf_state,
            reward_vf_state,
            key,
        ),
        callback_log,
    )


@checkify.checkify
def cost_train_step(agent_state):
    callback_log = {}

    def update_reward_critic(
        qf_state: RLTrainState,
        vf_state: RLTrainState,
        Expert_Batch: BatchData,
        Policy_Batch: BatchData,
        reward_qf_state: RLTrainState,
        reward_vf_state: RLTrainState,
        key: jax.random.KeyArray,
    ):
        key, subkey = jax.random.split(key, 2)
        Batch = jax.tree_map(
            lambda x, y: jnp.concatenate([x, y]), Policy_Batch, Expert_Batch
        )
        next_values = vf.apply(reward_vf_state.params, Batch.next_observations).reshape(
            -1
        )
        current_reward_V = vf.apply(
            reward_vf_state.params, Batch.observations * single_mask
        ).reshape(-1)
        current_V = vf.apply(vf_state.params, Batch.observations * single_mask).reshape(
            -1
        )
        next_V = (1 - Batch.dones) * vf.apply(
            vf_state.params, Batch.next_observations * single_mask
        ).reshape(-1)
        cost = -(
            ((next_V - current_V).clip(args.cost_min, args.cost_max) < -1)
            * 1.0
        )
        # cost=0.
        def mse_loss(params):
            current_reward_Q = qf.apply(
                params, Batch.observations, Batch.actions
            ).reshape(-1)
            discounted_rewards = cost + args.gamma * (current_reward_V)
            loss = ((current_reward_Q - discounted_rewards) ** 2).mean()
            return loss, current_reward_Q.mean()

        (reward_qf_loss_value, reward_qf_values), grads = jax.value_and_grad(
            mse_loss, has_aux=True
        )(reward_qf_state.params)
        reward_qf_state = reward_qf_state.apply_gradients(grads=grads)
        callback_log["reward_qf_loss_value"] = reward_qf_loss_value
        callback_log["reward_qf_values"] = reward_qf_values
        return reward_qf_state, key

    def update_reward_value_critic(
        Expert_Batch: BatchData,
        Policy_Batch: BatchData,
        reward_qf_state: RLTrainState,
        reward_vf_state: RLTrainState,
        key: jax.random.KeyArray,
    ):
        Batch = jax.tree_map(
            lambda x, y: jnp.concatenate([x, y]), Policy_Batch, Expert_Batch
        )
        current_reward_Q = qf.apply(
            reward_qf_state.target_params, Batch.observations, Batch.actions
        ).reshape(-1)
        lamb = args.lamb
        current_Q = qf.apply(
            qf_state.params, Batch.observations, Batch.actions
        ).reshape(-1)
        current_V = vf.apply(vf_state.params, Batch.observations * single_mask).reshape(
            -1
        )
        next_V = (1 - Batch.dones) * vf.apply(
            vf_state.params, Batch.next_observations * single_mask
        ).reshape(-1)

        def mse_loss(params):
            current_reward_V = vf.apply(params, Batch.observations).reshape(-1)
            next_reward_V = vf.apply(params, Batch.next_observations).reshape(-1)
            y = current_reward_Q - next_reward_V

            loss = (1 - lamb) * (next_reward_V).mean() + lamb * ((y).clip(0)).mean()
            return loss, (next_reward_V.mean(), current_reward_Q.mean())

        (
            rewad_vf_loss_value,
            (reward_vf_values, reward_qf_values),
        ), grads = jax.value_and_grad(mse_loss, has_aux=True)(reward_vf_state.params)
        reward_vf_state = reward_vf_state.apply_gradients(grads=grads)
        callback_log["reward_vf_loss_value"] = rewad_vf_loss_value
        callback_log["reward_vf_values"] = reward_vf_values
        callback_log["reward_qf_values"] = reward_qf_values
        return reward_vf_state, key

    def update_actor(
        actor_state: RLTrainState,
        qf_state: RLTrainState,
        vf_state: RLTrainState,
        Policy_Batch: BatchData,
        reward_qf_state: RLTrainState,
        reward_vf_state: RLTrainState,
        key: jax.random.KeyArray,
    ):
        Batch = jax.tree_map(
            lambda x, y: jnp.concatenate([x, y]), Policy_Batch, Expert_Batch
        )
        current_reward_V = vf.apply(reward_vf_state.params, Batch.observations).reshape(
            -1
        )
        current_reward_Q = qf.apply(
            reward_qf_state.target_params, Batch.observations, Batch.actions
        ).reshape(-1)

        vf_pi = vf.apply(vf_state.params, Batch.observations * single_mask).reshape(-1)
        next_vf_pi = (1 - Batch.dones) * vf.apply(
            vf_state.params, Batch.next_observations * single_mask
        ).reshape(-1)
        qf_pi = qf.apply(qf_state.params, Batch.observations, Batch.actions).reshape(-1)
        # Take min among all critics
        min_qf_pi = jnp.min(qf_pi, axis=0).reshape(-1)
        key, subkey = jax.random.split(key, 2)

        def actor_loss(params):
            mean, log_std = actor.apply(params, Batch.observations)
            log_prob = action_log_prob(Batch.actions, mean, log_std)
            y = (
                current_reward_Q
                - current_reward_V
                - ((next_vf_pi - vf_pi).clip(args.cost_min, args.cost_max) < -0.1) * 100
            )

            callback_log["policy_td_mean"] = jax.lax.stop_gradient(
                y[: len(Policy_Batch.observations)].mean()
            )
            callback_log["expert_td_mean"] = jax.lax.stop_gradient(
                y[len(Policy_Batch.observations) :].mean()
            )
            callback_log["policy_log_std"] = jax.lax.stop_gradient(
                log_std[: len(Policy_Batch.observations)].mean()
            )
            callback_log["expert_log_std"] = jax.lax.stop_gradient(
                log_std[len(Policy_Batch.observations) :].mean()
            )
            w = omega_star(y)
            actor_loss = -(
                # w.clip(max=100.0).flatten() * log_prob
                # w.flatten()
                jnp.exp(y * 10.0).clip(max=100.0)
                * log_prob
            ).mean()
            return actor_loss, -log_prob.mean()

        (actor_loss_value, entropy), grads = jax.value_and_grad(
            actor_loss, has_aux=True
        )(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        callback_log["actor_loss_value"] = actor_loss_value
        callback_log["entropy"] = entropy
        return actor_state

    (
        buffer_state,
        actor_state,
        vf_state,
        qf_state,
        reward_qf_state,
        reward_vf_state,
        key,
    ) = agent_state
    sample_key, key = jax.random.split(key, 2)
    observations, next_observations, actions, rewards, dones = buffer.sample_fn(
        buffer_state, sample_key, args.batch_size
    )
    sample_key, key = jax.random.split(key, 2)
    Policy_Batch = BatchData(
        observations,
        actions,
        next_observations,
        rewards,
        dones,
    )
    expert_indice = jax.random.randint(
        sample_key, minval=0, maxval=len(expert_obs), shape=(args.batch_size,)
    )
    Expert_Batch = BatchData(
        expert_obs[expert_indice],
        expert_action[expert_indice],
        expert_next_obs[expert_indice],
        expert_rewards[expert_indice],
        expert_dones[expert_indice],
    )
    reward_qf_state, key = update_reward_critic(
        qf_state,
        vf_state,
        Expert_Batch,
        Policy_Batch,
        reward_qf_state,
        reward_vf_state,
        key,
    )
    reward_vf_state, key = update_reward_value_critic(
        Expert_Batch,
        Policy_Batch,
        reward_qf_state,
        reward_vf_state,
        key,
    )
    actor_state = update_actor(
        actor_state,
        qf_state,
        vf_state,
        Policy_Batch,
        reward_qf_state,
        reward_vf_state,
        key,
    )

    reward_qf_state = soft_update(args.tau, reward_qf_state)

    return (
        AgentState(
            buffer_state,
            actor_state,
            vf_state,
            qf_state,
            reward_qf_state,
            reward_vf_state,
            key,
        ),
        callback_log,
    )


AgentState = collections.namedtuple(
    "AgentState",
    [
        "buffer_state",
        "actor_state",
        "vf_state",
        "qf_state",
        "reward_qf_state",
        "reward_vf_state",
        "key",
    ],
)


def eval_policy(envs, agentstate, global_step, round=10):
    reward_deque = []
    cost_deque = []
    length_deque = []
    cum_reward = 0
    cum_cost = 0
    lengths = 0
    obs, info = envs.reset()
    key = agentstate.key
    for i in range(10):
        while True:
            obs = obs.astype(np.float32)
            actions, key = sample_action(actor, agentstate.actor_state, obs, key)
            input_actions = unscale_action(envs.action_space, actions.__array__())
            (next_obs, rewards, dones, truncated, infos) = envs.step(input_actions)
            costs = cost_function(next_obs, rewards, dones, truncated, infos)
            if hasattr(env_module, "reward_function"):
                rewards = env_module.reward_function(
                    next_obs, rewards, dones, truncated, infos
                )
            next_obs = next_obs.astype(np.float32)

            obs = next_obs
            cum_reward += rewards
            cum_cost += costs
            lengths += 1
            if dones.any() or truncated.any():
                reward_deque.append(cum_reward)
                cost_deque.append(cum_cost)
                length_deque.append(lengths)
                cum_reward = 0
                cum_cost = 0
                lengths = 0
                (obs, rewards, dones, truncated, infos) = envs.step(input_actions)
                obs = obs.astype(np.float32)
                break
    log_data = {
        "charts/mean_return": np.average(reward_deque),
        "charts/mean_cost": np.average(cost_deque),
        "charts/mean_length": np.average(length_deque),
    }
    for k, v in log_data.items():
        print(k, v, global_step)
    # Log the data if args.track is True
    if args.track:
        # Assuming you have imported the required libraries (e.g., wandb)
        wandb.log({"global_step": global_step, **log_data})


def f_div(x):
    return (x - 1) ** 2


def omega_star(y):
    return (y / 2 + 1).clip(0)


def fp_star(y):
    return omega_star(y) * y - f_div(omega_star(y))


def split_data(expert_data):
    expert_action = jnp.array(expert_data["action"]).astype(jnp.float32)
    expert_obs = jnp.array(expert_data["traj_obs"]).astype(jnp.float32)
    expert_next_obs = jnp.array(expert_data["traj_next_obs"]).astype(jnp.float32)
    expert_dones = jnp.array(expert_data["dones"]).astype(jnp.float32)
    expert_rewards = jnp.array(expert_data["rewards"]).astype(jnp.float32)
    return expert_obs, expert_next_obs, expert_action, expert_rewards, expert_dones


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time()*1000)}"
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
    # Use a separate key, so running with/without eval doesn't affect the results
    eval_key = jax.random.PRNGKey(args.seed)
    envs, cost_function, env_module = make_cost_env_no_xla(
        args.env_id, 1, args.seed, get_module=True
    )

    # Create networks
    key, actor_key, qf_key, ent_key = jax.random.split(key, 4)

    obs = jnp.array([envs.observation_space.sample()]).astype(jnp.float32)
    action = jnp.array([envs.action_space.sample()]).astype(jnp.float32)
    if hasattr(env_module, "single_mask"):
        single_mask = env_module.single_mask
    else:
        single_mask = np.ones(len(obs[0]))
    buffer = dejax.uniform_replay(max_size=args.buffer_size)
    buffer_state = buffer.init_fn(
        (obs[0], obs[0], action[0], jnp.array(0.0, dtype=jnp.float32), jnp.array(True))
    )

    anti_buffer = dejax.uniform_replay(max_size=args.buffer_size)
    anti_buffer_state = anti_buffer.init_fn((obs[0], obs[0], action[0]))

    actor = Actor(action_dim=np.prod(envs.action_space.shape))

    update_period = args.update_period
    total_step = int((args.total_timesteps) / update_period)

    if args.decay_schedule == "linear":
        schedule_fn = optax.linear_onecycle_schedule(
            int(args.total_timesteps), -args.policy_lr
        )
        actor_optimiser = optax.chain(
            optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn)
        )
        schedule_fn = optax.linear_onecycle_schedule(
            int(args.total_timesteps), -args.policy_lr
        )
        q_optimiser = optax.chain(
            optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn)
        )
        v_optimiser = optax.chain(
            optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn)
        )
    else:
        actor_optimiser = optax.adam(learning_rate=args.policy_lr)
        q_optimiser = optax.adam(learning_rate=args.q_lr)
        v_optimiser = optax.adam(learning_rate=args.q_lr)

    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs),
        tx=actor_optimiser,
    )

    reward_qf = VectorCritic(n_critics=args.n_critics)

    qf_state = RLTrainState.create(
        apply_fn=reward_qf.apply,
        params=reward_qf.init({"params": qf_key}, obs, action),
        target_params=reward_qf.init({"params": qf_key}, obs, action),
        tx=q_optimiser,
    )

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    state = {"qf_state": qf_state, "actor_state": actor_state}
    # if args.reward_alpha!=0.0:
    _expert_data = checkpointer.restore(args.expert_data)
    if args.expert_size != 0:
        random_indice = np.random.permutation(len(_expert_data["traj_obs"]))[
            : args.expert_size
        ]
        expert_data = jax.tree_map(lambda x: x[random_indice], _expert_data)
    else:
        expert_data = _expert_data
    (
        expert_obs,
        expert_next_obs,
        expert_action,
        expert_rewards,
        expert_dones,
    ) = split_data(expert_data)

    vf = ValueCritic()
    vf.apply = jax.jit(vf.apply)
    vf_state = TrainState.create(
        apply_fn=vf.apply,
        params=vf.init(actor_key, obs),
        tx=v_optimiser,
    )

    qf = VectorCritic(n_critics=1)

    qf_state = RLTrainState.create(
        apply_fn=qf.apply,
        params=qf.init({"params": qf_key}, obs, action),
        target_params=qf.init({"params": qf_key}, obs, action),
        tx=q_optimiser,
    )

    qf = VectorCritic(n_critics=1)

    reward_qf_state = RLTrainState.create(
        apply_fn=qf.apply,
        params=qf.init({"params": qf_key}, obs, action),
        target_params=qf.init({"params": qf_key}, obs, action),
        tx=q_optimiser,
    )

    reward_vf = ValueCritic()
    reward_vf_state = TrainState.create(
        apply_fn=vf.apply,
        params=vf.init(actor_key, obs),
        tx=v_optimiser,
    )
    cost_qf_state = RLTrainState.create(
        apply_fn=qf.apply,
        params=qf.init({"params": qf_key}, obs, action),
        target_params=qf.init({"params": qf_key}, obs, action),
        tx=optax.adam(learning_rate=args.q_lr),
    )
    cost_vf_state = TrainState.create(
        apply_fn=vf.apply,
        params=vf.init(actor_key, obs),
        tx=optax.adam(learning_rate=args.policy_lr),
    )

    # Automatic entropy tuning
    if args.autotune:
        ent_coef = EntropyCoef(ent_coef_init=1.0)
        target_entropy = -np.prod(envs.action_space.shape).astype(np.float32)
        ent_coef_state = TrainState.create(
            apply_fn=ent_coef.apply,
            params=ent_coef.init(ent_key)["params"],
            tx=optax.adam(learning_rate=args.q_lr),
        )

    else:
        ent_coef_value = jnp.array(args.alpha)

    # Define update functions here to limit the need for static argname
    def get_V(actor_state, params, observations, subkey):
        mean, log_std = actor.apply(actor_state.params, observations)
        expert_next_state_actions, log_prob = sample_action_and_log_prob(
            mean, log_std, subkey
        )

        qf_next_target_values = qf.apply(
            params, observations, expert_next_state_actions
        )
        return qf_next_target_values - jnp.exp(0.001) * log_prob.reshape(-1, 1)

    # Define update functions here to limit the need for static argname

    @jax.jit
    def update_temperature(ent_coef_state: TrainState, entropy: float):
        def temperature_loss(params):
            ent_coef_value = ent_coef.apply({"params": params})
            ent_coef_loss = ent_coef_value * (entropy - target_entropy).mean()
            return ent_coef_loss

        ent_coef_loss, grads = jax.value_and_grad(temperature_loss)(
            ent_coef_state.params
        )
        ent_coef_state = ent_coef_state.apply_gradients(grads=grads)

        return ent_coef_state, ent_coef_loss

    # add=jax.jit(buffer.add_fn,donate_argnums=(0,))
    @partial(jax.jit, donate_argnums=(0))
    def rollout(buffer_state, obs, next_obs, actions, rewards, dones):
        rewards = jnp.nan_to_num(rewards.astype(jnp.float32), neginf=0, posinf=0)
        buffer_state = buffer.add_fn(
            buffer_state, (obs, next_obs, actions, rewards, dones)
        )
        return buffer_state

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset()
    obs = obs.astype(np.float32)
    # Display progress bar if available
    cum_reward = 0
    cum_cost = 0
    lengths = 0
    agentstate = AgentState(
        buffer_state,
        actor_state,
        vf_state,
        qf_state,
        reward_qf_state,
        reward_vf_state,
        key,
    )
    dataset = env_module.offline_dataset()
    agentstate = agentstate._replace(
        buffer_state=buffer.add_batch_fn(
            agentstate.buffer_state,
            (
                dataset["observations"][..., : obs.shape[-1]].astype(jnp.float32),
                dataset["next_observations"][..., : obs.shape[-1]].astype(jnp.float32),
                dataset["actions"].astype(jnp.float32),
                dataset["rewards"].astype(jnp.float32),
                (dataset["terminals"] * (1 - dataset["timeouts"])).astype(jnp.bool_),
            ),
        )
    )

    _train_step = jax.jit(train_step)
    _reward_train_step = jax.jit(reward_train_step)
    _cost_train_step = jax.jit(cost_train_step)
    # _max_reward_train_step= jax.jit(max_reward_train_step)
    # agentstate = agentstate._replace(
    #     buffer_state=buffer.add_batch_fn(
    #         agentstate.buffer_state,
    #         split_data(expert_data),
    #     )
    # )

    def train_step_body(carry, step):
        agentstate, anti_buffer_state = carry
        _, (agentstate, callback_log) = _train_step(agentstate, anti_buffer_state)
        return (agentstate, anti_buffer_state), callback_log

    def reward_train_step_body(carry, step):
        agentstate = carry
        _, (agentstate, callback_log) = _reward_train_step(agentstate)
        return agentstate, callback_log

    def cost_train_step_body(carry, step):
        agentstate = carry
        _, (agentstate, callback_log) = _cost_train_step(agentstate)
        return agentstate, callback_log

    agentstate, reward_callback_log = jax.lax.scan(
        reward_train_step_body,
        (agentstate),
        (),
        length=10000,
    )
    old_reward_vf_loss_value = reward_callback_log["reward_vf_loss_value"].mean()

    # @partial(jax.jit, donate_argnums=(0))
    # def update_anti_actions(anti_buffer_state,actor_state):
    #     new_action=sample_action(actor, actor_state, expert_obs, key)[0]
    #     action_diff=jnp.abs((new_action-expert_action)).sum(-1)
    #     index=jnp.argpartition(action_diff,-100)[-100:]
    #     anti_buffer_state=anti_buffer.add_batch_fn(anti_buffer_state,(expert_obs[index],expert_next_obs[index],new_action[index]))
    #     return anti_buffer_state,action_diff[index].mean()

    @partial(jax.jit, donate_argnums=(0))
    def update_anti_actions(anti_buffer_state, actor_state):
        mean, log_std = actor.apply(actor_state.params, expert_obs)
        log_prob = action_log_prob(expert_action, mean, log_std)
        index = jnp.argpartition(log_prob, 500)[:500]
        new_action = jnp.tanh(mean)
        anti_buffer_state = anti_buffer.add_batch_fn(
            anti_buffer_state,
            (expert_obs[index], expert_next_obs[index], new_action[index]),
        )
        return (
            anti_buffer_state,
            log_prob[index].mean(),
            log_std[index].mean(),
            ((new_action - expert_action)[index] ** 2).mean(),
        )

    (
        anti_buffer_state,
        least_log_mean,
        action_diff_log,
        action_diff,
    ) = update_anti_actions(anti_buffer_state, agentstate.actor_state)
    reward_gap = float("inf")
    # total_obs = dataset["observations"][..., : obs.shape[-1]].astype(jnp.float32)
    # total_action = dataset["actions"].astype(jnp.float32)
    # total_infos = dataset['infos']['x_velocity'][1:]
    for global_step in range(total_step):
        if global_step % (25000 // update_period) == 0:
            eval_policy(envs, agentstate, global_step * args.update_period)
            # new_agentstate, reward_callback_log = jax.lax.scan(
            # cost_train_step_body,
            # (
            #     AgentState(
            #         agentstate.buffer_state,
            #         agentstate.actor_state,
            #         agentstate.vf_state,
            #         agentstate.qf_state,
            #         cost_qf_state,
            #         cost_vf_state,
            #         key,
            #     )
            # ),
            # (),
            # length=10000,
            # )
            # plot_vf_state(
            #     new_agentstate.reward_vf_state, total_obs, total_infos
            # )
            # wandb.log({"cost_plot": wandb.Image(plt)})
            # plt.close()
        # if global_step % (25000 // update_period) == 0:
        #     state = {
        #         "qf_state": agentstate.qf_state,
        #         "vf_state": agentstate.vf_state,
        #         "reward_qf_state": agentstate.reward_qf_state,
        #         "reward_vf_state": agentstate.reward_vf_state,
        #         "actor_state": agentstate.actor_state,
        #     }
        #     save_args = orbax_utils.save_args_from_target(state)
        #     checkpointer.save(f"debugpolicy/{args.env_id}/{run_name}_{global_step}", state, save_args=save_args)
            # agentstate=agentstate._replace(    qf_state = RLTrainState.create(
            #         apply_fn=qf.apply,
            #         params=qf.init({"params": qf_key}, obs, action),
            #         target_params=qf.init({"params": qf_key}, obs, action),
            #         tx=optax.adam(learning_rate=args.q_lr),
            #             ),
            #             vf_state = TrainState.create(
            #     apply_fn=vf.apply,
            #     params=vf.init(actor_key, obs),
            #     tx=optax.adam(learning_rate=args.policy_lr),
            # )
            # )
        log_data = {}
        if args.update_V:
            if (not args.stable_update) or reward_gap < 0.1:
                (agentstate, _), callback_log = jax.lax.scan(
                    train_step_body,
                    (agentstate, anti_buffer_state),
                    (),
                    length=update_period,
                )
                log_data = log_data | (
                    jax.tree_map(jnp.mean, callback_log) if args.update_V else {}
                )
        agentstate, reward_callback_log = jax.lax.scan(
            reward_train_step_body,
            (agentstate),
            (),
            length=update_period,
        )
        if global_step % (2500 // update_period) == 0:
            if args.update_buffer:
                (
                    anti_buffer_state,
                    least_log_mean,
                    action_diff_log,
                    action_diff,
                ) = update_anti_actions(anti_buffer_state, agentstate.actor_state)

        log_data = log_data | jax.tree_map(jnp.mean, reward_callback_log)
        

        old_reward_vf_loss_value = reward_callback_log["reward_vf_loss_value"].mean()
        if global_step % (1000 // update_period) == 0:
            log_data = log_data | {
                "charts/SPS": int(
                    (global_step * args.update_period) / (time.time() - start_time)
                ),
                "charts/expert_log_prob": get_log_prob(
                    agentstate.actor_state.params, expert_action, expert_obs
                ),
                "charts/action_diff_log": action_diff_log,
                "charts/least_log_mean": least_log_mean,
                "charts/action_diff": action_diff,
            }
            for k, v in log_data.items():
                print(k, v, global_step)
            if args.track:
                wandb.log({"global_step": global_step * args.update_period, **log_data})

    if args.track:
        total_obs, _, total_action, _, _ = buffer.sample_fn(
            agentstate.buffer_state, key, 20000
        )
        # plt.close()
        # plot_state_whole(
        #     agentstate.vf_state, agentstate.qf_state, total_obs, total_action
        # )
        # wandb.log({"cost_plot": wandb.Image(plt)})
        # plt.close()
        # plot_vf_state_whole(agentstate.vf_state, total_obs)
        # wandb.log({"v_plot": wandb.Image(plt)})
        # plt.close()
    eval_policy(envs, agentstate, total_step * args.update_period, 100)
    state = {
        "qf_state": agentstate.qf_state,
        "vf_state": agentstate.vf_state,
        "actor_state": agentstate.actor_state,
    }
    save_args = orbax_utils.save_args_from_target(state)
    checkpointer.save(f"sacpolicy/{args.env_id}/{run_name}", state, save_args=save_args)
    # envs.close()
