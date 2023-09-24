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
import gym as ogym
import d4rl
import gymnasium as gym
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
    parser.add_argument("--env-id", type=str, default="BlockedHalfCheetah",
        help="the id of the environment to be used")
    parser.add_argument("--expert-data", type=str, default="expert_data/BlockedHalfCheetah",
        help="the name of the expert data file")
    parser.add_argument("--expert-size", type=int, default=5000)
    parser.add_argument("--decay-cycle", type=float, default=10,
        help="the reward alpha parameter")
    parser.add_argument("--total-timesteps", type=int, default=100000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=512,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--expert-batch-size", type=int, default=64,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=5e3,
        help="timestep to start learning")
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
    parser.add_argument("--lamb", type=float, default=0.8,
        help="entropy regularization coefficient")
    parser.add_argument("--reg-mult", type=float, default=1,
        help="entropy regularization coefficient")
    parser.add_argument('--update-V', type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=False,)
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=False,
        help="automatic tuning of the entropy coefficient")
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


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
    log_std_max: float = 2

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
    key: jax.random.KeyArray,
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
    subkey: jax.random.KeyArray,
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
    action = action.clip(-1 + 1e-4, 1 - 1e-4)
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


def scale_action(action_space: gym.spaces.Box, action: np.ndarray) -> np.ndarray:
    """
    Rescale the action from [low, high] to [-1, 1]
    (no need for symmetric action space)

    :param action: Action to scale
    :return: Scaled action
    """
    low, high = action_space.low, action_space.high
    return 2.0 * ((action - low) / (high - low)) - 1.0


def unscale_action(
    action_space: gym.spaces.Box, scaled_action: np.ndarray
) -> np.ndarray:
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


@jax.jit
def get_log_prob(params, actions, observations):
    mean, log_std = actor.apply(params, observations)
    log_prob = action_log_prob(actions, mean, log_std)
    return log_prob.mean()


@checkify.checkify
def train_step(agent_state):
    callback_log = {}

    def update_reward_critic(
        qf_state: RLTrainState,
        vf_state: RLTrainState,
        Policy_Batch: BatchData,
        Expert_Batch: BatchData,
        key: jax.random.KeyArray,
    ):
        key, subkey = jax.random.split(key, 2)
        policy_next_values = vf.apply(
            vf_state.params, Policy_Batch.next_observations
        ).reshape(-1)

        def mse_loss(params):
            current_policy_Q = qf.apply(
                params, Policy_Batch.observations, Policy_Batch.actions
            ).reshape(-1)
            current_expert_Q = qf.apply(
                params, Expert_Batch.observations, Expert_Batch.actions
            ).reshape(-1)
            discounted_rewards = -1 / args.reg_mult + (
                1 - Policy_Batch.dones
            ) * args.gamma * (policy_next_values)
            loss = ((current_policy_Q - discounted_rewards) ** 2).mean() + (
                (current_expert_Q - Q_max) ** 2
            ).mean()
            return loss, current_policy_Q.mean()

        (qf_loss_value, qf_values), grads = jax.value_and_grad(mse_loss, has_aux=True)(
            qf_state.params
        )
        qf_state = qf_state.apply_gradients(grads=grads)
        callback_log["qf_loss_value"] = qf_loss_value
        callback_log["qf_values"] = qf_values
        return qf_state, key

    def update_reward_value_critic(
        qf_state: RLTrainState,
        vf_state: RLTrainState,
        Policy_Batch: BatchData,
        Expert_Batch: BatchData,
        key: jax.random.KeyArray,
    ):
        Batch = jax.tree_map(
            lambda x, y: jnp.concatenate([x, y]), Policy_Batch, Expert_Batch
        )
        current_reward_Q = qf.apply(
            qf_state.target_params, Batch.observations, Batch.actions
        ).reshape(-1)
        lamb = args.lamb

        def mse_loss(params):
            current_reward_V = vf.apply(params, Batch.observations).reshape(-1)
            y = current_reward_Q - current_reward_V

            loss = (1 - lamb) * (current_reward_V).mean() + lamb * (
                (y + y**2 / 2).clip(0)
            ).mean()
            return loss, (current_reward_V.mean(), current_reward_Q.mean())

        (
            vf_loss_value,
            (vf_values, qf_values),
        ), grads = jax.value_and_grad(mse_loss, has_aux=True)(vf_state.params)
        vf_state = vf_state.apply_gradients(grads=grads)
        callback_log["vf_loss_value"] = vf_loss_value
        callback_log["vf_values"] = vf_values
        callback_log["qf_values"] = qf_values
        return vf_state, key

    def update_actor(
        actor_state: RLTrainState,
        qf_state: RLTrainState,
        vf_state: RLTrainState,
        Policy_Batch: BatchData,
        Expert_Batch: BatchData,
        key: jax.random.KeyArray,
    ):
        Batch = jax.tree_map(
            lambda x, y: jnp.concatenate([x, y]), Policy_Batch, Expert_Batch
        )
        current_reward_V = vf.apply(vf_state.params, Batch.observations).reshape(-1)
        current_reward_Q = qf.apply(
            qf_state.target_params, Batch.observations, Batch.actions
        ).reshape(-1)

        key, subkey = jax.random.split(key, 2)

        def actor_loss(params):
            mean, log_std = actor.apply(params, Batch.observations)
            log_prob = action_log_prob(Batch.actions, mean, log_std)
            y = current_reward_Q - current_reward_V
            w = jnp.exp(y*10).clip(max=100.)
            # actor_loss = -(y.clip(0) * log_prob).mean()
            actor_loss = -(w * log_prob).mean()
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
        key,
    ) = agent_state
    sample_key, key = jax.random.split(key, 2)
    observations, next_observations, actions, rewards, dones = buffer.sample_fn(
        buffer_state, sample_key, args.batch_size
    )
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
        jnp.zeros_like(rewards),
        expert_dones[expert_indice],
    )

    qf_state, key = update_reward_critic(
        qf_state,
        vf_state,
        Policy_Batch,
        Expert_Batch,
        key,
    )
    vf_state, key = update_reward_value_critic(
        qf_state,
        vf_state,
        Policy_Batch,
        Expert_Batch,
        key,
    )
    actor_state = update_actor(
        actor_state,
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
        "key",
    ],
)


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


def eval_policy(envs, agentstate,round=10):
    reward_deque = []
    cost_deque = []
    length_deque = []
    cum_reward = 0
    cum_cost = 0
    lengths = 0
    obs, info = envs.reset()
    key=agentstate.key
    for i in range(10):
        while True:
            obs = obs.astype(np.float32)
            actions, key = sample_action(
                        actor, agentstate.actor_state, obs, key
                    )
            input_actions = unscale_action(
                        envs.action_space, actions.__array__()
                    )
            (next_obs, rewards, dones, truncated, infos) = envs.step(
                        input_actions
                    )
            if hasattr(env_module, 'reward_function'):
                rewards=env_module.reward_function(next_obs, rewards, dones, truncated, infos)
            costs = cost_function(next_obs, rewards, dones, truncated, infos)
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
                (obs, rewards, dones, truncated, infos) = envs.step(
                            input_actions
                        )
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
if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time()*10)}"
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

    Q_max = 1.0 / (args.reg_mult * (1 - args.gamma))
    Q_min = -1.0 / (args.reg_mult * (1 - args.gamma))

    # Create networks
    key, actor_key, qf_key, ent_key = jax.random.split(key, 4)

    obs = jnp.array([envs.observation_space.sample()]).astype(jnp.float32)
    action = jnp.array([envs.action_space.sample()]).astype(jnp.float32)

    buffer = dejax.uniform_replay(max_size=args.buffer_size)
    buffer_state = buffer.init_fn(
        (obs[0], obs[0], action[0], jnp.array(0.0, dtype=jnp.float32), jnp.array(True))
    )

    actor = Actor(action_dim=np.prod(envs.action_space.shape))

    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs),
        tx=optax.adam(learning_rate=args.policy_lr),
    )

    reward_qf = VectorCritic(n_critics=args.n_critics)

    qf_state = RLTrainState.create(
        apply_fn=reward_qf.apply,
        params=reward_qf.init({"params": qf_key}, obs, action),
        target_params=reward_qf.init({"params": qf_key}, obs, action),
        tx=optax.adam(learning_rate=args.q_lr),
    )

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    state = {"qf_state": qf_state, "actor_state": actor_state}
    _expert_data = checkpointer.restore(args.expert_data)
    if args.expert_size!=0:
        random_indice = np.random.permutation(len(_expert_data["traj_obs"]))[
            : args.expert_size
        ]
        expert_data = jax.tree_map(lambda x: x[random_indice], _expert_data)
    else:
        expert_data =_expert_data
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
        tx=optax.adam(learning_rate=args.policy_lr),
    )

    qf = VectorCritic(n_critics=1)

    qf_state = RLTrainState.create(
        apply_fn=qf.apply,
        params=qf.init({"params": qf_key}, obs, action),
        target_params=qf.init({"params": qf_key}, obs, action),
        tx=optax.adam(learning_rate=args.q_lr),
    )

    qf = VectorCritic(n_critics=1)

    # Automatic entropy tuning
    # if args.autotune:
    #     ent_coef = EntropyCoef(ent_coef_init=1.0)
    #     target_entropy = -np.prod(envs.action_space.shape).astype(np.float32)
    #     ent_coef_state = TrainState.create(
    #         apply_fn=ent_coef.apply,
    #         params=ent_coef.init(ent_key)["params"],
    #         tx=optax.adam(learning_rate=args.q_lr),
    #     )

    # else:
    #     ent_coef_value = jnp.array(args.alpha)

    # Define update functions here to limit the need for static argname

    # @jax.jit
    # def update_temperature(ent_coef_state: TrainState, entropy: float):
    #     def temperature_loss(params):
    #         ent_coef_value = ent_coef.apply({"params": params})
    #         ent_coef_loss = ent_coef_value * (entropy - target_entropy).mean()
    #         return ent_coef_loss

    #     ent_coef_loss, grads = jax.value_and_grad(temperature_loss)(
    #         ent_coef_state.params
    #     )
    #     ent_coef_state = ent_coef_state.apply_gradients(grads=grads)

    #     return ent_coef_state, ent_coef_loss

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

    reward_deque = collections.deque(maxlen=10)
    cost_deque = collections.deque(maxlen=10)
    length_deque = collections.deque(maxlen=10)

    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset()
    obs = obs.astype(np.float32)
    # Display progress bar if available
    agentstate = AgentState(
        buffer_state,
        actor_state,
        vf_state,
        qf_state,
        key,
    )
    dataset = env_module.offline_dataset()
    agentstate = agentstate._replace(
        buffer_state=buffer.add_batch_fn(
            agentstate.buffer_state,
            (
                dataset["observations"][..., : obs.shape[-1]],
                dataset["next_observations"][..., : obs.shape[-1]],
                dataset["actions"],
                dataset["rewards"],
                dataset["terminals"] * (1 - dataset["timeouts"]),
            ),
        )
    )
    agentstate = agentstate._replace(
        buffer_state=buffer.add_batch_fn(
            agentstate.buffer_state,
            split_data(_expert_data),
        )
    )

    _train_step = jax.jit(train_step)
    agentstate = agentstate._replace(
        actor_state=TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, obs),
            tx=optax.adam(learning_rate=args.policy_lr),
        )
    )

    def train_step_body(carry, step):
        agentstate = carry
        _, (agentstate, callback_log) = _train_step(agentstate)
        return agentstate, callback_log

    for global_step in range(int((args.total_timesteps - args.learning_starts) / 50)):
        if global_step % 1000 == 0:
            eval_policy(envs, agentstate)


        agentstate, callback_log = jax.lax.scan(
            train_step_body,
            agentstate,
            (),
            length=50,
        )
        if global_step%10 == 0:
            log_data = {
                "charts/SPS": int(global_step / (time.time() - start_time)),
            }
            log_data = jax.tree_map(jnp.mean, callback_log) | log_data
            for k, v in log_data.items():
                print(k, v, global_step)
            if args.track:
                wandb.log({"global_step": global_step, **log_data})
    eval_policy(envs, agentstate,100)
    state = {
        "qf_state": agentstate.qf_state,
        "vf_state": agentstate.vf_state,
        "actor_state": agentstate.actor_state,
    }
    save_args = orbax_utils.save_args_from_target(state)
    checkpointer.save(f"sacpolicy/{args.env_id}/{run_name}", state, save_args=save_args)
