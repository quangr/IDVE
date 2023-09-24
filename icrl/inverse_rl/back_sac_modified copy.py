# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
# Implementation adapted from https://github.com/araffin/sbx
import argparse
from copy import deepcopy
import os

# os.environ["JAX_PLATFORM_NAME"] = "cpu"
import random
import time
from icrl.common.Lagrange import MultiLagrange
from distutils.util import strtobool
from functools import partial
from typing import Sequence
from jax.experimental import checkify
import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import orbax_utils
import orbax.checkpoint
import collections

# import pybullet_envs  # noqa
from flax.training.train_state import TrainState
import dejax

from icrl.env.cost_env import make_cost_env_no_xla
from icrl.inverse_rl.ERE import ERE_replay

# Add progress bar if available
try:
    from tqdm.rich import tqdm
except ImportError:
    tqdm = None


@flax.struct.dataclass
class BatchData:
    observations: jnp.ndarray
    actions: jnp.ndarray
    next_observations: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ICRL",
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
    parser.add_argument("--env-id", type=str, default="BiasedReacher",
        help="the id of the environment")
    parser.add_argument("--expert-state-path", type=str, default="sacpolicy/Reacher-expert",
        help="the id of the environment")
    parser.add_argument("--expert-data", type=str, default="expert_data",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.5,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=5e3,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=1e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=5e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--n-critics", type=int, default=2,
        help="the number of critic networks")
    parser.add_argument("--policy-frequency", type=int, default=1,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--alpha", type=float, default=0.001,
        help="entropy regularization coefficient")
    parser.add_argument("--reward-limit", type=float, default=-6,
        help="entropy regularization coefficient")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=False,
        help="automatic tuning of the entropy coefficient")
    args = parser.parse_args()
    # fmt: on
    return args


class Dual(flax.struct.PyTreeNode):
    state: TrainState = None

    @classmethod
    def create(cls, init_value):
        dual = jnp.array(init_value)

        trainstate = TrainState.create(
            apply_fn=None,
            params=dual,
            tx=optax.adam(learning_rate=0.02),
        )

        return cls(
            state=trainstate,
        )

    @staticmethod
    def _estep_dual_loss(params, cost_q, reward_q, K):
        eta, lamb = params
        _estep_kl = 0.02
        loss = eta * _estep_kl
        combined_q = reward_q - lamb * cost_q  # (B, K)
        loss += lamb * (0.01)
        loss += (
            eta
            * (
                jax.scipy.special.logsumexp(combined_q / (eta), axis=0) - jnp.log(K)
            ).mean()
        )
        return loss

    def update_dual_multiplier(self, cost_q, reward_q, K):
        loss_value, grads = jax.value_and_grad(self._estep_dual_loss)(
            self.state.params, cost_q, reward_q, K
        )
        state = self.state.apply_gradients(grads=grads)
        return self.replace(
            state=state.replace(params=jnp.clip(state.params, 1e-6, 20))
        )


class Critic(nn.Module):
    n_units: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([x, action], -1)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
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
    gaussian_action = mean + action_std * jax.random.normal(subkey, shape=mean.shape,dtype=jnp.float32)
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


@jax.jit
@checkify.checkify
def train_step(
    cost_buffer_state,
    buffer_state,
    qf_state,
    reward_qf_state,
    dual,
    policy_lag,
    actor_state,
    old_actor_state,
    key,
    step
):
    sample_key, key = jax.random.split(key, 2)
    (
        observations,
        next_observations,
        actions,
        rewards,
        costs,
        dones,
    ) = cost_buffer.sample_fn(cost_buffer_state, sample_key, args.batch_size)
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
        jnp.zeros_like(rewards),
        expert_dones[expert_indice],
    )
    qf_state, (qf_loss_value, qf_values), key = update_critic(
        actor_state,
        qf_state,
        Policy_Batch,
        Expert_Batch,
        key,
    )
    observations, next_observations, actions, rewards, costs, dones = buffer.sample_fn(
        buffer_state, sample_key, args.batch_size,step
    )
    sample_key, key = jax.random.split(key, 2)
    Policy_Batch = BatchData(
        observations,
        actions,
        next_observations,
        rewards,
        dones,
    )
    reward_qf_state, (reward_qf_loss_value, reward_qf_values), key = update_eval_critic(
        actor_state,
        reward_qf_state,
        Policy_Batch,
        key,
    )
    old_mean, old_log_std, actions, update_q, dual = E_step(
        old_actor_state, qf_state, dual, reward_qf_state, observations, key
    )
    (actor_state, policy_lag, actor_loss_value, key, entropy, kl) = update_actor(
        actor_state,
        old_mean,
        old_log_std,
        actions,
        update_q,
        policy_lag,
        observations,
        key,
    )
    qf_state = soft_update(args.tau, qf_state)
    reward_qf_state = soft_update(args.tau, reward_qf_state)

    return (
        actor_state,
        qf_state,
        reward_qf_state,
        dual,
        qf_loss_value,
        qf_values,
        reward_qf_loss_value,
        reward_qf_values,
        actor_loss_value,
        key,
        policy_lag,
        kl,
    )

@jax.jit
@checkify.checkify
def train_step_cost(
    cost_buffer_state,
    buffer_state,
    qf_state,
    reward_qf_state,
    dual,
    policy_lag,
    actor_state,
    old_actor_state,
    key,
    step
):
    sample_key, key = jax.random.split(key, 2)
    jax.random.randint(
        sample_key, minval=0, maxval=len(expert_obs), shape=(args.batch_size,)
    )
    observations, next_observations, actions, rewards, costs, dones = buffer.sample_fn(
    buffer_state, sample_key, args.batch_size,step
    )
    Policy_Batch = BatchData(
        observations,
        actions,
        next_observations,
        rewards,
        dones,
    )
    sample_key, key = jax.random.split(key, 2)

    qf_state, (qf_loss_value, qf_values), key = update_cost_critic(
        actor_state,
        old_actor_state,
        qf_state,
        Policy_Batch,
        key,
    )
    reward_qf_state, (reward_qf_loss_value, reward_qf_values), key = update_eval_critic(
        actor_state,
        reward_qf_state,
        Policy_Batch,
        key,
    )
    old_mean, old_log_std, actions, update_q, dual = E_step(
        old_actor_state, qf_state, dual, reward_qf_state, observations, key
    )
    (actor_state, policy_lag, actor_loss_value, key, entropy, kl) = update_actor(
        actor_state,
        old_mean,
        old_log_std,
        actions,
        update_q,
        policy_lag,
        observations,
        key,
    )
    qf_state = soft_update(args.tau, qf_state)
    reward_qf_state = soft_update(args.tau, reward_qf_state)

    return (
        actor_state,
        qf_state,
        reward_qf_state,
        dual,
        qf_loss_value,
        qf_values,
        reward_qf_loss_value,
        reward_qf_values,
        actor_loss_value,
        key,
        policy_lag,
        kl,
    )


@jax.jit
@checkify.checkify
def half_train_step(buffer_state, reward_qf_state, actor_state, key):
    sample_key, key = jax.random.split(key, 2)
    observations, next_observations, actions, rewards, costs, dones = buffer.sample_fn(
        buffer_state, sample_key, args.batch_size,0
    )
    Policy_Batch = BatchData(
        observations,
        actions,
        next_observations,
        rewards,
        dones,
    )

    reward_qf_state, (reward_qf_loss_value, reward_qf_values), key = update_eval_critic(
        actor_state,
        reward_qf_state,
        Policy_Batch,
        key,
    )
    reward_qf_state = soft_update(0.4, reward_qf_state)

    return actor_state, reward_qf_state, reward_qf_loss_value, reward_qf_values, key


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

    reg_mult = 5
    Q_max = 1.0 / (reg_mult * (1 - args.gamma))
    Q_min = -1.0 / (reg_mult * (1 - args.gamma))
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    # Use a separate key, so running with/without eval doesn't affect the results

    # wrappers=[]
    envs, cost_function = make_cost_env_no_xla(args.env_id, 1, args.seed)
    max_episode_steps = envs.config["max_episode_steps"]
    reward_threshold = envs.config['reward_threshold']

    reward_threshold = (
        reward_threshold
        * (1 - args.gamma**max_episode_steps)
        / (max_episode_steps * (1 - args.gamma))
    )

    reward_limit = (
        args.reward_limit
        * (1 - args.gamma**max_episode_steps)
        / (max_episode_steps * (1 - args.gamma))
    )
    print("q_limit", reward_limit)
    # Create networks
    key, actor_key, qf_key, ent_key = jax.random.split(key, 4)

    obs = jnp.array([envs.observation_space.sample()]).astype(jnp.float32)
    action = jnp.array([envs.action_space.sample()]).astype(jnp.float32)
    train_round=60
    buffer = ERE_replay(max_size=args.buffer_size,K=train_round)
    buffer_state = buffer.init_fn(
        (
            obs[0],
            obs[0],
            action[0],
            jnp.array([0.0], dtype=jnp.float32),
            jnp.array([0.0], dtype=jnp.float32),
            jnp.array([True]),
        )
    )

    cost_buffer = dejax.uniform_replay(max_size=args.buffer_size)
    cost_buffer_state = cost_buffer.init_fn(
        (
            obs[0],
            obs[0],
            action[0],
            jnp.array([0.0], dtype=jnp.float32),
            jnp.array([0.0], dtype=jnp.float32),
            jnp.array([True]),
        )
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
    state = checkpointer.restore(args.expert_state_path, item=state)
    reward_qf_state = state["qf_state"]
    actor_state = state["actor_state"]
    expert_data = checkpointer.restore(args.expert_data)
    expert_action = jnp.concatenate(expert_data["action"]).astype(jnp.float32)
    expert_obs = jnp.concatenate(expert_data["traj"][:-1]).astype(jnp.float32)
    expert_next_obs = jnp.concatenate(expert_data["traj"][1:]).astype(jnp.float32)
    expert_dones = jnp.concatenate([jnp.zeros(len(expert_obs) - 500), jnp.ones(500)])

    qf = VectorCritic(n_critics=1)

    qf_state = RLTrainState.create(
        apply_fn=qf.apply,
        params=qf.init({"params": qf_key}, obs, action),
        target_params=qf.init({"params": qf_key}, obs, action),
        tx=optax.adam(learning_rate=args.q_lr),
    )

    reward_qf_state = RLTrainState.create(
        apply_fn=qf.apply,
        params=qf.init({"params": qf_key}, obs, action),
        target_params=qf.init({"params": qf_key}, obs, action),
        tx=optax.adam(learning_rate=args.q_lr),
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
    def update_critic(
        actor_state: TrainState,
        qf_state: RLTrainState,
        Policy_Batch: BatchData,
        Expert_Batch: BatchData,
        key: jax.random.KeyArray,
    ):
        key, subkey = jax.random.split(key, 2)

        key1, key2 = jax.random.split(subkey, 2)
        expert_next_values = get_V(
            actor_state, qf_state.target_params, Expert_Batch.next_observations, key1
        ).clip(Q_min, Q_max)
        policy_next_values = get_V(
            actor_state, qf_state.target_params, Policy_Batch.next_observations, key2
        ).clip(Q_min, Q_max)
        key1, key2 = jax.random.split(key1, 2)

        def mse_loss(params):
            expert_current_Q = qf.apply(
                params, Expert_Batch.observations, Expert_Batch.actions
            )
            policy_current_Q = qf.apply(
                params, Policy_Batch.observations, Policy_Batch.actions
            )
            expert_values = get_V(actor_state, params, Expert_Batch.observations, key1)
            policy_values = get_V(actor_state, params, Policy_Batch.observations, key2)
            reward = expert_values - (
                (1 - Expert_Batch.dones).reshape(-1, 1)
                * args.gamma
                * expert_next_values
            )
            reward1 = policy_current_Q - (
                (1 - Policy_Batch.dones).reshape(-1, 1)
                * args.gamma
                * policy_next_values
            )
            loss = ((expert_current_Q - Q_max) ** 2).mean()
            loss += (
                (
                    expert_values
                    - (1 - Expert_Batch.dones).reshape(-1, 1)
                    * args.gamma
                    * expert_next_values
                ).mean()
                + (
                    policy_values
                    - (1 - Policy_Batch.dones).reshape(-1, 1)
                    * args.gamma
                    * policy_next_values
                ).mean()
            ) / 2.0
            chi2_loss = (
                (1 - args.gamma)
                * reg_mult
                * (jnp.concatenate([reward, reward1]) ** 2).mean()
            )
            sim_loss = 0.01 * ((reward1 * Policy_Batch.rewards).mean()) ** 2
            loss += chi2_loss + sim_loss
            return loss, jnp.concatenate([expert_current_Q, policy_current_Q]).mean()

        (qf_loss_value, qf_values), grads = jax.value_and_grad(mse_loss, has_aux=True)(
            qf_state.params
        )
        qf_state = qf_state.apply_gradients(grads=grads)

        return (
            qf_state,
            (qf_loss_value, qf_values),
            key,
        )

    @jax.jit
    def update_eval_critic(
        actor_state: TrainState,
        qf_state: RLTrainState,
        Policy_Batch: BatchData,
        key: jax.random.KeyArray,
    ):
        key, subkey = jax.random.split(key, 2)
        mean, log_std = actor.apply(actor_state.params, Policy_Batch.next_observations)
        next_state_actions, next_log_prob = sample_action_and_log_prob(
            mean, log_std, subkey
        )

        qf_next_values = qf.apply(
            qf_state.target_params, Policy_Batch.next_observations, next_state_actions
        ).clip(max=reward_threshold)
        next_q_values = jnp.min(qf_next_values, axis=0)
        # td error + entropy term
        next_q_values = next_q_values - 0.001 * next_log_prob.reshape(-1, 1)
        # shape is (batch_size, 1)
        target_q_values = (
            Policy_Batch.rewards.reshape(-1, 1)
            + (1 - Policy_Batch.dones.reshape(-1, 1)) * args.gamma * next_q_values
        ).clip(max=reward_threshold)

        def mse_loss(params):
            # shape is (n_critics, batch_size, 1)
            current_q_values = qf.apply(
                params, Policy_Batch.observations, Policy_Batch.actions
            )
            # mean over the batch and then sum for each critic
            critic_loss = (
                0.5 * ((target_q_values - current_q_values) ** 2).mean(axis=1).sum()
            )
            return critic_loss, current_q_values.mean()

        (qf_loss_value, qf_values), grads = jax.value_and_grad(mse_loss, has_aux=True)(
            qf_state.params
        )
        qf_state = qf_state.apply_gradients(grads=grads)

        return (
            qf_state,
            (qf_loss_value, qf_values),
            key,
        )


    @jax.jit
    def update_cost_critic(
        actor_state: TrainState,
        old_actor_state: TrainState,
        qf_state: RLTrainState,
        Policy_Batch: BatchData,
        key: jax.random.KeyArray,
    ):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        mean, log_std = actor.apply(actor_state.params, Policy_Batch.next_observations)
        next_state_actions, next_log_prob = sample_action_and_log_prob(
            mean, log_std, subkey1
        )
        qf_next_values = qf.apply(
            qf_state.target_params, Policy_Batch.next_observations, next_state_actions
        ).clip(max=reward_threshold)
        next_q_values = jnp.min(qf_next_values, axis=0)
        next_q_values_new = next_q_values - 0.001 * next_log_prob.reshape(-1, 1)

        mean, log_std = actor.apply(old_actor_state.params, Policy_Batch.next_observations)
        next_state_actions, next_log_prob = sample_action_and_log_prob(
            mean, log_std, subkey1
        )
        qf_next_values = qf.apply(
            qf_state.target_params, Policy_Batch.next_observations, next_state_actions
        ).clip(max=reward_threshold)
        next_q_values = jnp.min(qf_next_values, axis=0)
        next_q_values_old = next_q_values - 0.001 * next_log_prob.reshape(-1, 1)


        q_values = qf.apply(
            qf_state.params, Policy_Batch.observations, Policy_Batch.actions
        )
        target_q_values = (
            q_values
            + (1 - Policy_Batch.dones.reshape(-1, 1)) * args.gamma *  (next_q_values_new-next_q_values_old)
        ).clip(Q_min,Q_max)

        def mse_loss(params):
            # shape is (n_critics, batch_size, 1)
            current_q_values = qf.apply(
                params, Policy_Batch.observations, Policy_Batch.actions
            )
            # mean over the batch and then sum for each critic
            critic_loss = (
                0.5 * ((target_q_values - current_q_values) ** 2).mean(axis=1).sum()
            )
            return critic_loss, current_q_values.mean()

        (qf_loss_value, qf_values), grads = jax.value_and_grad(mse_loss, has_aux=True)(
            qf_state.params
        )
        qf_state = qf_state.apply_gradients(grads=grads)

        return (
            qf_state,
            (qf_loss_value, qf_values),
            key,
        )



    @jax.jit
    def action_log_prob(
        action: jnp.ndarray,
        mean: jnp.ndarray,
        log_std: jnp.ndarray,
    ):
        action_std = jnp.exp(log_std)
        gaussian_action = jnp.arctanh(action)
        log_prob = (
            -0.5 * ((gaussian_action - mean) / action_std) ** 2
            - 0.5 * jnp.log(2.0 * jnp.pi)
            - log_std
        )
        log_prob -= jnp.log((1 - action**2) + 1e-6)
        return log_prob.sum(-1)

    def gaussian_kl(mu_old, log_std_old, mu, log_std):
        """Decoupled KL between two multivariate Gaussians with diagonal covariance.

        See https://arxiv.org/pdf/1812.02256.pdf Sec. 4.2.1 for details. kl_mu = KL(
        pi(mu_old, std_old) || pi(mu, std_old) ) kl_std = KL( pi(mu_old, std_old) ||
        pi(mu_old, std) )

        :param mu_old: (B, n)
        :param mu: (B, n)
        :param std_old: (B, n)
        :param std: (B, n)
        :return: kl_mu, kl_std: scalar mean and covariance terms of the KL
        """
        std_old = jnp.exp(log_std_old)
        std = jnp.exp(log_std)
        var_old, var = std_old**2, std**2
        # for numerical stability
        var_old = var_old.clip(1e-6)
        var = var.clip(1e-6)

        # note, this kl's demoninator is the old var rather than the new var
        kl_mu = 0.5 * (mu_old - mu) ** 2 / var_old
        kl_mu = jnp.sum(kl_mu, axis=-1).mean()

        kl_std = 0.5 * (jnp.log(var / var_old) + var_old / var - 1)
        kl_std = jnp.sum(kl_std, axis=-1).mean()  # Sum over the dimensions

        return kl_mu, kl_std

    dual = Dual.create(jnp.array([1.0, 0.1]))

    def update_actor(
        actor_state: RLTrainState,
        old_mean,
        old_log_std,
        actions,
        update_q,
        policy_lag: MultiLagrange,
        observations: jnp.ndarray,
        key: jax.random.KeyArray,
    ):
        key, subkey = jax.random.split(key, 2)
        # E step
        dual_mu = policy_lag.state.params[0]
        dual_std = policy_lag.state.params[1]

        def actor_loss(params):
            mean, log_std = actor.apply(params, observations)
            log_prob = action_log_prob(actions, old_mean, log_std) + action_log_prob(
                actions, mean, old_log_std
            )
            loss_mle = -(log_prob * update_q).mean()
            kl_mu, kl_std = gaussian_kl(old_mean, old_log_std, mean, log_std)
            actor_loss = (
                loss_mle
                + dual_mu * (kl_mu - 0.005).clip(0)
                + dual_std * (kl_std - 0.0005).clip(0)
            )
            return actor_loss, (-log_prob.mean(), kl_mu, kl_std)

        (actor_loss_value, (entropy, kl_mu, kl_std)), grads = jax.value_and_grad(
            actor_loss, has_aux=True
        )(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)

        policy_lag = policy_lag.update_lagrange_multiplier(
            jnp.array([(kl_mu - 0.005), (kl_std - 0.0005)])
        )

        return actor_state, policy_lag, actor_loss_value, key, entropy, (kl_mu, kl_std)

    @jax.jit
    def get_q_action_and_log_prob(
        actor_state, qf_state, reward_qf_state, observations, key
    ):
        old_mean, old_log_std = actor.apply(actor_state.params, observations)
        K = 16
        keys = jax.random.split(key, K)
        actions, log_probs = jax.vmap(
            partial(sample_action_and_log_prob, old_mean, old_log_std)
        )(keys)
        extend_obs = jnp.repeat(jnp.expand_dims(observations, 0), K, axis=0)
        cost_q = qf.apply(qf_state.params, extend_obs, actions)
        reward_q = qf.apply(reward_qf_state.params, extend_obs, actions)
        cost_q = jnp.min(cost_q, axis=0)
        reward_q = jnp.min(reward_q, axis=0)
        return old_mean, old_log_std, K, actions, cost_q, reward_q

    @jax.jit
    def E_step(actor_state, qf_state, dual, reward_qf_state, observations, key):
        old_mean, old_log_std, K, actions, cost_q, reward_q = get_q_action_and_log_prob(
            actor_state, qf_state, reward_qf_state, observations, key
        )
        # with jax.disable_jit():
        cost_q = -cost_q
        dual = dual.update_dual_multiplier(cost_q=cost_q, reward_q=reward_q, K=K)
        # result = pg.run(w_init,cost_q=cost_q,reward_q=reward_q,K=K,hyperparams_proj=(1e-6,20))

        # result=lbfgsb.run(w_init, bounds=bounds,cost_q=cost_q,reward_q=-reward_q,K=K)
        eta = dual.state.params[0]
        lamb = dual.state.params[1]
        update_q = jax.nn.softmax(((reward_q - lamb * cost_q) / eta).squeeze(), axis=0)
        return old_mean, old_log_std, actions.clip(-1 + 1e-2, 1 - 1e-2), update_q, dual

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

    @partial(jax.jit, donate_argnums=(0))
    def rollout(buffer_state, obs, next_obs, actions, rewards, costs, dones):
        rewards = jnp.nan_to_num(rewards.astype(jnp.float32), neginf=0, posinf=0)
        buffer_state = buffer.add_fn(
            buffer_state, (obs, next_obs, actions, rewards, costs, dones)
        )
        return buffer_state

    @partial(jax.jit, donate_argnums=(0, 1))
    def rollout_cost(
        buffer_state, cost_buffer_state, obs, next_obs, actions, rewards, costs, dones
    ):
        rewards = jnp.nan_to_num(rewards.astype(jnp.float32), neginf=0, posinf=0)
        buffer_state = buffer.add_fn(
            buffer_state, (obs, next_obs, actions, rewards, costs, dones)
        )
        cost_buffer_state = cost_buffer.add_fn(
            cost_buffer_state, (obs, next_obs, actions, rewards, costs, dones)
        )
        return buffer_state, cost_buffer_state

    start_time = time.time()

    reward_deque = collections.deque(maxlen=50)
    cost_deque = collections.deque(maxlen=50)

    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset()
    obs = obs.astype(np.float32)
    # Display progress bar if available
    generator = tqdm(range(int(args.learning_starts)))
    policy_lag = MultiLagrange.create(jnp.ones(2) * 0, 1., 0.0, 0.001, 2)
    old_actor_state = deepcopy(actor_state)
    cum_reward = 0
    cum_cost = 0
    for global_step in generator:
        actions, key = sample_action(actor, actor_state, obs, key)
        input_actions = unscale_action(envs.action_space, actions.__array__())
        (next_obs, rewards, dones, truncated, infos) = envs.step(input_actions)
        costs = cost_function(next_obs, rewards, dones, truncated, infos)
        next_obs = next_obs.astype(np.float32)

        buffer_state, cost_buffer_state = rollout_cost(
            buffer_state,
            cost_buffer_state,
            obs[0],
            next_obs[0],
            actions[0],
            rewards,
            costs,
            dones * (1 - truncated).astype(bool),
        )
        obs = next_obs
        cum_reward += rewards
        cum_cost += costs
        if dones.any() or truncated.any():
            reward_deque.append(cum_reward)
            cost_deque.append(cum_cost)
            cum_reward = 0
            cum_cost = 0
            (obs, rewards, dones, truncated, infos) = envs.step(input_actions)
            obs = obs.astype(np.float32)



    for i in range(20000):
        _, (
            actor_state,
            reward_qf_state,
            reward_qf_loss_value,
            reward_qf_values,
            key,
        ) = half_train_step(buffer_state, reward_qf_state, actor_state, key)

        # ALGO LOGIC: training.

    @jax.jit
    def train_step_body(carry, step):
        (
            cost_buffer_state,
            buffer_state,
            qf_state,
            reward_qf_state,
            dual,
            policy_lag,
            actor_state,
            old_actor_state,
            key,
        ) = carry
        _, (
            actor_state,
            qf_state,
            reward_qf_state,
            dual,
            qf_loss_value,
            qf_values,
            reward_qf_loss_value,
            reward_qf_values,
            actor_loss_value,
            key,
            policy_lag,
            kl,
        ) = train_step(
            cost_buffer_state,
            buffer_state,
            qf_state,
            reward_qf_state,
            dual,
            policy_lag,
            actor_state,
            old_actor_state,
            key,
            step
        )
        return (
            cost_buffer_state,
            buffer_state,
            qf_state,
            reward_qf_state,
            dual,
            policy_lag,
            actor_state,
            old_actor_state,
            key,
        ), (
            qf_loss_value.mean(),
            qf_values,
            reward_qf_loss_value,
            reward_qf_values.mean(),
            actor_loss_value,
            kl,
            dual.state.params,
        )

    def speeduptrain(
        cost_buffer_state,
        buffer_state,
        qf_state,
        reward_qf_state,
        dual,
        policy_lag,
        actor_state,
        old_actor_state,
        key,
    ):
        (
            cost_buffer_state,
            buffer_state,
            qf_state,
            reward_qf_state,
            dual,
            policy_lag,
            actor_state,
            old_actor_state,
            key,
        ), (
            qf_loss_value,
            qf_values,
            reward_qf_loss_value,
            reward_qf_values,
            actor_loss_value,
            kl,
            dual_data,
        ) = jax.lax.scan(
            train_step_body,
            (
                cost_buffer_state,
                buffer_state,
                qf_state,
                reward_qf_state,
                dual,
                policy_lag,
                actor_state,
                old_actor_state,
                key,
            ),
            jnp.arange(train_round)
        )
        return (
            cost_buffer_state,
            buffer_state,
            qf_state,
            reward_qf_state,
            dual,
            policy_lag,
            actor_state,
            old_actor_state,
            key,
        ), (
            qf_loss_value,
            qf_values,
            reward_qf_loss_value,
            reward_qf_values,
            actor_loss_value,
            kl,
            dual_data,
        )


    def train_step_cost_body(carry, step):
        (
            cost_buffer_state,
            buffer_state,
            qf_state,
            reward_qf_state,
            dual,
            policy_lag,
            actor_state,
            old_actor_state,
            key,
        ) = carry
        _, (
            actor_state,
            qf_state,
            reward_qf_state,
            dual,
            qf_loss_value,
            qf_values,
            reward_qf_loss_value,
            reward_qf_values,
            actor_loss_value,
            key,
            policy_lag,
            kl,
        ) = train_step_cost(
            cost_buffer_state,
            buffer_state,
            qf_state,
            reward_qf_state,
            dual,
            policy_lag,
            actor_state,
            old_actor_state,
            key,
            step
        )
        return (
            cost_buffer_state,
            buffer_state,
            qf_state,
            reward_qf_state,
            dual,
            policy_lag,
            actor_state,
            old_actor_state,
            key,
        ), (
            qf_loss_value.mean(),
            qf_values,
            reward_qf_loss_value,
            reward_qf_values.mean(),
            actor_loss_value,
            kl,
            dual.state.params,
        )

    def speeduptrain_cost(
        cost_buffer_state,
        buffer_state,
        qf_state,
        reward_qf_state,
        dual,
        policy_lag,
        actor_state,
        old_actor_state,
        key,
    ):
        (
            cost_buffer_state,
            buffer_state,
            qf_state,
            reward_qf_state,
            dual,
            policy_lag,
            actor_state,
            old_actor_state,
            key,
        ), (
            qf_loss_value,
            qf_values,
            reward_qf_loss_value,
            reward_qf_values,
            actor_loss_value,
            kl,
            dual_data,
        ) = jax.lax.scan(
            train_step_cost_body,
            (
                cost_buffer_state,
                buffer_state,
                qf_state,
                reward_qf_state,
                dual,
                policy_lag,
                actor_state,
                old_actor_state,
                key,
            ),
            jnp.arange(train_round)
        )
        return (
            cost_buffer_state,
            buffer_state,
            qf_state,
            reward_qf_state,
            dual,
            policy_lag,
            actor_state,
            old_actor_state,
            key,
        ), (
            qf_loss_value,
            qf_values,
            reward_qf_loss_value,
            reward_qf_values,
            actor_loss_value,
            kl,
            dual_data,
        )


    _speeduptrain=speeduptrain
    for global_step in range(int((args.total_timesteps - args.learning_starts) / 50)):

        for i in range(50):
            actions, key = sample_action(actor, actor_state, obs, key)
            # Rescale to proper domain when using squashing
            input_actions = unscale_action(envs.action_space, actions.__array__())
            # Part 2
            (next_obs, rewards, dones, truncated, infos) = envs.step(input_actions)
            costs = cost_function(next_obs, rewards, dones, truncated, infos)
            next_obs = next_obs.astype(np.float32)

            if reward_qf_values.mean() > reward_limit:
                _speeduptrain=speeduptrain
                buffer_state, cost_buffer_state = rollout_cost(
                    buffer_state,
                    cost_buffer_state,
                    obs[0],
                    next_obs[0],
                    actions[0],
                    rewards,
                    costs,
                    dones * (1 - truncated).astype(bool),
                )
            else:
                _speeduptrain=speeduptrain_cost
                buffer_state = rollout(
                    buffer_state,
                    obs[0],
                    next_obs[0],
                    actions[0],
                    rewards,
                    costs,
                    dones * (1 - truncated).astype(bool),
                )

            obs = next_obs
            cum_reward += rewards
            cum_cost += costs
            if dones.any() or truncated.any():
                reward_deque.append(cum_reward)
                cost_deque.append(cum_cost)
                cum_reward = 0
                cum_cost = 0
                (obs, rewards, dones, truncated, infos) = envs.step(input_actions)
                obs = obs.astype(np.float32)

        (
            cost_buffer_state,
            buffer_state,
            qf_state,
            reward_qf_state,
            dual,
            policy_lag,
            actor_state,
            old_actor_state,
            key,
        ), (
            qf_loss_value,
            qf_values,
            reward_qf_loss_value,
            reward_qf_values,
            actor_loss_value,
            kl,
            dual_data,
        ) = _speeduptrain(
            cost_buffer_state,
            buffer_state,
            qf_state,
            reward_qf_state,
            dual,
            policy_lag,
            actor_state,
            old_actor_state,
            key,
        )
        # (cost_buffer_state,buffer_state, qf_state, reward_qf_state, dual, policy_lag, actor_state, old_actor_state, key), (qf_loss_value, qf_values, reward_qf_loss_value, reward_qf_values, actor_loss_value, kl,dual_data) = speeduptrain(cost_buffer_state,buffer_state, qf_state, reward_qf_state, dual, policy_lag, actor_state, old_actor_state, key)
        if np.average(reward_deque) < -5000 or np.isnan(np.average(reward_deque)):
            import debugpy

            # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
            debugpy.listen(5678)
            print("Waiting for debugger attach")
            debugpy.wait_for_client()
            debugpy.breakpoint()
            print("break on this line")
        old_actor_state = actor_state
        eta, lamb = dual_data.mean(0)
        kl_mu = kl[0].mean()
        kl_std = kl[1].mean()
        log_data = {
            "losses/reward_qf_values": reward_qf_values.mean().item(),
            "losses/reward_qf_loss": reward_qf_loss_value.mean().item(),
            "losses/qf_values": qf_values.mean().item(),
            "losses/qf_loss": qf_loss_value.mean().item(),
            "losses/actor_loss": actor_loss_value.mean().item(),
            "losses/alpha": ent_coef_value.mean().item(),
            "charts/SPS": int(global_step / (time.time() - start_time)),
            "charts/mean_return": np.average(reward_deque),
            "charts/mean_cost": np.average(cost_deque),
            "charts/cost_buffer_size": cost_buffer.size_fn(cost_buffer_state),
            "charts/eta": eta,
            "charts/lamb": lamb,
            "charts/kl_mu": kl_mu,
            "charts/kl_std": kl_std,
        }

        # Print the data
        for k, v in log_data.items():
            print(k, v, global_step)

        # Log the data if args.track is True
        if args.track:
            # Assuming you have imported the required libraries (e.g., wandb)
            wandb.log({"global_step": global_step, **log_data})

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    state = {"qf_state": qf_state, "actor_state": actor_state}
    save_args = orbax_utils.save_args_from_target(state)
    checkpointer.save(f"sacpolicy/{time.time()}", state, save_args=save_args)
    # envs.close()
